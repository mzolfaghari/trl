"""Memory-efficient random access into large JSONL files."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

try:
    import orjson

    def _loads(line: bytes) -> dict:
        return orjson.loads(line)

except ImportError:

    def _loads(line: bytes) -> dict:
        return json.loads(line)


# Caller is responsible for calling `torch.distributed.init_process_group()`
# before constructing a JsonlIndex when running under DDP. The barrier below
# relies on the process group being initialized; otherwise `_is_master()`
# returns True on every rank and `_barrier()` is a no-op, which is the
# correct behavior for single-process use.
try:
    import torch.distributed as dist

    def _is_dist() -> bool:
        return dist.is_available() and dist.is_initialized()

    def _is_master() -> bool:
        return not _is_dist() or dist.get_rank() == 0

    def _barrier() -> None:
        if _is_dist():
            dist.barrier()
except Exception:

    def _is_dist() -> bool:
        return False

    def _is_master() -> bool:
        return True

    def _barrier() -> None:
        pass


_FH_BY_PID: dict[tuple[int, str], object] = {}


def _local_rank() -> int:
    val = os.environ.get("LOCAL_RANK")
    if val is not None:
        return int(val)
    return 0


def _node_id() -> str:
    return os.environ.get("SLURM_NODEID", os.environ.get("SLURMD_NODENAME", "0"))


def _free_gb(path: Path) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / (1024**3)


def _pick_cache_dir(src: Path) -> Path | None:
    need_gb = src.stat().st_size / (1024**3) + 5
    candidates: list[Path] = []
    for raw in (os.environ.get("SLURM_TMPDIR", ""), "/tmp", "/dev/shm"):
        if raw:
            candidates.append(Path(raw))
    for root in candidates:
        if not root.is_dir():
            continue
        try:
            if _free_gb(root) >= need_gb:
                cache = root / "nemotron_jsonl_cache"
                cache.mkdir(parents=True, exist_ok=True)
                return cache
        except OSError:
            continue
    return None


def _local_cache_path(src: Path) -> Path | None:
    cache_dir = _pick_cache_dir(src)
    if cache_dir is None:
        return None
    node = _node_id()
    stamp = int(src.stat().st_mtime)
    return cache_dir / f"{src.stem}_{stamp}_node{node}.jsonl"


def _read_line_seek(path: Path, offset: int) -> bytes:
    key = (os.getpid(), str(path))
    fh = _FH_BY_PID.get(key)
    if fh is None:
        fh = open(path, "rb")
        _FH_BY_PID[key] = fh
    fh.seek(offset)
    line = fh.readline()
    if not line.strip():
        raise ValueError(f"empty JSONL line at offset {offset} in {path}")
    return line


def _ensure_node_local_copy(src: Path) -> Path:
    """Copy JSONL to node-local disk once so all local ranks share page cache."""
    local = _local_cache_path(src)
    if local is None:
        if _is_master():
            print(f"[jsonl] no local cache space; reading from {src}", flush=True)
        return src

    ready = local.with_suffix(local.suffix + ".ready")
    src_mtime = src.stat().st_mtime
    src_size = src.stat().st_size

    if _local_rank() == 0:
        valid = (
            local.is_file()
            and local.stat().st_size == src_size
            and local.stat().st_mtime >= src_mtime
        )
        if not valid:
            if _is_master():
                print(f"[jsonl] copying {src.name} -> {local}", flush=True)
            tmp = local.with_suffix(".jsonl.part")
            if tmp.exists():
                tmp.unlink()
            shutil.copyfile(src, tmp)
            tmp.rename(local)
        ready.write_text("ok")

    deadline = time.time() + 3600
    while not ready.is_file():
        if time.time() > deadline:
            raise TimeoutError(f"timed out waiting for node-local JSONL: {local}")
        time.sleep(0.5)

    _barrier()
    return local


class JsonlIndex:
    """Byte-offset index for O(1) line lookup without loading the full file.

    Args:
        path: Source JSONL file path.
        node_local: If True, copy the JSONL to /dev/shm or /tmp on each node
            before indexing. Avoids NFS contention across local ranks.
        cache_dir: Directory for the `.offsets.npy` cache file. If `None`,
            defaults to `~/.cache/nemotron_jsonl_offsets/`. The cache is keyed
            by the absolute source path so multiple JSONLs don't collide.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        node_local: bool = False,
        cache_dir: str | Path | None = None,
    ):
        self.src_path = Path(path).resolve()
        self.node_local = node_local
        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None
            else Path.home() / ".cache" / "nemotron_jsonl_offsets"
        )
        self.path = _ensure_node_local_copy(self.src_path) if node_local else self.src_path
        self.offsets = self._load_offsets()

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> dict:
        line = _read_line_seek(self.path, int(self.offsets[idx]))
        return _loads(line)

    def _cache_path(self) -> Path:
        # Use just the JSONL basename — clean and readable. Different JSONLs
        # with the same basename at different paths would collide; pass an
        # explicit `cache_dir` per run to avoid that (the script does).
        return self.cache_dir / f"{self.src_path.name}.offsets.npy"

    def _load_offsets(self) -> np.ndarray:
        cache = self._cache_path()
        src_mtime = self.src_path.stat().st_mtime
        cache_valid = cache.is_file() and cache.stat().st_mtime >= src_mtime

        if not cache_valid and _is_master():
            cache.parent.mkdir(parents=True, exist_ok=True)
            offsets: list[int] = []
            pos = 0
            with open(self.src_path, "rb") as f:
                for line in f:
                    if line.strip():
                        offsets.append(pos)
                    pos += len(line)
            # Write to a sibling tmp file then atomically rename so non-master
            # ranks never observe a partial cache file. The tmp name must end
            # in `.npy` because np.save silently appends `.npy` otherwise.
            tmp = cache.parent / f"{cache.stem}.tmp{cache.suffix}"
            np.save(tmp, np.asarray(offsets, dtype=np.uint64))
            os.replace(tmp, cache)

        _barrier()

        if not cache.is_file():
            raise FileNotFoundError(f"JSONL offset cache missing after build: {cache}")
        return np.load(cache, mmap_mode="r")


class IndexedJsonlDataset:
    """Subset view over a JsonlIndex."""

    def __init__(self, index: JsonlIndex, indices: np.ndarray):
        self.index = index
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        return self.index[int(self.indices[idx])]
