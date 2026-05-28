"""Unified image resolution for Nemotron-Image-Training-v3 (loose files + shared pools + WDS)."""

from __future__ import annotations

import io
import re
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from functools import lru_cache
from pathlib import Path

from PIL import Image

DEFAULT_ROOT = Path("/home/pengfei/datasets/Nemotron-Image-Training-v3")

_geom_idx: dict[int, Path] | None = None
_wds_index_lock = threading.Lock()
_wds_index_cache: dict[str, dict[str, tuple[Path, str]]] = {}


def _build_geom_index(root: Path) -> dict[int, Path]:
    global _geom_idx
    if _geom_idx is not None:
        return _geom_idx
    index_lists: dict[int, list[Path]] = {}
    gv = root / "GeomVerse"
    if gv.is_dir():
        for p in gv.rglob("*.jpeg"):
            if p.parent.name.lower() != "images":
                continue
            try:
                k = int(p.stem)
            except ValueError:
                continue
            index_lists.setdefault(k, []).append(p)

    def score(path: Path) -> tuple[int, str]:
        s = str(path)
        pri = 0 if "/TRAIN/" in s else 1 if "/VAL/" in s else 2 if "/TEST/" in s else 3
        return (pri, s)

    out: dict[int, Path] = {}
    for k, cands in index_lists.items():
        out[k] = sorted(cands, key=score)[0]
    _geom_idx = out
    return out


def geom_src_for(root: Path, rel: str) -> Path | None:
    m = re.match(r"^0*(\d+)\.jpg$", rel, re.I)
    if not m:
        return None
    k = int(m.group(1))
    return _build_geom_index(root).get(k)


def candidate_paths(root: Path, config: str, rel: str) -> list[Path]:
    rel = rel.strip()
    cands: list[Path] = [
        root / config / rel,
        root / config / "media" / rel,
        root / rel,
    ]
    if "/" not in rel and rel.lower().endswith(".jpg"):
        for sub in ("train2017", "test2017", "unlabeled2017", "val2017"):
            cands.append(root / sub / rel)
        cands.append(root / "gqa_images" / rel)
        cands.append(root / "images" / rel)
    if rel.startswith("train_images/"):
        cands.append(root / "train_images" / rel.split("/", 1)[1])
    if rel.startswith("figures/"):
        cands.append(root / "figures" / rel.split("/", 1)[1])
        cands.append(root / "figures" / "figures" / Path(rel).name)
    if config == "geomverse":
        gs = geom_src_for(root, rel)
        if gs is not None:
            cands.append(gs)
    if config == "ecd" and "/" not in rel:
        cands.append(root / "ecd" / rel)
    if rel.startswith("train/data/"):
        cands.append(root / "train" / "data" / rel.split("/", 2)[-1])
    return cands


def has_webdataset_shards(root: Path, config: str) -> bool:
    md = root / config / "media"
    if not md.is_dir():
        return False
    return any(p.is_file() and p.stat().st_size > 1000 for p in md.glob("shard_*.tar"))


def _is_image_member(name: str) -> bool:
    base = name.rstrip("/").split("/")[-1]
    if not base or base.startswith("."):
        return False
    return Path(base).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def build_wds_index(root: Path, config: str) -> dict[str, tuple[Path, str]]:
    """Map image basename -> (tar_path, member_name)."""
    with _wds_index_lock:
        if config in _wds_index_cache:
            return _wds_index_cache[config]

    index: dict[str, tuple[Path, str]] = {}
    md = root / config / "media"
    if md.is_dir():
        for tar_path in sorted(md.glob("shard_*.tar")):
            try:
                with tarfile.open(tar_path, "r:*") as tf:
                    for m in tf.getmembers():
                        if not m.isfile() or not _is_image_member(m.name):
                            continue
                        base = Path(m.name.replace("\\", "/")).name
                        if base and base not in index:
                            index[base] = (tar_path, m.name)
            except (tarfile.TarError, OSError):
                continue

    _wds_index_cache[config] = index
    return index


def read_image_from_wds(root: Path, config: str, rel: str) -> Image.Image | None:
    basename = Path(rel.replace("\\", "/")).name
    if not basename:
        return None
    index = build_wds_index(root, config)
    entry = index.get(basename)
    if entry is None:
        return None
    tar_path, member_name = entry
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            extracted = tf.extractfile(member_name)
            if extracted is None:
                return None
            data = extracted.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None


class NemotronImageResolver:
    """Resolve Nemotron JSONL image paths to RGB PIL images."""

    def __init__(self, root: str | Path = DEFAULT_ROOT):
        self.root = Path(root)
        self._resolve_stats = {"loose": 0, "wds": 0, "miss": 0}

    def resolve_path(self, config: str, rel: str) -> Path | None:
        for p in candidate_paths(self.root, config, rel):
            try:
                if p.is_file():
                    return p
            except OSError:
                continue
        if has_webdataset_shards(self.root, config):
            basename = Path(rel.replace("\\", "/")).name
            if basename and basename in build_wds_index(self.root, config):
                return Path(f"wds://{config}/{basename}")
        return None

    def is_resolvable(self, config: str, rel: str) -> bool:
        return self.resolve_path(config, rel) is not None

    def open_image(self, config: str, rel: str, timeout: float = 120.0) -> Image.Image | None:
        """Open image with a wall-clock timeout so one rank cannot hang NFS/DDP."""
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(self._open_image_impl, config, rel)
            try:
                return fut.result(timeout=timeout)
            except FuturesTimeout:
                return None

    def _open_image_impl(self, config: str, rel: str) -> Image.Image | None:
        for p in candidate_paths(self.root, config, rel):
            try:
                if p.is_file():
                    self._resolve_stats["loose"] += 1
                    return Image.open(p).convert("RGB")
            except Exception:
                continue

        if has_webdataset_shards(self.root, config):
            img = read_image_from_wds(self.root, config, rel)
            if img is not None:
                self._resolve_stats["wds"] += 1
                return img

        self._resolve_stats["miss"] += 1
        return None

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._resolve_stats)

    def reset_stats(self) -> None:
        self._resolve_stats = {"loose": 0, "wds": 0, "miss": 0}


def warmup_media_indexes(root: str | Path, configs: set[str] | list[str]) -> None:
    """Build WDS / GeomVerse indexes before training so no rank stalls mid-step."""
    root = Path(root)
    for config in sorted(set(configs)):
        if has_webdataset_shards(root, config):
            build_wds_index(root, config)
    if "geomverse" in {c.lower() for c in configs}:
        _build_geom_index(root)


@lru_cache(maxsize=1)
def default_resolver() -> NemotronImageResolver:
    return NemotronImageResolver()
