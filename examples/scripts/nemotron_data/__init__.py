# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Self-contained data plumbing for the Nemotron-Image-Training-v3 VQA dataset.

`jsonl_index`:
    Memory-efficient byte-offset index over a large JSONL manifest, with an
    optional `/dev/shm`/`/tmp` node-local copy mode to avoid NFS contention
    when multiple ranks share a node.

`nemotron_media`:
    Image resolver for Nemotron-Image-Training-v3 (loose files, shared image
    pools, WebDataset shards, GeomVerse) with a wall-clock timeout to keep
    NFS hangs from stalling DDP, plus a `warmup_media_indexes()` helper to
    pre-build tar indexes before training starts.

Origin: ported from Pengfei's qvac-edge-vlm nanoVLM repo, kept self-contained
inside TRL so the example script doesn't depend on an external path.
"""

from .jsonl_index import IndexedJsonlDataset, JsonlIndex
from .nemotron_media import (
    DEFAULT_ROOT,
    NemotronImageResolver,
    warmup_media_indexes,
)


__all__ = [
    "DEFAULT_ROOT",
    "IndexedJsonlDataset",
    "JsonlIndex",
    "NemotronImageResolver",
    "warmup_media_indexes",
]
