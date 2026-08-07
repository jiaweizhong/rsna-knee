from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from .common import atomic_text_writer, json_dumps, stable_shard


def _iter_files(root: Path, include_hidden: bool = False) -> Iterable[Path]:
    for directory, dir_names, file_names in os.walk(root):
        dir_names[:] = sorted(
            name for name in dir_names if include_hidden or not name.startswith(".")
        )
        for name in sorted(file_names):
            if not include_hidden and name.startswith("."):
                continue
            yield Path(directory) / name


def build_file_index(
    dicom_root: str | Path,
    output_root: str | Path,
    num_shards: int = 128,
    include_hidden: bool = False,
    force: bool = False,
) -> dict[str, object]:
    root = Path(dicom_root).resolve()
    output = Path(output_root)
    index_dir = output / "index"
    done_path = index_dir / "index.done.json"
    if done_path.exists() and not force:
        state = json.loads(done_path.read_text(encoding="utf-8"))
        if Path(state["dicom_root"]).resolve() != root:
            raise ValueError("Existing index points at a different DICOM root; use --force")
        if int(state["num_shards"]) != num_shards:
            raise ValueError("Existing index uses a different shard count; use --force")
        return state

    if not root.is_dir():
        raise FileNotFoundError(f"DICOM root does not exist: {root}")

    shard_dir = index_dir / "shard_paths"
    shard_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths = [shard_dir / f"part-{i:05d}.txt.tmp" for i in range(num_shards)]
    handles = [path.open("w", encoding="utf-8", newline="\n") for path in temporary_paths]
    counts = [0] * num_shards
    total_files = 0
    total_bytes = 0
    inventory_path = index_dir / "files.jsonl"

    try:
        with atomic_text_writer(inventory_path) as inventory:
            for path in _iter_files(root, include_hidden=include_hidden):
                relative = path.relative_to(root).as_posix()
                stat = path.stat()
                shard = stable_shard(relative, num_shards)
                handles[shard].write(relative + "\n")
                counts[shard] += 1
                total_files += 1
                total_bytes += stat.st_size
                inventory.write(
                    json_dumps(
                        {
                            "relative_path": relative,
                            "size_bytes": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns,
                            "shard": shard,
                        }
                    )
                    + "\n"
                )
    finally:
        for handle in handles:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

    for shard, temporary in enumerate(temporary_paths):
        temporary.replace(shard_dir / f"part-{shard:05d}.txt")

    state: dict[str, object] = {
        "dicom_root": str(root),
        "num_shards": num_shards,
        "total_files": total_files,
        "total_bytes": total_bytes,
        "shard_counts": counts,
    }
    with atomic_text_writer(done_path) as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
    return state

