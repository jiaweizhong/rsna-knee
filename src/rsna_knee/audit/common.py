from __future__ import annotations

import hashlib
import json
import os
import secrets
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO


def stable_shard(value: str, num_shards: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % num_shards


def stable_fraction(value: str) -> float:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") / float(2**64 - 1)


def json_dumps(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error


@contextmanager
def atomic_text_writer(path: str | Path) -> Iterator[TextIO]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)


def get_or_create_patient_salt(output_root: str | Path) -> str:
    private_dir = Path(output_root) / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    salt_path = private_dir / "patient_salt.txt"
    if salt_path.exists():
        return salt_path.read_text(encoding="utf-8").strip()
    salt = secrets.token_hex(32)
    with atomic_text_writer(salt_path) as handle:
        handle.write(salt)
    return salt


def hash_identifier(value: str | None, salt: str) -> str | None:
    if not value:
        return None
    payload = f"{salt}\0{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def list_part_files(directory: str | Path, suffix: str = ".jsonl") -> list[Path]:
    return sorted(Path(directory).glob(f"part-*{suffix}"))

