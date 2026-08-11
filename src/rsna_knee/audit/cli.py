from __future__ import annotations

import argparse
import functools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

from .common import atomic_text_writer, get_or_create_patient_salt, json_dumps
from .coverage import compute_coverage
from .dicom import read_header_record
from .duplicates import compute_duplicates
from .index import build_file_index
from .pixels import read_pixel_record_sampled
from .summarize import summarize_audit


def _parse_shards(value: str | None, num_shards: int) -> list[int]:
    if not value or value.lower() == "all":
        return list(range(num_shards))
    selected: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if "-" in token:
            start, end = token.split("-", 1)
            selected.update(range(int(start), int(end) + 1))
        else:
            selected.add(int(token))
    invalid = [index for index in selected if index < 0 or index >= num_shards]
    if invalid:
        raise ValueError(f"Shard indices out of range: {invalid}")
    return sorted(selected)


def _map_records(
    worker: Callable[[str, str], dict[str, Any]],
    root: Path,
    relative_paths: Iterable[str],
    workers: int,
) -> Iterable[dict[str, Any]]:
    pairs = ((str(root / relative), relative) for relative in relative_paths)
    if workers <= 1:
        for absolute, relative in pairs:
            yield worker(absolute, relative)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(lambda_pair_worker, ((worker, pair) for pair in pairs), chunksize=16)


def lambda_pair_worker(arguments: tuple[Callable[..., dict[str, Any]], tuple[str, str]]) -> dict[str, Any]:
    """Top-level multiprocessing trampoline; the worker itself must be picklable."""
    worker, pair = arguments
    return worker(*pair)


def _load_index_state(audit_root: Path) -> dict[str, Any]:
    path = audit_root / "index" / "index.done.json"
    if not path.exists():
        raise FileNotFoundError(f"Run the index stage first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_partitioned_stage(
    audit_root: Path,
    stage: str,
    shards: str | None,
    workers: int,
    worker_factory: Callable[[int], Callable[[str, str], dict[str, Any]]],
    force: bool,
) -> None:
    state = _load_index_state(audit_root)
    root = Path(state["dicom_root"])
    num_shards = int(state["num_shards"])
    selected = _parse_shards(shards, num_shards)
    output_dir = audit_root / stage
    output_dir.mkdir(parents=True, exist_ok=True)
    for shard in selected:
        source = audit_root / "index" / "shard_paths" / f"part-{shard:05d}.txt"
        destination = output_dir / f"part-{shard:05d}.jsonl"
        if destination.exists() and not force:
            print(f"[{stage}] shard {shard:05d}: already complete")
            continue
        relative_paths = [
            line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        worker = worker_factory(shard)
        ok = 0
        errors = 0
        with atomic_text_writer(destination) as handle:
            for record in _map_records(worker, root, relative_paths, workers):
                handle.write(json_dumps(record) + "\n")
                ok += int(record.get("status") == "ok")
                errors += int(record.get("status") != "ok")
        print(f"[{stage}] shard {shard:05d}: files={len(relative_paths)} ok={ok} errors={errors}")


def _cmd_index(args: argparse.Namespace) -> None:
    state = build_file_index(
        args.dicom_root,
        args.output,
        num_shards=args.num_shards,
        include_hidden=args.include_hidden,
        force=args.force,
    )
    print(json.dumps(state, indent=2))


def _cmd_headers(args: argparse.Namespace) -> None:
    audit_root = Path(args.audit_root)
    salt = get_or_create_patient_salt(audit_root)

    def factory(_: int) -> Callable[[str, str], dict[str, Any]]:
        return functools.partial(read_header_record, patient_salt=salt, force=not args.strict)

    _run_partitioned_stage(
        audit_root, "headers", args.shards, args.workers, factory, args.force
    )


def _cmd_pixels(args: argparse.Namespace) -> None:
    audit_root = Path(args.audit_root)

    def factory(_: int) -> Callable[[str, str], dict[str, Any]]:
        return functools.partial(
            read_pixel_record_sampled,
            deep_sample_rate=args.deep_sample_rate,
            hash_pixels=args.hash_pixels,
            force=not args.strict,
        )

    _run_partitioned_stage(
        audit_root, "pixels", args.shards, args.workers, factory, args.force
    )


def _cmd_summarize(args: argparse.Namespace) -> None:
    summary = summarize_audit(
        args.audit_root,
        train_csv=args.train_csv,
        train_series_csv=args.train_series_csv,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _cmd_coverage(args: argparse.Namespace) -> None:
    summary = compute_coverage(
        args.audit_root,
        train_csv=args.train_csv,
        train_series_csv=args.train_series_csv,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _cmd_duplicates(args: argparse.Namespace) -> None:
    summary = compute_duplicates(args.audit_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scalable RSNA knee DICOM audit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Stream the filesystem into stable shards")
    index_parser.add_argument("--dicom-root", required=True)
    index_parser.add_argument("--output", required=True)
    index_parser.add_argument("--num-shards", type=int, default=128)
    index_parser.add_argument("--include-hidden", action="store_true")
    index_parser.add_argument("--force", action="store_true")
    index_parser.set_defaults(func=_cmd_index)

    for name, function in [("headers", _cmd_headers), ("pixels", _cmd_pixels)]:
        stage_parser = subparsers.add_parser(name)
        stage_parser.add_argument("--audit-root", required=True)
        stage_parser.add_argument("--shards", default="all", help="all, 0,3,5-10")
        stage_parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
        stage_parser.add_argument("--strict", action="store_true", help="Disable pydicom force mode")
        stage_parser.add_argument("--force", action="store_true", help="Rebuild completed parts")
        stage_parser.set_defaults(func=function)
    pixel_parser = subparsers.choices["pixels"]
    pixel_parser.add_argument("--deep-sample-rate", type=float, default=0.10)
    pixel_parser.add_argument("--hash-pixels", action="store_true")

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--audit-root", required=True)
    summary_parser.add_argument("--train-csv")
    summary_parser.add_argument("--train-series-csv")
    summary_parser.set_defaults(func=_cmd_summarize)

    coverage_parser = subparsers.add_parser(
        "coverage", help="Compare studies present on disk against train.csv, prioritizing gold labels"
    )
    coverage_parser.add_argument("--audit-root", required=True)
    coverage_parser.add_argument("--train-csv", required=True)
    coverage_parser.add_argument("--train-series-csv")
    coverage_parser.set_defaults(func=_cmd_coverage)

    duplicates_parser = subparsers.add_parser(
        "duplicates",
        help="Find exact-match duplicates (SOP UID reuse, identical pixel bytes, identical series "
        "signature); requires summarize to have run first",
    )
    duplicates_parser.add_argument("--audit-root", required=True)
    duplicates_parser.set_defaults(func=_cmd_duplicates)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
