from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rsna_knee.data.dataset import _decode_dicom, _normalize


def _to_image(array: np.ndarray, size: int) -> Image.Image:
    normalized = _normalize(array, "per_slice_robust")
    pixels = np.nan_to_num(normalized * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="L").resize((size, size), Image.Resampling.BILINEAR)


def _uniform(items: list[Any], count: int) -> list[Any]:
    if len(items) <= count:
        return items
    indices = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[index] for index in indices]


def render_study_montage(
    record: dict[str, Any], dicom_root: Path, output_path: Path, tile_size: int = 192
) -> None:
    slots: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for window in record.get("windows", []):
        key = (str(window.get("plane", "Unknown")), str(window.get("fluid_sensitive", "Unknown")))
        slots[key].append(window)
    rows = []
    for key in sorted(slots):
        rows.append((key, _uniform(slots[key], 9)))
    label_width = 180
    header_height = 26
    canvas = Image.new(
        "RGB",
        (label_width + 9 * tile_size, max(1, len(rows)) * (tile_size + header_height)),
        "black",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row_index, (slot, windows) in enumerate(rows):
        y = row_index * (tile_size + header_height)
        draw.text((5, y + 5), f"{slot[0]} | fluid={slot[1]}", fill="white", font=font)
        for column, window in enumerate(windows):
            path = dicom_root / window["paths"][len(window["paths"]) // 2]
            try:
                image = _to_image(_decode_dicom(path), tile_size).convert("RGB")
            except Exception as error:
                image = Image.new("RGB", (tile_size, tile_size), "darkred")
                ImageDraw.Draw(image).text((5, 5), type(error).__name__, fill="white", font=font)
            canvas.paste(image, (label_width + column * tile_size, y + header_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def generate_montages(
    manifest_path: str | Path,
    dicom_root: str | Path,
    output_dir: str | Path,
    max_studies: int = 20,
    seed: int = 2026,
    tile_size: int = 192,
) -> int:
    records = []
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    random.Random(seed).shuffle(records)
    selected = records[:max_studies]
    output = Path(output_dir)
    for record in selected:
        safe_id = str(record["study_id"]).replace(".", "_")
        render_study_montage(record, Path(dicom_root), output / f"{safe_id}.jpg", tile_size)
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render deterministic study/series audit montages")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dicom-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-studies", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tile-size", type=int, default=192)
    args = parser.parse_args()
    count = generate_montages(
        args.manifest,
        args.dicom_root,
        args.output_dir,
        args.max_studies,
        args.seed,
        args.tile_size,
    )
    print(f"Generated {count} study montages")


if __name__ == "__main__":
    main()

