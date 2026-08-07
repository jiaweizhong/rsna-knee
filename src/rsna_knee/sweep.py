from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import yaml

from rsna_knee.config import config_fingerprint, load_config, save_config, set_dot_path


def generate_sweep(base_paths: list[str], matrix_path: str, output_dir: str) -> list[Path]:
    base = load_config(base_paths)
    with Path(matrix_path).open("r", encoding="utf-8") as handle:
        specification = yaml.safe_load(handle) or {}
    matrix = specification.get("matrix", {})
    fixed = specification.get("fixed", {})
    keys = list(matrix)
    values = [matrix[key] for key in keys]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated = []
    queue = []
    for combination in itertools.product(*values):
        config = load_config(base_paths)
        for key, value in fixed.items():
            set_dot_path(config, key, value)
        for key, value in zip(keys, combination):
            set_dot_path(config, key, value)
        fingerprint = config_fingerprint(config)
        config["run_name"] = f"sweep-{fingerprint}"
        path = output / f"{config['run_name']}.yaml"
        save_config(config, path)
        generated.append(path)
        queue.append(
            {
                "run_name": config["run_name"],
                "config": str(path),
                "overrides": dict(zip(keys, combination)),
            }
        )
    (output / "queue.json").write_text(
        json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Cartesian experiment configurations")
    parser.add_argument("--base", action="append", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = generate_sweep(args.base, args.matrix, args.output)
    for path in paths:
        print(f"python -m rsna_knee.train --config {path}")


if __name__ == "__main__":
    main()

