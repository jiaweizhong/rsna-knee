from __future__ import annotations

import argparse
import json

from rsna_knee.config import add_config_arguments, load_config
from rsna_knee.engine import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train study-level RSNA knee classifier")
    add_config_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, args.overrides)
    result = train(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

