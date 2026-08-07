from pathlib import Path

import yaml

from rsna_knee.config import deep_merge, load_config


def test_deep_merge_replaces_lists_and_preserves_nested_values(tmp_path: Path) -> None:
    assert deep_merge(
        {"model": {"name": "a", "params": {"k": 5}}, "values": [1, 2]},
        {"model": {"params": {"k": 10}}, "values": [3]},
    ) == {"model": {"name": "a", "params": {"k": 10}}, "values": [3]}

    base = tmp_path / "base.yaml"
    overlay = tmp_path / "overlay.yaml"
    base.write_text(yaml.safe_dump({"model": {"selector": {"k": 5}}}), encoding="utf-8")
    overlay.write_text(yaml.safe_dump({"model": {"name": "tiny"}}), encoding="utf-8")
    config = load_config([base, overlay], ["model.selector.k=15", "precision=bf16"])
    assert config["model"]["selector"]["k"] == 15
    assert config["model"]["name"] == "tiny"
    assert config["precision"] == "bf16"


def test_registry_overlay_replaces_incompatible_params() -> None:
    merged = deep_merge(
        {"model": {"backbone": {"name": "tiny_cnn", "params": {"width": 8}}}},
        {"model": {"backbone": {"name": "timm", "params": {"model_name": "vit"}}}},
    )
    assert merged["model"]["backbone"] == {
        "name": "timm",
        "params": {"model_name": "vit"},
    }
