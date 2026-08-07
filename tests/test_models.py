import torch

from rsna_knee.losses import CompositeLoss
from rsna_knee.models import build_model


def _config(selector: str = "uniform", aggregator: str = "per_label_query") -> dict:
    return {
        "num_labels": 12,
        "in_channels": 3,
        "metadata_dim": 5,
        "backbone": {"name": "tiny_cnn", "params": {"out_dim": 32, "width": 8}},
        "selector": {"name": selector, "params": {"k": 3, "width": 8}},
        "aggregator": {"name": aggregator, "params": {}},
    }


def _batch() -> dict[str, torch.Tensor]:
    return {
        "images": torch.randn(2, 6, 3, 32, 32),
        "metadata": torch.randn(2, 6, 5),
        "mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool),
        "labels": torch.tensor(
            [[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]],
            dtype=torch.float32,
        ),
    }


def test_all_builtin_aggregators_forward() -> None:
    for aggregator in ("mean_max", "attention", "per_label_query"):
        model = build_model(_config(aggregator=aggregator))
        output = model(_batch())
        assert output["logits"].shape == (2, 12)
        assert output["selector_indices"].shape == (2, 3)


def test_learned_selector_coverage_has_gradient() -> None:
    batch = _batch()
    batch["window_targets"] = torch.rand(2, 6, 12)
    model = build_model(_config(selector="recall_safe_topk"))
    output = model(batch)
    loss, components = CompositeLoss(coverage_weight=1.0, selector_k=3)(output, batch)
    loss.backward()
    assert components["coverage"].item() > 0
    assert model.selector.scorer.head.weight.grad is not None


def test_nan_labels_are_ignored() -> None:
    batch = _batch()
    batch["labels"][0, :3] = float("nan")
    model = build_model(_config())
    output = model(batch)
    loss, _ = CompositeLoss()(output, batch)
    assert torch.isfinite(loss)

