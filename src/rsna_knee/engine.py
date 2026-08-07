from __future__ import annotations

import contextlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from rsna_knee.config import config_fingerprint, save_config
from rsna_knee.data import StudyDataset, collate_studies
from rsna_knee.losses import build_loss
from rsna_knee.metrics import multilabel_auc
from rsna_knee.models import build_model


def _distributed_state() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        distributed.init_process_group(backend=backend)
    return rank, local_rank, world_size


def _set_seed(seed: int, rank: int) -> None:
    seed = int(seed) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _make_dataset(spec: Mapping[str, Any]) -> StudyDataset:
    return StudyDataset(
        manifest_path=spec["manifest_path"],
        dicom_root=spec["dicom_root"],
        image_size=tuple(spec.get("image_size", [224, 224])),
        normalization=str(spec.get("normalization", "per_slice_robust")),
        max_candidates=int(spec.get("max_candidates", 0)),
        on_decode_error=str(spec.get("on_decode_error", "raise")),
        input_mean=spec.get("input_mean"),
        input_std=spec.get("input_std"),
    )


def _make_loader(
    dataset: StudyDataset,
    spec: Mapping[str, Any],
    training: bool,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=training, drop_last=training)
    workers = int(spec.get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_size=int(spec.get("batch_size", 2)),
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=bool(spec.get("pin_memory", True)),
        persistent_workers=workers > 0 and bool(spec.get("persistent_workers", True)),
        prefetch_factor=int(spec.get("prefetch_factor", 2)) if workers > 0 else None,
        drop_last=training,
        collate_fn=collate_studies,
    )
    return loader, sampler


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _gather_objects(value: Any, world_size: int) -> list[Any]:
    if world_size == 1:
        return [value]
    gathered: list[Any] = [None] * world_size
    distributed.all_gather_object(gathered, value)
    return gathered


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    world_size: int,
) -> dict[str, Any]:
    model.eval()
    probabilities = []
    targets = []
    study_ids: list[str] = []
    for batch in loader:
        batch = _to_device(batch, device)
        with _autocast_context(device, precision):
            output = model(batch)
        probabilities.append(torch.sigmoid(output["logits"]).float().cpu().numpy())
        targets.append(batch["labels"].float().cpu().numpy())
        study_ids.extend(batch["study_id"])
    local = {
        "probabilities": np.concatenate(probabilities) if probabilities else np.empty((0, 12)),
        "targets": np.concatenate(targets) if targets else np.empty((0, 12)),
        "study_ids": study_ids,
    }
    gathered = _gather_objects(local, world_size)
    all_probabilities = np.concatenate([item["probabilities"] for item in gathered])
    all_targets = np.concatenate([item["targets"] for item in gathered])
    all_study_ids = sum((item["study_ids"] for item in gathered), [])
    # DistributedSampler pads validation shards to equal length. Keep the first
    # occurrence so AUC is not biased by duplicated studies.
    unique_indices = []
    seen = set()
    for index, study_id in enumerate(all_study_ids):
        if study_id not in seen:
            seen.add(study_id)
            unique_indices.append(index)
    all_probabilities = all_probabilities[unique_indices]
    all_targets = all_targets[unique_indices]
    metrics = multilabel_auc(all_targets, all_probabilities)
    metrics["studies"] = int(all_targets.shape[0])
    return metrics


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "metrics": dict(metrics),
            "config": dict(config),
        },
        path,
    )


def train(config: Mapping[str, Any]) -> dict[str, Any]:
    rank, local_rank, world_size = _distributed_state()
    seed = int(config.get("seed", 2026))
    _set_seed(seed, rank)
    device = _device(local_rank)
    torch.set_float32_matmul_precision(str(config.get("matmul_precision", "high")))

    run_root = Path(config.get("output_dir", "runs"))
    run_name = str(config.get("run_name") or config_fingerprint(config))
    run_dir = run_root / run_name
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, run_dir / "config.resolved.yaml")

    train_dataset = _make_dataset(config["data"]["train"])
    valid_dataset = _make_dataset(config["data"]["valid"])
    train_loader, train_sampler = _make_loader(
        train_dataset, config["loader"]["train"], True, rank, world_size
    )
    valid_loader, _ = _make_loader(
        valid_dataset, config["loader"]["valid"], False, rank, world_size
    )

    model = build_model(config["model"])
    if bool(config["model"].get("freeze_backbone", False)):
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
    model.to(device)
    resume_state = None
    if config.get("resume_from"):
        resume_state = torch.load(config["resume_from"], map_location="cpu", weights_only=False)
        model.load_state_dict(resume_state["model"], strict=True)
    if bool(config.get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer_spec = config.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(optimizer_spec.get("lr", 3e-4)),
        weight_decay=float(optimizer_spec.get("weight_decay", 0.05)),
        betas=tuple(optimizer_spec.get("betas", [0.9, 0.999])),
    )
    epochs = int(config.get("epochs", 10))
    accumulation = int(config.get("gradient_accumulation", 1))
    total_steps = max(1, epochs * math.ceil(len(train_loader) / accumulation))
    warmup_steps = int(config.get("scheduler", {}).get("warmup_steps", 0))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start_epoch = 0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        start_epoch = int(resume_state["epoch"]) + 1
    criterion = build_loss(config.get("loss", {})).to(device)
    precision = str(config.get("precision", "bf16"))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    clip_norm = float(config.get("gradient_clip_norm", 1.0))

    best_auc = -math.inf
    best_metrics: dict[str, Any] = {}
    global_step = 0
    metrics_path = run_dir / "metrics.jsonl"
    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        started = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            batch = _to_device(batch, device)
            with _autocast_context(device, precision):
                output = model(batch)
                loss, components = criterion(output, batch)
                loss = loss / accumulation
            scaler.scale(loss).backward()
            should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
            loss_total += float(components["total"].detach())

        metrics = evaluate(model, valid_loader, device, precision, world_size)
        metrics.update(
            {
                "epoch": epoch,
                "train_loss": loss_total / max(1, len(train_loader)),
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.perf_counter() - started,
                "global_step": global_step,
            }
        )
        if rank == 0:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            _save_checkpoint(
                run_dir / "last.pt", model, optimizer, scheduler, epoch, metrics, config
            )
            score = metrics.get("macro_auc")
            if score is not None and float(score) > best_auc:
                best_auc = float(score)
                best_metrics = dict(metrics)
                _save_checkpoint(
                    run_dir / "best.pt", model, optimizer, scheduler, epoch, metrics, config
                )
            print(json.dumps(metrics, ensure_ascii=False))
    if distributed.is_initialized():
        distributed.barrier()
    return best_metrics
