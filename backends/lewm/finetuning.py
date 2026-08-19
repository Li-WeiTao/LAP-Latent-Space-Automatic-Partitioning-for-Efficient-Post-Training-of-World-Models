"""Exact FP32 predictor-only optimizer used by the LeWM LAP backend."""

from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from lap.interfaces import (
    EncodedTransitions,
    PredictorTrainingResult,
    RegionalTrainingConfig,
)

from .adapter import LeWMBackend
from .cache import LeWMCachedPayload


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


@dataclass
class LeWMTrainConfig:
    history_size: int = 3
    num_preds: int = 1
    frameskip: int = 0
    img_size: int = 224
    train_fraction: float = 0.9
    split_seed: int = 3072
    seed: int = 42
    max_starts: int = 0
    min_region_samples: int = 256
    batch_size: int = 128
    epochs: int = 30
    lr: float = 5e-5
    weight_decay: float = 1e-3


def set_training_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_encoder_path(model: torch.nn.Module) -> None:
    for name in ("encoder", "projector", "action_encoder"):
        module = getattr(model, name, None)
        if module is not None:
            module.requires_grad_(False)
            module.eval()


def unfreeze_predictor_path(model: torch.nn.Module) -> None:
    for name in ("predictor", "pred_proj"):
        module = getattr(model, name, None)
        if module is not None:
            module.requires_grad_(True)
            module.train()


def trainable_predictor_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for name in ("predictor", "pred_proj"):
        module = getattr(model, name, None)
        if module is not None:
            params.extend(param for param in module.parameters() if param.requires_grad)
    return params


def predictor_loss(
    model: torch.nn.Module,
    emb: torch.Tensor,
    act_emb: torch.Tensor,
    history_size: int,
    num_preds: int,
) -> torch.Tensor:
    ctx_emb = emb[:, :history_size]
    ctx_act = act_emb[:, :history_size]
    tgt_emb = emb[:, num_preds:]
    pred_emb = model.predict(ctx_emb, ctx_act)
    return (pred_emb - tgt_emb).pow(2).mean()


@torch.no_grad()
def eval_predictor_loss(
    model: torch.nn.Module,
    emb: torch.Tensor,
    act_emb: torch.Tensor,
    cfg: LeWMTrainConfig,
    device: torch.device,
) -> float:
    model.eval()
    loader = DataLoader(
        TensorDataset(emb, act_emb),
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )
    total_sse = 0.0
    total_elems = 0
    for batch_emb, batch_act in loader:
        batch_emb = batch_emb.to(device)
        batch_act = batch_act.to(device)
        ctx_emb = batch_emb[:, : cfg.history_size]
        ctx_act = batch_act[:, : cfg.history_size]
        tgt_emb = batch_emb[:, cfg.num_preds :]
        pred_emb = model.predict(ctx_emb, ctx_act)
        total_sse += float((pred_emb - tgt_emb).pow(2).sum().detach().cpu())
        total_elems += tgt_emb.numel()
    return total_sse / max(total_elems, 1)


def _predictor_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name in ("predictor", "pred_proj"):
        module = getattr(model, name, None)
        if module is not None:
            state.update(
                {
                    f"{name}.{key}": value.detach().cpu().clone()
                    for key, value in module.state_dict().items()
                }
            )
    return state


def _load_predictor_state_dict(
    model: torch.nn.Module, state: dict[str, torch.Tensor]
) -> None:
    for name in ("predictor", "pred_proj"):
        module = getattr(model, name, None)
        if module is None:
            continue
        prefix = f"{name}."
        subset = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if subset:
            module.load_state_dict(subset)


def save_region_predictor(
    model: torch.nn.Module, path: Path, metadata: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, path)
    with path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(metadata), handle, indent=2)


def train_region_predictor(
    base_model: torch.nn.Module,
    emb: torch.Tensor,
    act_emb: torch.Tensor,
    cfg: LeWMTrainConfig,
    device: torch.device,
    *,
    save_epochs: list[int] | None = None,
    checkpoint_dir: Path | None = None,
    region: str | None = None,
    name_prefix: str = "",
    select_best_by_eval: bool = False,
    eval_each_epoch: bool = True,
    run_final_eval: bool = True,
    sync_batch_loss: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Train with the exact historical ordering, loss, and selection rule."""

    model = copy.deepcopy(base_model).to(device)
    freeze_encoder_path(model)
    unfreeze_predictor_path(model)
    params = trainable_predictor_params(model)
    if not params:
        raise RuntimeError("No trainable predictor parameters found")

    set_training_seed(cfg.seed)
    loader = DataLoader(
        TensorDataset(emb, act_emb),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=len(emb) >= cfg.batch_size,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    optimizer = torch.optim.AdamW(
        params, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    save_epochs_set = set(save_epochs or [])
    saved_checkpoints: dict[int, str] = {}
    best_epoch = 0
    best_eval_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []

    for epoch in range(cfg.epochs):
        epoch_losses: list[float] = []
        for batch_emb, batch_act in loader:
            batch_emb = batch_emb.to(device)
            batch_act = batch_act.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = predictor_loss(
                model, batch_emb, batch_act, cfg.history_size, cfg.num_preds
            )
            loss.backward()
            optimizer.step()
            if sync_batch_loss:
                epoch_losses.append(float(loss.detach().cpu()))
        epoch_no = epoch + 1
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        if eval_each_epoch:
            eval_loss = eval_predictor_loss(model, emb, act_emb, cfg, device)
        else:
            eval_loss = float("nan")
        history.append(
            {"epoch": epoch_no, "loss": train_loss, "eval_loss": eval_loss}
        )
        print(
            f"  [train] epoch {epoch_no}/{cfg.epochs}  "
            f"train_loss={train_loss:.6f}  eval_loss={eval_loss:.6f}",
            flush=True,
        )
        if select_best_by_eval and eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_epoch = epoch_no
            best_state = _predictor_state_dict(model)
        if epoch_no in save_epochs_set and checkpoint_dir is not None and region:
            checkpoint = (
                checkpoint_dir
                / f"P_{name_prefix}{region}_epoch{epoch_no}_object.ckpt"
            )
            save_region_predictor(
                model,
                checkpoint,
                {
                    "region": region,
                    "epoch": epoch_no,
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                },
            )
            saved_checkpoints[epoch_no] = str(checkpoint)

    if select_best_by_eval and best_state is not None:
        _load_predictor_state_dict(model, best_state)
        final_loss = best_eval_loss
        print(
            f"  [best] epoch {best_epoch}  eval_loss={best_eval_loss:.6f} "
            f"(selected over {cfg.epochs} epochs)",
            flush=True,
        )
    elif run_final_eval:
        final_loss = eval_predictor_loss(model, emb, act_emb, cfg, device)
        best_epoch = cfg.epochs
        best_eval_loss = final_loss
    else:
        final_loss = float("nan")
        best_epoch = cfg.epochs
        best_eval_loss = float("nan")
    return model, {
        "epochs": cfg.epochs,
        "final_loss": final_loss,
        "best_epoch": best_epoch,
        "best_eval_loss": best_eval_loss,
        "select_best_by_eval": select_best_by_eval,
        "saved_checkpoints": saved_checkpoints,
        "history": history,
    }


def prepare_region_predictor_training(
    base_model: torch.nn.Module,
    emb: torch.Tensor,
    act_emb: torch.Tensor,
    cfg: LeWMTrainConfig,
    device: torch.device,
) -> tuple[torch.nn.Module, DataLoader, torch.optim.AdamW]:
    """Construct a region predictor and optimizer outside epoch timing."""
    model = copy.deepcopy(base_model).to(device)
    freeze_encoder_path(model)
    unfreeze_predictor_path(model)
    params = trainable_predictor_params(model)
    if not params:
        raise RuntimeError("No trainable predictor parameters found")
    set_training_seed(cfg.seed)
    loader = DataLoader(
        TensorDataset(emb, act_emb),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=len(emb) >= cfg.batch_size,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    optimizer = torch.optim.AdamW(
        params, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    return model, loader, optimizer


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_predictor_epochs_timed(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: LeWMTrainConfig,
    device: torch.device,
    *,
    epochs: int,
) -> list[dict[str, Any]]:
    """Time pure predictor batch loops with symmetric CUDA synchronization."""
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(epochs):
        _sync_device(device)
        epoch_t0 = time.perf_counter()
        epoch_steps = 0
        for batch_emb, batch_act in loader:
            batch_emb = batch_emb.to(device)
            batch_act = batch_act.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = predictor_loss(
                model, batch_emb, batch_act, cfg.history_size, cfg.num_preds
            )
            loss.backward()
            optimizer.step()
            epoch_steps += 1
        _sync_device(device)
        elapsed = time.perf_counter() - epoch_t0
        epoch_rows.append(
            {
                "epoch": epoch + 1,
                "epoch_wall_sec": elapsed,
                "optimizer_steps": epoch_steps,
            }
        )
    return epoch_rows


class LeWMRegionalPredictorTrainer:
    """Bridge the generic regional trainer protocol to exact LeWM FP32 FT."""

    def __init__(
        self,
        device: torch.device,
        *,
        select_best_by_eval: bool = True,
        eval_each_epoch: bool = True,
        benchmark_pure_epochs: bool = False,
    ):
        self.device = device
        self.select_best_by_eval = select_best_by_eval
        self.eval_each_epoch = eval_each_epoch
        self.benchmark_pure_epochs = benchmark_pure_epochs

    def fit_region(
        self,
        backend: LeWMBackend,
        region_id: int,
        transitions: EncodedTransitions,
        config: RegionalTrainingConfig,
    ) -> PredictorTrainingResult:
        if not isinstance(transitions.payload, LeWMCachedPayload):
            raise TypeError("LeWM trainer requires LeWMCachedPayload")
        cfg = LeWMTrainConfig(
            history_size=int(config.options.get("history_size", 3)),
            num_preds=int(config.options.get("num_preds", 1)),
            seed=config.train_seed,
            min_region_samples=config.min_region_samples,
            batch_size=config.batch_size,
            epochs=config.epochs,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        if self.benchmark_pure_epochs:
            model, loader, optimizer = prepare_region_predictor_training(
                backend.model,
                transitions.payload.emb,
                transitions.payload.act_emb,
                cfg,
                self.device,
            )
            epoch_rows = train_predictor_epochs_timed(
                model,
                loader,
                optimizer,
                cfg,
                self.device,
                epochs=config.epochs,
            )
            return PredictorTrainingResult(
                predictor=model,
                metrics={
                    "epochs": config.epochs,
                    "benchmark_pure_epochs": True,
                    "epoch_timings": epoch_rows,
                },
            )
        model, metrics = train_region_predictor(
            backend.model,
            transitions.payload.emb,
            transitions.payload.act_emb,
            cfg,
            self.device,
            region=f"cluster{region_id}",
            name_prefix="train_",
            select_best_by_eval=self.select_best_by_eval,
            eval_each_epoch=self.eval_each_epoch,
            run_final_eval=not self.benchmark_pure_epochs,
            sync_batch_loss=not self.benchmark_pure_epochs,
        )
        return PredictorTrainingResult(predictor=model, metrics=metrics)
