from __future__ import annotations

import json
import random
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import PKADataset, collate_pka
from .metrics import regression_metrics
from .model import LiTENJointSitePKA, move_batch_to_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(
    table_path: str | Path,
    manifest_path: str | Path,
    data_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    shuffle: bool = False,
    conformer_mode: str = "all",
) -> DataLoader:
    dataset = PKADataset(
        table_path,
        manifest_path,
        id_column=data_cfg["id_column"],
        target_column=data_cfg["target_column"],
        center_column=data_cfg["center_column"],
        task_column=data_cfg["task_column"],
        smiles_column=data_cfg["smiles_column"],
        atom_feature_set=model_cfg.get("atom_feature_set", "none"),
        include_center_feature=bool(model_cfg.get("atom_feature_include_center", False)),
    )
    return DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=partial(collate_pka, conformer_mode=conformer_mode),
    )


@torch.no_grad()
def evaluate(model: LiTENJointSitePKA, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    ys: list[float] = []
    preds: list[float] = []
    site_correct_count = 0
    site_count = 0
    mol_ids: list[str] = []
    pred_centers: list[int] = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        pred = out["pka_pred"]
        y = batch["y"]
        ys.extend(y.detach().cpu().numpy().tolist())
        preds.extend(pred.detach().cpu().numpy().tolist())
        site_correct_count += int(out["site_correct_count"].item())
        site_count += int(out["site_count"].item())
        mol_ids.extend(batch["mol_ids"])
        pred_centers.extend(out["pred_center_idx"].detach().cpu().numpy().tolist())
    metrics = regression_metrics(np.asarray(ys), np.asarray(preds))
    metrics["site_acc"] = site_correct_count / site_count if site_count else float("nan")
    metrics["site_correct_count"] = site_correct_count
    metrics["site_count"] = site_count
    pred_df = pd.DataFrame({"mol_id": mol_ids, "y_true": ys, "y_pred": preds, "pred_center_idx": pred_centers})
    return metrics, pred_df


def train_one_seed(config: dict[str, Any], seed: int, experiment_name: str = "joint-site-pka") -> dict[str, Any]:
    """Train a joint ionization-site and pKa model using train/validation data only."""
    set_seed(seed)
    device = torch.device(config["project"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    train_cfg = config["train"]
    model_cfg = config["model"]
    manifest_path = Path(data_cfg["conformer_dir"]) / "manifest.csv"
    site_loss_weight = float(model_cfg.get("site_loss_weight", 1.0))

    train_loader = build_loader(
        data_cfg["train_file"],
        manifest_path,
        data_cfg,
        train_cfg,
        model_cfg,
        shuffle=True,
        conformer_mode=str(train_cfg.get("train_conformer_mode", "random_one")),
    )
    val_loader = build_loader(data_cfg["val_file"], manifest_path, data_cfg, train_cfg, model_cfg)

    model = LiTENJointSitePKA(config["liten"], model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    ckpt_dir = Path(train_cfg["checkpoint_dir"]) / experiment_name / f"seed_{seed}"
    report_dir = Path(train_cfg["report_dir"]) / experiment_name / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = -1
    patience = int(train_cfg.get("patience", 20))
    keep_top_k = int(train_cfg.get("keep_top_k", 1))
    save_last = bool(train_cfg.get("save_last", True))
    top_checkpoints: list[dict[str, Any]] = []
    history_path = report_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        losses: list[float] = []
        pka_losses: list[float] = []
        site_losses: list[float] = []
        site_correct_count = 0
        site_count = 0
        for batch in tqdm(train_loader, desc=f"{experiment_name}/seed{seed}/epoch{epoch}"):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            pka_loss = F.smooth_l1_loss(out["pka_pred"], batch["y"])
            site_loss = out["site_loss"]
            loss = pka_loss + site_loss_weight * site_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
            pka_losses.append(float(pka_loss.item()))
            site_losses.append(float(site_loss.item()))
            site_correct_count += int(out["site_correct_count"].item())
            site_count += int(out["site_count"].item())

        val_metrics, val_pred = evaluate(model, val_loader, device)
        report = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_pka_loss": float(np.mean(pka_losses)),
            "train_site_loss": float(np.mean(site_losses)),
            "train_site_acc": site_correct_count / site_count if site_count else float("nan"),
            "train_site_correct_count": site_correct_count,
            "train_site_count": site_count,
            "val_metrics": val_metrics,
        }
        history_path.open("a", encoding="utf-8").write(json.dumps(report) + "\n")
        payload = {"model": model.state_dict(), "config": config, "seed": seed, "epoch": epoch, **report}
        if save_last:
            torch.save(payload, ckpt_dir / "last.pt")

        val_mae = float(val_metrics["mae"])
        if keep_top_k > 0:
            top_path = ckpt_dir / f"top_epoch_{epoch:03d}_val_mae_{val_mae:.4f}.pt"
            torch.save(payload, top_path)
            top_checkpoints.append({"epoch": epoch, "val_mae": val_mae, "checkpoint": str(top_path)})
            top_checkpoints.sort(key=lambda item: item["val_mae"])
            while len(top_checkpoints) > keep_top_k:
                removed = top_checkpoints.pop()
                Path(removed["checkpoint"]).unlink(missing_ok=True)
            (report_dir / "top_checkpoints.json").write_text(json.dumps(top_checkpoints, indent=2), encoding="utf-8")

        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            torch.save(payload, ckpt_dir / "best.pt")
            val_pred.to_csv(report_dir / "val_predictions.csv", index=False)
        elif epoch - best_epoch >= patience:
            break

    summary = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_mae": best_val,
        "checkpoint": str(ckpt_dir / "best.pt"),
        "last_checkpoint": str(ckpt_dir / "last.pt") if save_last else "",
        "top_checkpoints": top_checkpoints,
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
