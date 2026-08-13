from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import PKADataset, collate_pka
from .metrics import regression_metrics
from .model import LiTENJointSitePKA, LiTENPKA, move_batch_to_device


def prepare_prediction_table(
    table_path: str | Path,
    config: dict[str, Any],
    mode: Literal["known_ic", "joint"] = "known_ic",
) -> tuple[Path, pd.DataFrame, tempfile.TemporaryDirectory[str] | None]:
    """Return a PKADataset-compatible CSV without modifying the user's input file."""
    data_cfg = config["data"]
    id_col = data_cfg["id_column"]
    smiles_col = data_cfg["smiles_column"]
    target_col = data_cfg["target_column"]
    center_col = data_cfg["center_column"]
    task_col = data_cfg["task_column"]

    table_path = Path(table_path)
    df = pd.read_csv(table_path)
    if smiles_col not in df.columns:
        raise ValueError(f"Input CSV must contain a '{smiles_col}' column.")

    changed = False
    if id_col not in df.columns:
        df[id_col] = [f"mol_{idx:06d}" for idx in range(len(df))]
        changed = True
    if target_col not in df.columns:
        df[target_col] = np.nan
        changed = True
    if task_col not in df.columns:
        df[task_col] = "unknown"
        changed = True
    if center_col not in df.columns:
        if mode == "known_ic":
            raise ValueError(
                f"Known-IC prediction requires '{center_col}'. Use --mode joint when ionization centers are unknown."
            )
        df[center_col] = -1
        changed = True
    if mode == "known_ic" and (df[center_col].astype(float) < 0).any():
        raise ValueError(f"Known-IC prediction requires non-negative '{center_col}' for every row.")

    if not changed:
        return table_path, df, None

    tmpdir = tempfile.TemporaryDirectory(prefix="liten_pka_predict_")
    prepared = Path(tmpdir.name) / "input_prepared.csv"
    df.to_csv(prepared, index=False)
    return prepared, df, tmpdir


def _build_dataset(
    table_path: str | Path,
    manifest_path: str | Path,
    config: dict[str, Any],
) -> PKADataset:
    data_cfg = config["data"]
    model_cfg = config.get("model", {})
    return PKADataset(
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


@torch.no_grad()
def predict_table(
    checkpoint: str | Path,
    table_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    batch_size: int = 8,
    device: str = "cuda",
    mode: Literal["known_ic", "joint"] = "known_ic",
) -> pd.DataFrame:
    state = torch.load(checkpoint, map_location="cpu")
    config: dict[str, Any] = state["config"]
    prepared_table, input_table, tmpdir = prepare_prediction_table(table_path, config, mode=mode)
    try:
        dataset = _build_dataset(prepared_table, manifest_path, config)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pka)
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        if mode == "known_ic":
            model = LiTENPKA(config["liten"], config["model"]).to(dev)
        elif mode == "joint":
            model = LiTENJointSitePKA(config["liten"], config["model"]).to(dev)
        else:
            raise ValueError(f"Unknown prediction mode: {mode}")
        model.load_state_dict(state["model"], strict=True)
        model.eval()

        rows: list[dict[str, Any]] = []
        for batch in loader:
            batch = move_batch_to_device(batch, dev)
            if mode == "known_ic":
                pred = model(batch)
                pred_center = [None] * len(batch["mol_ids"])
            else:
                out = model(batch)
                pred = out["pka_pred"]
                pred_center = out["pred_center_idx"].detach().cpu().numpy().tolist()
            y = batch["y"].detach().cpu().numpy().tolist()
            pred_values = pred.detach().cpu().numpy().tolist()
            for mol_id, yy, pp, pc in zip(batch["mol_ids"], y, pred_values, pred_center):
                row = {"mol_id": mol_id, "y_true": yy, "y_pred": pp}
                if pc is not None:
                    row["pred_center_idx"] = int(pc)
                rows.append(row)

        out_df = pd.DataFrame(rows)
        meta_cols = [
            col
            for col in ["mol_id", "smiles", "task", "center_atom_idx", "center_symbol", "source_name", "source_dataset"]
            if col in input_table.columns
        ]
        if meta_cols:
            out_df = out_df.merge(input_table[meta_cols].drop_duplicates(), on="mol_id", how="left")
        if out_df["y_true"].notna().all():
            metrics = regression_metrics(out_df["y_true"].to_numpy(), out_df["y_pred"].to_numpy())
            for key, value in metrics.items():
                out_df.attrs[key] = value
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_path, index=False)
        return out_df
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()


def summarize_micro_to_macro(predictions: pd.DataFrame, task_column: str = "task") -> pd.DataFrame:
    if task_column not in predictions.columns:
        return predictions
    rows = []
    for mol_id, group in predictions.groupby("mol_id"):
        acid = group[group[task_column].str.lower() == "acid"] if task_column in group else pd.DataFrame()
        base = group[group[task_column].str.lower() == "base"] if task_column in group else pd.DataFrame()
        if len(acid):
            chosen = acid.sort_values("y_pred").iloc[0].to_dict()
            chosen["macro_rule"] = "acid_min"
            rows.append(chosen)
        if len(base):
            chosen = base.sort_values("y_pred", ascending=False).iloc[0].to_dict()
            chosen["macro_rule"] = "base_max"
            rows.append(chosen)
    return pd.DataFrame(rows)
