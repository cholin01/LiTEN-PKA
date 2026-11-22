from __future__ import annotations

import argparse
import json
import hashlib
import math
import re
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "liten-pka-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from liten_pka.config import load_config
from liten_pka.dataset import PKADataset, collate_pka
from liten_pka.model import LiTENPKA, move_batch_to_device
from liten_pka.train import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pretrained LiTEN fine-tuning and scratch training in parameter and embedding spaces."
    )
    parser.add_argument("--pretrained-ckpt", required=True, help="Fine-tuned checkpoint initialized from LiTEN_SPICE.")
    parser.add_argument("--scratch-ckpt", required=True, help="Scratch-trained checkpoint with the same architecture.")
    parser.add_argument("--spice-config", default="configs/pkalearn_liten.yaml")
    parser.add_argument("--spice-checkpoint", default="", help="Override the original SPICE reference weight path.")
    parser.add_argument("--scratch-config", default="configs/pkalearn_liten_scratch.yaml")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", default="results/figure5/data/interpretability")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pair-sample", type=int, default=100000)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def checkpoint_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Expected a LiTEN-PKA training checkpoint with a `model` key: {path}")
    return payload


def build_model_from_checkpoint(path: str | Path, fallback_config: dict[str, Any], device: torch.device) -> LiTENPKA:
    payload = checkpoint_payload(path)
    cfg = deepcopy(payload.get("config") or fallback_config)
    cfg["liten"]["root"] = str(Path(__file__).resolve().parents[1] / "vendor/LiTEN")
    cfg["liten"]["checkpoint"] = ""
    model = LiTENPKA(cfg["liten"], cfg["model"])
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


def build_reference_model(config: dict[str, Any], seed: int, device: torch.device) -> LiTENPKA:
    set_seed(seed)
    return LiTENPKA(config["liten"], config["model"]).to(device).eval()


def backbone_state(model: LiTENPKA) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().float() for k, v in model.encoder.backbone.state_dict().items()}


def layer_group(name: str) -> str:
    match = re.match(r"Interaction\.(\d+)\.", name)
    if match:
        return f"layer_{int(match.group(1)) + 1}"
    if name.startswith("embedding") or name.startswith("radial_fn"):
        return "input"
    if name.startswith("out_norm"):
        return "output_norm"
    return "other"


def flatten_group(state: dict[str, torch.Tensor], group: str, keys: list[str]) -> torch.Tensor:
    chunks = [state[k].reshape(-1) for k in keys if layer_group(k) == group and state[k].is_floating_point()]
    if not chunks:
        return torch.empty(0)
    return torch.cat(chunks)


def effective_rank_for_tensor(tensor: torch.Tensor) -> tuple[float, float]:
    if tensor.ndim < 2 or min(tensor.shape) < 2:
        return math.nan, math.nan
    mat = tensor.reshape(tensor.shape[0], -1).float()
    try:
        singular = torch.linalg.svdvals(mat).detach().cpu().numpy()
    except RuntimeError:
        return math.nan, math.nan
    singular = singular[singular > 1e-12]
    if singular.size == 0:
        return math.nan, math.nan
    prob = singular / singular.sum()
    entropy = float(-(prob * np.log(prob + 1e-12)).sum())
    return float(np.exp(entropy)), entropy


def group_effective_rank(state: dict[str, torch.Tensor], group: str, keys: list[str]) -> tuple[float, float]:
    values: list[tuple[float, float, int]] = []
    for key in keys:
        if layer_group(key) != group or not state[key].is_floating_point():
            continue
        erank, entropy = effective_rank_for_tensor(state[key])
        if not math.isnan(erank):
            values.append((erank, entropy, state[key].numel()))
    if not values:
        return math.nan, math.nan
    total = sum(weight for _, _, weight in values)
    erank = sum(value * weight for value, _, weight in values) / total
    entropy = sum(value * weight for _, value, weight in values) / total
    return float(erank), float(entropy)


def parameter_metrics(
    spice_model: LiTENPKA,
    ft_model: LiTENPKA,
    scratch_model: LiTENPKA,
    scratch_init_model: LiTENPKA,
) -> pd.DataFrame:
    states = {
        "spice_init": backbone_state(spice_model),
        "pretrained_ft": backbone_state(ft_model),
        "scratch_final": backbone_state(scratch_model),
        "scratch_init": backbone_state(scratch_init_model),
    }
    keys = sorted(set.intersection(*(set(state) for state in states.values())))
    groups = ["input"] + [f"layer_{idx}" for idx in range(1, 7)] + ["output_norm", "other"]
    rows: list[dict[str, Any]] = []

    comparisons = [
        ("pretrained_ft_vs_spice", "pretrained_ft", "spice_init"),
        ("scratch_final_vs_spice", "scratch_final", "spice_init"),
        ("scratch_final_vs_random_init", "scratch_final", "scratch_init"),
    ]
    for group in groups:
        for label, left, right in comparisons:
            a = flatten_group(states[left], group, keys)
            b = flatten_group(states[right], group, keys)
            if a.numel() == 0 or b.numel() == 0:
                continue
            diff = a - b
            drift = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(b).clamp_min(1e-12))
            cosine = float(F.cosine_similarity(a, b, dim=0).item())
            rows.append(
                {
                    "group": group,
                    "comparison": label,
                    "n_params": int(a.numel()),
                    "relative_l2_drift": drift,
                    "cosine_similarity": cosine,
                }
            )

        for state_name, state in states.items():
            erank, entropy = group_effective_rank(state, group, keys)
            rows.append(
                {
                    "group": group,
                    "comparison": f"{state_name}_effective_rank",
                    "n_params": int(flatten_group(state, group, keys).numel()),
                    "relative_l2_drift": math.nan,
                    "cosine_similarity": math.nan,
                    "effective_rank": erank,
                    "singular_value_entropy": entropy,
                }
            )
    return pd.DataFrame(rows)


def build_loader(
    table_path: str | Path,
    manifest_path: str | Path,
    config: dict[str, Any],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    data_cfg = config["data"]
    model_cfg = config["model"]
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
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_pka,
    )


@torch.no_grad()
def extract_embeddings(model: LiTENPKA, loader: DataLoader, device: torch.device, mode_name: str, split: str) -> pd.DataFrame:
    model.eval()
    if model.conf_pool != "node_mean" or not model.use_site_readout:
        raise ValueError("Figure 5 extraction requires the known-IC node_mean model.")
    rows: list[dict[str, Any]] = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        node_h = model.encoder.forward_nodes(
            batch["z"],
            batch["pos"],
            batch["conf_batch"],
            atom_features=batch.get("atom_features"),
        )
        task_ids = batch["task_id"]
        conf_batch = batch["conf_batch"]
        mol_for_conf = batch["mol_for_conf"]
        center_indices = batch["center_indices"]
        for mol_idx, mol_id in enumerate(batch["mol_ids"]):
            conf_ids = torch.nonzero(mol_for_conf == mol_idx, as_tuple=False).flatten()
            conf_nodes = []
            local_center = 0
            for j, conf_id in enumerate(conf_ids.tolist()):
                atom_indices = torch.nonzero(conf_batch == conf_id, as_tuple=False).flatten()
                if j == 0:
                    local_center = int((center_indices[conf_id] - atom_indices[0]).item())
                conf_nodes.append(node_h[atom_indices])
            atom_h = torch.stack(conf_nodes, dim=0).mean(dim=0)
            graph_h = atom_h.mean(dim=0)
            if not 0 <= local_center < atom_h.size(0):
                raise ValueError(f"Invalid supplied ionization center for {mol_id}")
            center = local_center
            site_h = atom_h[center]
            task_h = model.task_embedding(task_ids[mol_idx])
            mol_repr = torch.cat([graph_h, site_h, task_h], dim=-1)
            rows.append(
                {
                    "mode": mode_name,
                    "split": split,
                    "mol_id": mol_id,
                    "y": float(batch["y"][mol_idx].detach().cpu()),
                    "y_pred": float(model.conformer_head(mol_repr.unsqueeze(0)).item()),
                    "task_id": int(task_ids[mol_idx].detach().cpu()),
                    "graph_h": graph_h.detach().cpu().numpy(),
                    "site_h": site_h.detach().cpu().numpy(),
                    "mol_repr": mol_repr.detach().cpu().numpy(),
                }
            )
    return pd.DataFrame(rows)


def stack_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    return np.stack(frame[column].to_numpy()).astype(np.float32)


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def linear_probe_metrics(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    alphas = np.logspace(-4, 4, 17)
    for mode in sorted(train_df["mode"].unique()):
        train_mode = train_df[train_df["mode"] == mode]
        test_mode = test_df[test_df["mode"] == mode]
        for embedding_name in ["graph_h", "site_h", "mol_repr"]:
            x_train = stack_column(train_mode, embedding_name)
            y_train = train_mode["y"].to_numpy(dtype=np.float32)
            x_test = stack_column(test_mode, embedding_name)
            y_test = test_mode["y"].to_numpy(dtype=np.float32)
            probe = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
            probe.fit(x_train, y_train)
            pred = probe.predict(x_test)
            row = {"mode": mode, "embedding": embedding_name, "analysis": "linear_probe"}
            row.update(regression_report(y_test, pred))
            rows.append(row)
    return pd.DataFrame(rows)


def pairwise_pka_geometry(test_df: pd.DataFrame, pair_sample: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for mode in sorted(test_df["mode"].unique()):
        mode_df = test_df[test_df["mode"] == mode].reset_index(drop=True)
        y = mode_df["y"].to_numpy(dtype=np.float32)
        n = len(mode_df)
        for embedding_name in ["graph_h", "site_h", "mol_repr"]:
            x = stack_column(mode_df, embedding_name)
            x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
            n_pairs = min(pair_sample, n * (n - 1) // 2)
            i = rng.integers(0, n, size=n_pairs)
            j = rng.integers(0, n, size=n_pairs)
            mask = i != j
            i, j = i[mask], j[mask]
            cosine_distance = 1.0 - np.sum(x[i] * x[j], axis=1)
            pka_distance = np.abs(y[i] - y[j])
            rho, pvalue = spearmanr(cosine_distance, pka_distance)
            rows.append(
                {
                    "mode": mode,
                    "embedding": embedding_name,
                    "analysis": "embedding_distance_vs_abs_delta_pka",
                    "n_pairs": int(len(i)),
                    "spearman_rho": float(rho),
                    "spearman_pvalue": float(pvalue),
                }
            )
    return pd.DataFrame(rows)


def knn_smoothness(train_df: pd.DataFrame, test_df: pd.DataFrame, k: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mode in sorted(train_df["mode"].unique()):
        train_mode = train_df[train_df["mode"] == mode]
        test_mode = test_df[test_df["mode"] == mode]
        y_train = train_mode["y"].to_numpy(dtype=np.float32)
        y_test = test_mode["y"].to_numpy(dtype=np.float32)
        for embedding_name in ["graph_h", "site_h", "mol_repr"]:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(stack_column(train_mode, embedding_name))
            x_test = scaler.transform(stack_column(test_mode, embedding_name))
            nn = NearestNeighbors(n_neighbors=min(k, len(train_mode)), metric="cosine")
            nn.fit(x_train)
            _, indices = nn.kneighbors(x_test)
            pred = y_train[indices].mean(axis=1)
            row = {"mode": mode, "embedding": embedding_name, "analysis": f"knn_{k}_pka_smoothness"}
            row.update(regression_report(y_test, pred))
            rows.append(row)
    return pd.DataFrame(rows)


def write_embedding_tables(train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path) -> None:
    for split, frame in [("train", train_df), ("test", test_df)]:
        for embedding_name in ["graph_h", "site_h", "mol_repr"]:
            rows = []
            for _, row in frame.iterrows():
                coords = row[embedding_name]
                flat = {
                    "mode": row["mode"],
                    "split": row["split"],
                    "mol_id": row["mol_id"],
                    "y": row["y"],
                    "task_id": row["task_id"],
                }
                flat.update({f"dim_{idx}": float(value) for idx, value in enumerate(coords)})
                rows.append(flat)
            pd.DataFrame(rows).to_csv(out_dir / f"{split}_{embedding_name}_embeddings.csv", index=False)


def write_projection_tables(test_df: pd.DataFrame, out_dir: Path) -> None:
    for embedding_name in ["graph_h", "site_h", "mol_repr"]:
        rows = []
        for mode in sorted(test_df["mode"].unique()):
            mode_df = test_df[test_df["mode"] == mode].reset_index(drop=True)
            x = StandardScaler().fit_transform(stack_column(mode_df, embedding_name))
            xy = PCA(n_components=2, random_state=0).fit_transform(x)
            for idx, row in mode_df.iterrows():
                rows.append(
                    {
                        "mode": mode,
                        "embedding": embedding_name,
                        "mol_id": row["mol_id"],
                        "y": row["y"],
                        "task_id": row["task_id"],
                        "pca1": float(xy[idx, 0]),
                        "pca2": float(xy[idx, 1]),
                    }
                )
        pd.DataFrame(rows).to_csv(out_dir / f"test_{embedding_name}_pca.csv", index=False)


def plot_parameter_metrics(parameter_df: pd.DataFrame, out_dir: Path) -> None:
    ordered_groups = ["input"] + [f"layer_{idx}" for idx in range(1, 7)] + ["output_norm"]
    drift_df = parameter_df[parameter_df["comparison"].isin(["pretrained_ft_vs_spice", "scratch_final_vs_spice"])]
    for metric, ylabel, filename in [
        ("relative_l2_drift", "relative L2 drift", "layerwise_parameter_drift.png"),
        ("cosine_similarity", "cosine similarity", "layerwise_cosine_to_spice.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4))
        width = 0.35
        x = np.arange(len(ordered_groups))
        for offset, comparison in [(-width / 2, "pretrained_ft_vs_spice"), (width / 2, "scratch_final_vs_spice")]:
            values = []
            for group in ordered_groups:
                subset = drift_df[(drift_df["group"] == group) & (drift_df["comparison"] == comparison)]
                values.append(float(subset[metric].iloc[0]) if len(subset) else np.nan)
            ax.bar(x + offset, values, width=width, label=comparison)
        ax.set_xticks(x)
        ax.set_xticklabels(ordered_groups, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=300)
        plt.close(fig)

    erank_df = parameter_df[parameter_df["comparison"].str.endswith("_effective_rank", na=False)]
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.25
    x = np.arange(len(ordered_groups))
    for idx, state_name in enumerate(["spice_init", "pretrained_ft", "scratch_final"]):
        values = []
        label = f"{state_name}_effective_rank"
        for group in ordered_groups:
            subset = erank_df[(erank_df["group"] == group) & (erank_df["comparison"] == label)]
            values.append(float(subset["effective_rank"].iloc[0]) if len(subset) else np.nan)
        ax.bar(x + (idx - 1) * width, values, width=width, label=state_name)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_groups, rotation=35, ha="right")
    ax.set_ylabel("weighted effective rank")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "layerwise_effective_rank.png", dpi=300)
    plt.close(fig)


def write_prediction_tables(test_df: pd.DataFrame, dataset: PKADataset, out_dir: Path) -> None:
    from rdkit import Chem
    table = dataset.table
    records = pd.DataFrame({"mol_id": table[dataset.id_column].astype(str),
                            "smiles": table[dataset.smiles_column],
                            "pka": table[dataset.target_column],
                            "task": table[dataset.task_column],
                            "center_atom_idx": table[dataset.center_column]})
    records["center_symbol"] = [Chem.AddHs(Chem.MolFromSmiles(s)).GetAtomWithIdx(int(c)).GetSymbol()
                                for s, c in zip(records.smiles, records.center_atom_idx)]
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode, name in (("pretrained_ft", "pretrained"), ("scratch", "scratch")):
        frame = test_df[test_df["mode"] == mode].reset_index(drop=True)
        if records.mol_id.tolist() != frame.mol_id.tolist():
            raise ValueError("Embedding and prediction metadata order differ.")
        result = records.copy()
        result["y_true"] = frame.y.to_numpy()
        result["y_pred"] = frame.y_pred.to_numpy()
        result["abs_error"] = np.abs(result.y_pred - result.y_true)
        result.to_csv(out_dir / f"{name}_predictions.csv", index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    spice_cfg = load_config(args.spice_config)
    if args.spice_checkpoint:
        spice_cfg["liten"]["checkpoint"] = args.spice_checkpoint
    spice_cfg["liten"]["root"] = str(Path(__file__).resolve().parents[1] / "vendor/LiTEN")
    if not Path(spice_cfg["liten"]["checkpoint"]).is_file():
        raise FileNotFoundError("Original SPICE reference weights required; supply --spice-checkpoint.")
    scratch_cfg = load_config(args.scratch_config)
    pretrained_payload = checkpoint_payload(args.pretrained_ckpt)
    scratch_payload = checkpoint_payload(args.scratch_ckpt)
    pretrained_cfg = deepcopy(pretrained_payload.get("config") or spice_cfg)
    scratch_final_cfg = deepcopy(scratch_payload.get("config") or scratch_cfg)

    spice_model = build_reference_model(spice_cfg, args.seed, device)
    pretrained_model = build_model_from_checkpoint(args.pretrained_ckpt, pretrained_cfg, device)
    scratch_model = build_model_from_checkpoint(args.scratch_ckpt, scratch_final_cfg, device)
    scratch_init_cfg = deepcopy(scratch_final_cfg)
    scratch_init_cfg["liten"]["checkpoint"] = ""
    scratch_init_cfg["liten"]["root"] = str(Path(__file__).resolve().parents[1] / "vendor/LiTEN")
    scratch_init_model = build_reference_model(scratch_init_cfg, int(scratch_payload.get("seed", args.seed)), device)

    parameter_df = parameter_metrics(spice_model, pretrained_model, scratch_model, scratch_init_model)
    parameter_df.to_csv(out_dir / "parameter_layer_metrics.csv", index=False)
    plot_parameter_metrics(parameter_df, out_dir)

    train_loader = build_loader(args.train_file, args.manifest, pretrained_cfg, args.batch_size, args.num_workers)
    test_loader = build_loader(args.test_file, args.manifest, pretrained_cfg, args.batch_size, args.num_workers)
    train_frames = [
        extract_embeddings(pretrained_model, train_loader, device, "pretrained_ft", "train"),
        extract_embeddings(scratch_model, train_loader, device, "scratch", "train"),
    ]
    test_frames = [
        extract_embeddings(pretrained_model, test_loader, device, "pretrained_ft", "test"),
        extract_embeddings(scratch_model, test_loader, device, "scratch", "test"),
    ]
    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)

    probe_df = linear_probe_metrics(train_df, test_df)
    pairwise_df = pairwise_pka_geometry(test_df, args.pair_sample, args.seed)
    knn_df = knn_smoothness(train_df, test_df, args.knn_k)
    probe_df.to_csv(out_dir / "embedding_probe_metrics.csv", index=False)
    pairwise_df.to_csv(out_dir / "embedding_pairwise_metrics.csv", index=False)
    knn_df.to_csv(out_dir / "embedding_knn_metrics.csv", index=False)
    write_embedding_tables(train_df, test_df, out_dir)
    write_projection_tables(test_df, out_dir)
    write_prediction_tables(test_df, test_loader.dataset, out_dir.parent / "predictions")

    manifest = {
        "pretrained_checkpoint": args.pretrained_ckpt,
        "scratch_checkpoint": args.scratch_ckpt,
        "train_file": args.train_file,
        "test_file": args.test_file,
        "manifest": args.manifest,
        "device": str(device),
        "seed": args.seed,
        "pair_sample": args.pair_sample,
        "knn_k": args.knn_k,
        "test_records_per_model": test_df.groupby("mode").size().to_dict(),
        "train_records_per_model": train_df.groupby("mode").size().to_dict(),
        "input_sha256": {
            str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in [args.pretrained_ckpt, args.scratch_ckpt, args.train_file, args.test_file, args.manifest]
        },
        "outputs": sorted(path.name for path in out_dir.iterdir()),
    }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] Interpretability analysis written to {out_dir}")


if __name__ == "__main__":
    main()
