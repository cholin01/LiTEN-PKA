#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from liten_pka.chem import canonicalize_smiles
from liten_pka.config import deep_update, ensure_dirs, load_config
from liten_pka.conformer_generation import generate_conformers_for_tables
from liten_pka.train import train_one_seed as train_known_ic
from liten_pka.train_joint_site_pka import train_one_seed as train_joint


def _common_updates(args, train_file: Path, val_file: Path) -> dict:
    return {
        "data": {
            "conformer_dir": args.conformer_dir,
            "train_file": str(train_file),
            "val_file": str(val_file),
        },
        "train": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "checkpoint_dir": args.checkpoint_dir,
            "report_dir": args.report_dir,
            "train_conformer_mode": "random_one",
        },
        "project": {"device": args.device},
    }


def split_train_pool(
    train_pool_file: str | Path,
    split_dir: str | Path,
    smiles_column: str = "smiles",
    validation_fraction: float = 0.1,
    split_seed: int = 0,
) -> dict[str, object]:
    """Create a deterministic, molecule-grouped train/validation split.

    Rows with the same canonical molecular structure stay in the same partition,
    including records with different pKa labels or ionization sites. This function
    accepts only the training pool; it has no test-set input.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    table = pd.read_csv(train_pool_file)
    if smiles_column not in table.columns:
        raise ValueError(f"Training pool must contain a '{smiles_column}' column")
    if len(table) < 2:
        raise ValueError("Training pool must contain at least two records")

    grouped_indices: dict[str, list[int]] = {}
    for row_idx, smiles in enumerate(table[smiles_column].astype(str)):
        structure_key = canonicalize_smiles(smiles)
        grouped_indices.setdefault(structure_key, []).append(row_idx)

    rng = np.random.default_rng(split_seed)
    group_keys = np.asarray(sorted(grouped_indices), dtype=object)
    rng.shuffle(group_keys)
    target_val_rows = max(1, int(round(len(table) * validation_fraction)))

    val_groups: set[str] = set()
    val_rows = 0
    for key_value in group_keys:
        key = str(key_value)
        group_size = len(grouped_indices[key])
        current_error = abs(target_val_rows - val_rows)
        added_error = abs(target_val_rows - (val_rows + group_size))
        if added_error <= current_error:
            val_groups.add(key)
            val_rows += group_size
        if val_rows >= target_val_rows:
            break

    if not val_groups:
        smallest = min(grouped_indices, key=lambda key: len(grouped_indices[key]))
        val_groups.add(smallest)
    if len(val_groups) == len(grouped_indices):
        largest = max(val_groups, key=lambda key: len(grouped_indices[key]))
        val_groups.remove(largest)

    row_keys = [canonicalize_smiles(str(smiles)) for smiles in table[smiles_column]]
    val_mask = np.asarray([key in val_groups for key in row_keys], dtype=bool)
    train_table = table.loc[~val_mask].reset_index(drop=True)
    val_table = table.loc[val_mask].reset_index(drop=True)
    if train_table.empty or val_table.empty:
        raise ValueError("Could not create non-empty train and validation partitions")

    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / "train.csv"
    val_path = split_dir / "val.csv"
    train_table.to_csv(train_path, index=False)
    val_table.to_csv(val_path, index=False)
    return {
        "train_file": str(train_path),
        "val_file": str(val_path),
        "train_records": int(len(train_table)),
        "validation_records": int(len(val_table)),
        "train_structures": int(len(set(row_keys) - val_groups)),
        "validation_structures": int(len(val_groups)),
        "validation_fraction_observed": float(len(val_table) / len(table)),
        "split_seed": int(split_seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train LiTEN-pKa from a training pool only. A deterministic 9:1 "
            "molecule-grouped train/validation split is created internally; "
            "held-out test data are never loaded by this script."
        )
    )
    parser.add_argument("--config", default="configs/pkalearn_liten.yaml")
    parser.add_argument("--mode", choices=["known_ic", "joint", "both"], default="both")
    parser.add_argument("--train-file", default="data/article_protocols/article_entry2_noC_train_pool.csv")
    parser.add_argument("--conformer-dir", default="")
    parser.add_argument("--checkpoint-dir", default="models_reproduce")
    parser.add_argument("--report-dir", default="reports_reproduce")
    parser.add_argument("--known-ic-name", default="known_ic_entry2_noC")
    parser.add_argument("--joint-name", default="joint_entry2_noC")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--site-loss-weight", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-conformers", action="store_true")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    data_cfg = base_cfg["data"]
    train_pool = Path(args.train_file)
    if not train_pool.is_file():
        raise FileNotFoundError(f"Training pool not found: {train_pool}")
    conformer_dir = args.conformer_dir or str(Path("data/conformers_reproduce") / train_pool.stem)
    args.conformer_dir = conformer_dir

    run_tag = train_pool.stem
    split_info = split_train_pool(
        train_pool,
        Path(args.report_dir) / "splits" / f"{run_tag}_split_seed_{args.split_seed}",
        smiles_column=data_cfg["smiles_column"],
        validation_fraction=0.1,
        split_seed=args.split_seed,
    )
    split_info["source_training_pool"] = str(train_pool)
    print("Training/validation split:")
    print(json.dumps(split_info, indent=2))

    cfg = deep_update(
        base_cfg,
        _common_updates(args, Path(split_info["train_file"]), Path(split_info["val_file"])),
    )
    ensure_dirs(cfg)

    # Conformers are generated only for the supplied training pool. The test CSV
    # is handled by the separate test_checkpoints.py entry point after training.
    manifest_path = Path(args.conformer_dir) / "manifest.csv"
    force_conformers = args.force_conformers
    if manifest_path.exists() and not force_conformers:
        cached_smiles = set(pd.read_csv(manifest_path)["input_smiles"].astype(str))
        pool_smiles = set(pd.read_csv(train_pool)[data_cfg["smiles_column"]].astype(str))
        force_conformers = not pool_smiles.issubset(cached_smiles)
    generate_conformers_for_tables(
        [train_pool],
        args.conformer_dir,
        cfg,
        workers=args.workers,
        force=force_conformers,
    )
    manifest = pd.read_csv(manifest_path)
    cached_smiles = set(manifest["input_smiles"].astype(str))
    pool_smiles = set(pd.read_csv(train_pool)[data_cfg["smiles_column"]].astype(str))
    missing_smiles = pool_smiles - cached_smiles
    if missing_smiles:
        examples = sorted(missing_smiles)[:3]
        raise RuntimeError(
            f"No conformer was generated for {len(missing_smiles)} training-pool SMILES; "
            f"examples: {examples}. Training stopped before model fitting."
        )

    summaries: dict[str, object] = {"split": split_info, "seed": args.seed, "models": {}}
    models: dict[str, object] = summaries["models"]
    if args.mode in {"known_ic", "both"}:
        models["known_ic"] = train_known_ic(cfg, args.seed, args.known_ic_name)
    if args.mode in {"joint", "both"}:
        joint_cfg = deep_update(cfg, {"model": {"site_loss_weight": args.site_loss_weight}})
        models["joint"] = train_joint(joint_cfg, args.seed, args.joint_name)

    out = Path(args.report_dir) / "reproduce_train_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
