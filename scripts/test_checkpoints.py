#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from liten_pka.config import load_config
from liten_pka.conformer_generation import generate_conformers_for_tables
from liten_pka.predict import predict_table

BENCHMARKS = {
    "entry1": "data/article_protocols/article_entry1_test.csv",
    "entry2_noC": "data/article_protocols/article_entry2_noC_test.csv",
    "entry3_fromNS": "data/article_protocols/article_entry3_fromNS_test.csv",
    "official_test": "data/processed/test.csv",
    "sampl8": "data/downstream/SAMPL8.csv",
    "eurosampl": "data/downstream/euroSAMPL.csv",
}

TASK_NOTES = {
    "entry1": "pKaLearn article Table 1 entry 1 testing set.",
    "entry2_noC": "pKaLearn article Table 1 entry 2: no-carbon training protocol evaluated on no-carbon testing set.",
    "entry3_fromNS": "pKaLearn article Table 1 entry 3 held-out Novartis test set.",
    "official_test": "Included pKaLearn official held-out test split prepared in data/processed/test.csv.",
    "sampl8": "External SAMPL8 benchmark table included in data/downstream/SAMPL8.csv.",
    "eurosampl": "External euroSAMPL benchmark table included in data/downstream/euroSAMPL.csv.",
}


def _write_metrics(pred_df, output_csv: Path) -> None:
    metrics = {key: float(value) for key, value in pred_df.attrs.items()}
    if metrics:
        output_csv.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def evaluation_records(input_csv: Path, manifest: Path, cfg: dict) -> pd.DataFrame:
    table = pd.read_csv(input_csv)
    cache = pd.read_csv(manifest)
    id_column = cfg["data"]["id_column"]
    if id_column not in table:
        table[id_column] = [f"mol_{idx:06d}" for idx in range(len(table))]
    column = cfg["data"]["smiles_column"] if "input_smiles" in cache else cfg["data"]["id_column"]
    cache_column = "input_smiles" if "input_smiles" in cache else column
    if cache[cache_column].duplicated().any():
        raise ValueError("Duplicate conformer-cache keys would duplicate evaluation records.")
    table["input_row_index"] = np.arange(len(table))
    return table.loc[table[column].astype(str).isin(cache[cache_column].astype(str))].reset_index(drop=True)


def check_prediction_order(pred_df: pd.DataFrame, records: pd.DataFrame, cfg: dict) -> None:
    if pred_df.mol_id.astype(str).tolist() != records[cfg["data"]["id_column"]].astype(str).tolist():
        raise ValueError("Prediction order differs from evaluation records.")
    if not np.allclose(pred_df.y_true, records[cfg["data"]["target_column"]], atol=1e-6, rtol=1e-6):
        raise ValueError("Prediction labels differ from evaluation records.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the two released LiTEN-pKa checkpoints on named benchmark tasks.")
    parser.add_argument("--config", default="configs/pkalearn_liten.yaml")
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), default="entry2_noC")
    parser.add_argument("--input", default="", help="Optional custom labeled CSV. Overrides --benchmark.")
    parser.add_argument("--conformer-dir", default="", help="Default: data/conformers_<benchmark>.")
    parser.add_argument("--output-dir", default="", help="Default: predictions/<benchmark>.")
    parser.add_argument("--known-ic-checkpoint", default="checkpoints/known_ic_entry2_noC/best.pt")
    parser.add_argument("--joint-checkpoint", default="checkpoints/joint_entry2_noC/best.pt")
    parser.add_argument("--mode", choices=["known_ic", "joint", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-conformers", action="store_true")
    parser.add_argument("--site-candidates", choices=["heavy_inference", "legacy_all"], default="heavy_inference",
                        help="legacy_all is only for explicitly reproducing an older all-atom evaluation.")
    args = parser.parse_args()

    input_csv = Path(args.input or BENCHMARKS[args.benchmark])
    conformer_dir = Path(args.conformer_dir or f"data/conformers_{args.benchmark}")
    output_dir = Path(args.output_dir or f"predictions/{args.benchmark}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    manifest = generate_conformers_for_tables([input_csv], conformer_dir, cfg, workers=args.workers, force=args.force_conformers)
    records = evaluation_records(input_csv, manifest, cfg)
    records_path = output_dir / "evaluation_records.csv"
    records.to_csv(records_path, index=False)
    (output_dir / "evaluation_protocol.json").write_text(json.dumps({
        "input": str(input_csv), "manifest": str(manifest),
        "input_sha256": hashlib.sha256(input_csv.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        "records_file": records_path.name,
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "mode": args.mode, "joint_site_candidates": args.site_candidates,
        "selection": "Supplied checkpoint only; no checkpoint or seed ranking.",
        "checkpoints": {
            mode: {"path": checkpoint, "sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()}
            for mode, checkpoint in [("known_ic", args.known_ic_checkpoint), ("joint", args.joint_checkpoint)]
            if args.mode in {mode, "both"}
        },
    }, indent=2) + "\n")

    print(TASK_NOTES.get(args.benchmark, f"Custom labeled benchmark: {input_csv}"))
    if args.mode in {"known_ic", "both"}:
        output_csv = output_dir / "known_ic_predictions.csv"
        pred_df = predict_table(args.known_ic_checkpoint, input_csv, manifest, output_csv, batch_size=args.batch_size, device=args.device, mode="known_ic")
        check_prediction_order(pred_df, records, cfg)
        _write_metrics(pred_df, output_csv)
        print(f"known_ic -> {output_csv}")
    if args.mode in {"joint", "both"}:
        output_csv = output_dir / "joint_predictions.csv"
        pred_df = predict_table(args.joint_checkpoint, input_csv, manifest, output_csv, batch_size=args.batch_size, device=args.device, mode="joint", site_candidate_policy=args.site_candidates)
        check_prediction_order(pred_df, records, cfg)
        _write_metrics(pred_df, output_csv)
        print(f"joint -> {output_csv}")


if __name__ == "__main__":
    main()
