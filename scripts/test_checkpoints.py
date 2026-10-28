#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    args = parser.parse_args()

    input_csv = Path(args.input or BENCHMARKS[args.benchmark])
    conformer_dir = Path(args.conformer_dir or f"data/conformers_{args.benchmark}")
    output_dir = Path(args.output_dir or f"predictions/{args.benchmark}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    manifest = generate_conformers_for_tables([input_csv], conformer_dir, cfg, workers=args.workers, force=args.force_conformers)

    print(TASK_NOTES.get(args.benchmark, f"Custom labeled benchmark: {input_csv}"))
    if args.mode in {"known_ic", "both"}:
        output_csv = output_dir / "known_ic_predictions.csv"
        pred_df = predict_table(args.known_ic_checkpoint, input_csv, manifest, output_csv, batch_size=args.batch_size, device=args.device, mode="known_ic")
        _write_metrics(pred_df, output_csv)
        print(f"known_ic -> {output_csv}")
    if args.mode in {"joint", "both"}:
        output_csv = output_dir / "joint_predictions.csv"
        pred_df = predict_table(args.joint_checkpoint, input_csv, manifest, output_csv, batch_size=args.batch_size, device=args.device, mode="joint")
        _write_metrics(pred_df, output_csv)
        print(f"joint -> {output_csv}")


if __name__ == "__main__":
    main()
