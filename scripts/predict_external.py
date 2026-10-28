#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from liten_pka.config import load_config
from liten_pka.conformer_generation import generate_conformers_for_tables
from liten_pka.predict import predict_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict pKa for an external CSV. Joint mode accepts a CSV with only a smiles column.")
    parser.add_argument("--input", required=True, help="External CSV. Minimum column: smiles.")
    parser.add_argument("--output", default="predictions/external_predictions.csv")
    parser.add_argument("--config", default="configs/pkalearn_liten.yaml")
    parser.add_argument("--mode", choices=["joint", "known_ic"], default="joint")
    parser.add_argument("--checkpoint", default="", help="Defaults to the released checkpoint matching --mode.")
    parser.add_argument("--conformer-dir", default="data/conformers_external")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-conformers", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint or (
        "checkpoints/joint_entry2_noC/best.pt" if args.mode == "joint" else "checkpoints/known_ic_entry2_noC/best.pt"
    )
    input_csv = Path(args.input)
    cfg = load_config(args.config)
    manifest = generate_conformers_for_tables([input_csv], args.conformer_dir, cfg, workers=args.workers, force=args.force_conformers)
    out = predict_table(checkpoint, input_csv, manifest, args.output, batch_size=args.batch_size, device=args.device, mode=args.mode)
    print(f"Wrote predictions: {args.output}")
    if out.attrs:
        print({key: float(value) for key, value in out.attrs.items()})


if __name__ == "__main__":
    main()
