#!/usr/bin/env python
"""Extract real incoming site attention from fixed pretrained and scratch checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from extract_figure5 import build_loader, build_model_from_checkpoint, checkpoint_payload
from liten_pka.attention import extract_cases, write_attention_outputs
from liten_pka.predict import prepare_prediction_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True, help="CSV of preselected case molecules.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pretrained-ckpt", type=Path, required=True)
    parser.add_argument("--scratch-ckpt", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=6, help="1-based backbone layer; default: final LiTEN layer.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/figure5/attention"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    pairs = []
    checkpoints = {"pretrained_ft": args.pretrained_ckpt, "scratch": args.scratch_ckpt}
    for mode, path in checkpoints.items():
        payload = checkpoint_payload(path)
        config = payload["config"]
        model = build_model_from_checkpoint(path, config, device)
        prepared, table, cleanup = prepare_prediction_table(args.table, config, "known_ic")
        try:
            loader = build_loader(prepared, args.manifest, config, 1, 0)
            if len(loader.dataset) != len(table):
                raise ValueError("Every requested case must have exactly one matching conformer cache.")
            pairs.append(extract_cases(model, loader.dataset, range(len(loader.dataset)), mode, path, args.layer))
        finally:
            if cleanup is not None:
                cleanup.cleanup()
        del model
    write_attention_outputs(pairs, checkpoints, args.table, args.manifest, args.out_dir, args.layer)
    print(args.out_dir / "atom_attention.csv")


if __name__ == "__main__":
    main()
