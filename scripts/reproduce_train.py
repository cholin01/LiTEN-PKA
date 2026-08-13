#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from liten_pka.config import deep_update, ensure_dirs, load_config
from liten_pka.conformer_generation import generate_conformers_for_tables
from liten_pka.train import train_one_seed as train_known_ic
from liten_pka.train_joint_site_pka import train_one_seed as train_joint


def _common_updates(args) -> dict:
    return {
        "data": {
            "conformer_dir": args.conformer_dir,
            "train_file": args.train_file,
            "val_file": args.val_file,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce LiTEN-pKa entry2/no-carbon training results using train/validation data only.")
    parser.add_argument("--config", default="configs/pkalearn_liten.yaml")
    parser.add_argument("--mode", choices=["known_ic", "joint", "both"], default="both")
    parser.add_argument("--train-file", default="data/article_protocols/article_entry2_noC_train.csv")
    parser.add_argument("--val-file", default="data/article_protocols/article_entry2_noC_val.csv")
    parser.add_argument("--conformer-dir", default="data/conformers_entry2_noC_trainval")
    parser.add_argument("--checkpoint-dir", default="models_reproduce")
    parser.add_argument("--report-dir", default="reports_reproduce")
    parser.add_argument("--known-ic-name", default="known_ic_entry2_noC")
    parser.add_argument("--joint-name", default="joint_entry2_noC")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--site-loss-weight", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-conformers", action="store_true")
    args = parser.parse_args()

    cfg = deep_update(load_config(args.config), _common_updates(args))
    ensure_dirs(cfg)
    generate_conformers_for_tables([args.train_file, args.val_file], args.conformer_dir, cfg, workers=args.workers, force=args.force_conformers)

    summaries = {}
    if args.mode in {"known_ic", "both"}:
        summaries["known_ic"] = train_known_ic(cfg, args.seed, args.known_ic_name)
    if args.mode in {"joint", "both"}:
        joint_cfg = deep_update(cfg, {"model": {"site_loss_weight": args.site_loss_weight}})
        summaries["joint"] = train_joint(joint_cfg, args.seed, args.joint_name)
    out = Path(args.report_dir) / "reproduce_train_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
