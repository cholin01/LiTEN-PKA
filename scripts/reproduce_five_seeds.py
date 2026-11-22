#!/usr/bin/env python
"""Five-run training or held-out evaluation; never rank runs by test performance."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def commands(args):
    entry = {"entry1": "entry1", "entry2": "entry2_noC", "entry3": "entry3_fromNS"}[args.entry]
    tag = f"{entry}_{args.model}_{args.mode}"
    base = Path(args.output_dir).resolve()
    for seed in args.seeds:
        run = base / tag / f"seed_{seed}"
        conformers = base / "conformers" / entry
        if args.phase == "train":
            command = [
                sys.executable, str(ROOT / "scripts/reproduce_train.py"),
                "--config", "configs/pkalearn_liten.yaml",
                "--train-file", f"data/article_protocols/article_{entry}_train_pool.csv",
                "--split-seed", str(args.split_seed),
                "--baseline", args.model, "--mode", args.mode, "--seed", str(seed),
                "--known-ic-name", tag, "--joint-name", tag,
                "--checkpoint-dir", str(run / "models"), "--report-dir", str(run / "reports"),
                "--conformer-dir", str(conformers), "--epochs", str(args.epochs),
                "--batch-size", "32", "--lr", "0.0001",
                "--workers", str(args.workers), "--device", args.device,
            ]
            if getattr(args, "pretrain_checkpoint", ""):
                command += ["--pretrain-checkpoint", str(Path(args.pretrain_checkpoint).resolve())]
            yield command
        else:
            checkpoint = run / "models" / tag / f"seed_{seed}" / "best.pt"
            benchmarks = {"entry1": ["entry1"], "entry2": ["entry2_noC", "sampl8", "eurosampl"],
                          "entry3": ["entry3_fromNS"]}[args.entry]
            for benchmark in benchmarks:
                flag = "--known-ic-checkpoint" if args.mode == "known_ic" else "--joint-checkpoint"
                yield [
                    sys.executable, str(ROOT / "scripts/test_checkpoints.py"),
                    "--benchmark", benchmark, "--mode", args.mode, flag, str(checkpoint),
                    "--conformer-dir", str(conformers), "--output-dir", str(run / "test" / benchmark),
                    "--workers", str(args.workers), "--device", args.device,
                ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["train", "test"], required=True)
    parser.add_argument("--entry", choices=["entry1", "entry2", "entry3"], default="entry1")
    parser.add_argument("--model", choices=["ft", "scratch", "frozen", "schnet"], default="ft")
    parser.add_argument("--mode", choices=["known_ic", "joint"], default="known_ic")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/fresh_five_runs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pretrain-checkpoint", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(args.seeds) != 5 or len(set(args.seeds)) != 5:
        parser.error("Specify exactly five distinct seeds fixed before training.")
    if args.model == "schnet" and args.mode != "known_ic":
        parser.error("The SchNet baseline uses provided ionization centers.")
    plan = list(commands(args))
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return
    for command in plan:
        if args.phase == "train":
            report = Path(command[command.index("--report-dir") + 1])
            if report.exists() and any(report.iterdir()):
                raise FileExistsError(f"Refusing to overwrite an existing run: {report}")
        subprocess.run(command, cwd=ROOT, check=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{args.entry}_{args.model}_{args.mode}_{args.phase}_commands.json"
    (out / name).write_text(json.dumps(plan, indent=2) + "\n")
    if args.phase == "test":
        groups = {}
        for command in plan:
            folder = Path(command[command.index("--output-dir") + 1])
            benchmark = command[command.index("--benchmark") + 1]
            metrics = json.loads((folder / f"{args.mode}_predictions.metrics.json").read_text())
            groups.setdefault(benchmark, []).append(metrics)
        summaries = {
            name: {"n_runs": len(rows), "seeds": args.seeds,
                   **{f"{metric}_{stat}": function([row[metric] for row in rows])
                      for metric in ("mae", "rmse")
                      for stat, function in [("mean", statistics.mean), ("sample_sd", statistics.stdev)]}}
            for name, rows in groups.items()
        }
        target = out / f"{args.entry}_{args.model}_{args.mode}_test_summary.json"
        target.write_text(json.dumps(summaries, indent=2) + "\n")
        print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
