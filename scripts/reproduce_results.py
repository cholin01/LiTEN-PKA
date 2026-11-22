#!/usr/bin/env python
"""Summarize five fixed runs and export aligned predictions for Bootstrap."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = {"entry1": "entry1", "entry2": "entry2_noC", "entry3": "entry3_fromNS"}
BENCHMARKS = {"entry1": {"entry1": "Testing"},
              "entry2": {"entry2_noC": "NoC", "sampl8": "SAMPL8", "eurosampl": "euroSAMPL"},
              "entry3": {"entry3_fromNS": "Novartis"}}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_structure(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES in evaluation records: {smiles}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, isomericSmiles=True)


def validation_selection(run: Path, tag: str, seed: int) -> dict:
    report = run / "reports" / tag / f"seed_{seed}"
    summary = json.loads((report / "summary.json").read_text())
    if summary["seed"] != seed:
        raise ValueError(f"Training seed differs from requested seed: {report}")
    history = [json.loads(line) for line in (report / "history.jsonl").read_text().splitlines() if line.strip()]
    if not history or any(not np.isfinite(row["val_metrics"]["mae"]) for row in history):
        raise ValueError(f"Missing or non-finite validation history: {report}")
    best = min(history, key=lambda row: row["val_metrics"]["mae"])
    if summary["best_epoch"] != best["epoch"] or not np.isclose(
        summary["best_val_mae"], best["val_metrics"]["mae"], atol=1e-12, rtol=0
    ):
        raise ValueError(f"Checkpoint selection disagrees with validation history: {report}")
    top = summary["top_checkpoints"]
    expected = sorted(history, key=lambda row: row["val_metrics"]["mae"])[:5]
    if [row["epoch"] for row in top] != [row["epoch"] for row in expected]:
        raise ValueError(f"Top checkpoints are not the validation top-5: {report}")
    run_summary = json.loads((run / "reports/reproduce_train_summary.json").read_text())
    checkpoint = run / "models" / tag / f"seed_{seed}/best.pt"
    return {"seed": seed, "epoch": best["epoch"], "path": str(checkpoint),
            "sha256": sha256(checkpoint), "validation_mae": best["val_metrics"]["mae"],
            "split_sha256": run_summary["split"]["input_sha256"]}


def read_evaluation(folder: Path, mode: str, selection: dict):
    protocol = json.loads((folder / "evaluation_protocol.json").read_text())
    if protocol["checkpoints"][mode]["sha256"] != selection["sha256"]:
        raise ValueError(f"Evaluation did not use the validation-best checkpoint: {folder}")
    records_path = folder / protocol["records_file"]
    if sha256(records_path) != protocol["records_sha256"]:
        raise ValueError(f"Evaluation record checksum mismatch: {folder}")
    records = pd.read_csv(records_path, dtype={"mol_id": str})
    predictions = pd.read_csv(folder / f"{mode}_predictions.csv", dtype={"mol_id": str})
    if records.mol_id.tolist() != predictions.mol_id.tolist():
        raise ValueError(f"Evaluation row order mismatch: {folder}")
    if not np.allclose(records.pka, predictions.y_true, atol=1e-6, rtol=1e-6):
        raise ValueError(f"Evaluation label mismatch: {folder}")
    if not np.isfinite(predictions[["y_true", "y_pred"]].to_numpy()).all():
        raise ValueError(f"Non-finite predictions: {folder}")
    records["canonical_smiles"] = records.smiles.map(canonical_structure)
    records["row_idx"] = np.arange(len(records))
    return records, predictions


def collect(runs_dir: Path, entry: str, models: list[str], mode: str, seeds: list[int]):
    datasets, rows, split_hashes = {}, [], None
    for model in models:
        tag = f"{ENTRIES[entry]}_{model}_{mode}"
        for seed in seeds:
            run = runs_dir / tag / f"seed_{seed}"
            selection = validation_selection(run, tag, seed)
            if split_hashes is None:
                split_hashes = selection["split_sha256"]
            if selection["split_sha256"] != split_hashes:
                raise ValueError("Runs use different training/validation splits.")
            for benchmark, name in BENCHMARKS[entry].items():
                records, predictions = read_evaluation(run / "test" / benchmark, mode, selection)
                y = predictions.y_true.to_numpy(dtype=float)
                pred = predictions.y_pred.to_numpy(dtype=float)
                keys = ["input_row_index", "mol_id", "smiles", "pka", "center_atom_idx", "task"]
                if name not in datasets:
                    datasets[name] = {"records": json.loads(records.to_json(orient="records")),
                                      "y_true": y.tolist(), "models": []}
                else:
                    reference = pd.DataFrame(datasets[name]["records"])
                    pd.testing.assert_frame_equal(reference[keys], records[keys], check_dtype=False)
                    np.testing.assert_array_equal(datasets[name]["y_true"], y)
                datasets[name]["models"].append(dict(selection, group=model, predictions=pred.tolist()))
                error = pred - y
                rows.append({"entry": entry, "dataset": name, "model": model, "seed": seed,
                             "epoch": selection["epoch"], "n_records": len(y),
                             "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error ** 2))),
                             "validation_mae": selection["validation_mae"],
                             "checkpoint_sha256": selection["sha256"]})
    return {"datasets": datasets, "split_sha256": split_hashes}, pd.DataFrame(rows)


def load_predictions(source: Path):
    paths = sorted(source.glob("entry*_predictions.json"))
    if not paths:
        raise FileNotFoundError(f"No entry prediction tables in {source}")
    for path in paths:
        family = path.stem.removesuffix("_predictions")
        for dataset, payload in json.loads(path.read_text())["datasets"].items():
            y = np.asarray(payload["y_true"], dtype=float)
            if y.ndim != 1 or len(y) < 2 or len(y) != len(payload["records"]) or not np.isfinite(y).all():
                raise ValueError(f"Invalid labels: {path}/{dataset}")
            for group in {run["group"] for run in payload["models"]}:
                runs = [run for run in payload["models"] if run["group"] == group]
                if len(runs) != 5 or len({run["seed"] for run in runs}) != 5:
                    raise ValueError(f"Expected five distinct runs: {dataset}/{group}")
                for run in runs:
                    pred = np.asarray(run["predictions"], dtype=float)
                    if pred.shape != y.shape or not np.isfinite(pred).all():
                        raise ValueError(f"Invalid predictions: {dataset}/{group}")
            yield family, dataset, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "results/fresh_five_runs")
    parser.add_argument("--entry", choices=list(ENTRIES), default="entry1")
    parser.add_argument("--models", nargs="+", choices=["ft", "scratch", "frozen", "schnet"], default=["ft", "scratch"])
    parser.add_argument("--mode", choices=["known_ic", "joint"], default="known_ic")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/reproduced")
    args = parser.parse_args()
    if len(args.seeds) != 5 or len(set(args.seeds)) != 5 or len(set(args.models)) != len(args.models):
        parser.error("Specify five distinct seeds and distinct model names.")
    payload, runs = collect(args.runs_dir.resolve(), args.entry, args.models, args.mode, args.seeds)
    rows = []
    for (dataset, model), group in runs.groupby(["dataset", "model"], sort=True):
        rows.append({"entry": args.entry, "dataset": dataset, "model": model, "n_runs": len(group),
                     **{f"{metric}_{stat}": float(getattr(group[metric], method)())
                        for metric in ("mae", "rmse") for stat, method in (("mean", "mean"), ("sample_sd", "std"))}})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.entry}_predictions.json").write_text(json.dumps(payload, indent=2) + "\n")
    runs.to_csv(args.output_dir / f"{args.entry}_individual_runs.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / f"{args.entry}_five_run_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
