from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

import hashlib
from rdkit import Chem

from .chem import atom_numbers, canonicalize_smiles, embed_rdkit_conformers

MANIFEST_COLUMNS = ["conformer_id", "input_smiles", "canonical_smiles", "order_smiles", "conformer_npz"]
FAILURE_COLUMNS = ["conformer_id", "smiles", "error"]


def _write_multiconf_sdf(mol: Chem.Mol, path: Path, n_conformers: int) -> None:
    writer = Chem.SDWriter(str(path))
    for conf in list(mol.GetConformers())[:n_conformers]:
        writer.write(mol, confId=conf.GetId())
    writer.close()


def stable_conformer_id(smiles: str) -> str:
    return hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:16]


def _generate_one(args: tuple[int, str, dict[str, Any]]) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    row_idx, smiles, cfg = args
    output_dir = Path(cfg["output_dir"])
    conformer_id = stable_conformer_id(smiles)
    npz_path = output_dir / f"{conformer_id}.npz"
    raw_sdf = output_dir / f"{conformer_id}.rdkit.sdf"
    try:
        canonical = canonicalize_smiles(smiles)
        row = {
            "conformer_id": conformer_id,
            "input_smiles": smiles,
            "canonical_smiles": canonical,
            "order_smiles": smiles,
            "conformer_npz": str(npz_path),
        }
        if npz_path.is_file() and not cfg.get("force", False):
            return row, None
        mol, pos, energy = embed_rdkit_conformers(
            smiles,
            n_conformers=int(cfg["n_conformers"]),
            seed=int(cfg["seed"]) + int(row_idx),
            max_attempts=int(cfg["max_embed_attempts"]),
            prune_rms_thresh=float(cfg["prune_rms_thresh"]),
        )
        z = atom_numbers(mol)
        _write_multiconf_sdf(mol, raw_sdf, int(cfg["n_conformers"]))
        used_backend = "rdkit"
        np.savez_compressed(
            npz_path,
            z=z.astype(np.int64),
            pos=pos.astype(np.float32),
            energy=energy.astype(np.float32),
            smiles=np.asarray(smiles),
            canonical_smiles=np.asarray(canonical),
            backend=np.asarray(used_backend),
        )
        return row, None
    except Exception as exc:
        return None, {"conformer_id": conformer_id, "smiles": smiles, "error": repr(exc)}


def generate_conformers_for_tables(
    table_paths: Iterable[str | Path],
    output_dir: str | Path,
    config: dict[str, Any],
    workers: int = 8,
    force: bool = False,
) -> Path:
    """Reuse available conformers, generate missing ones, and record failures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    data_cfg = config["data"]
    conf_cfg = config["conformers"]
    tables = [pd.read_csv(path) for path in table_paths]
    table = pd.concat(tables, ignore_index=True)
    smiles_column = data_cfg["smiles_column"]
    if smiles_column not in table.columns:
        raise ValueError(f"Input table must contain '{smiles_column}'.")
    unique = table[[smiles_column]].drop_duplicates().reset_index(drop=True)
    requested_smiles = set(unique[smiles_column].astype(str))
    cached = pd.DataFrame(columns=MANIFEST_COLUMNS)
    if manifest_path.exists():
        try:
            cached = pd.read_csv(manifest_path)
        except pd.errors.EmptyDataError:
            pass
    if not cached.empty:
        cached = cached.loc[cached["conformer_npz"].map(lambda path: Path(str(path)).is_file())]
        cached = cached.drop_duplicates("input_smiles", keep="last")
    available = set(cached["input_smiles"].astype(str))
    if not force and requested_smiles.issubset(available):
        return manifest_path

    worker_cfg = {
        "output_dir": str(output_dir),
        "n_conformers": int(conf_cfg["n_conformers"]),
        "seed": int(conf_cfg["random_seed"]),
        "max_embed_attempts": int(conf_cfg["max_embed_attempts"]),
        "prune_rms_thresh": float(conf_cfg["prune_rms_thresh"]),
        "force": force,
    }
    jobs = [
        (idx, str(row[smiles_column]), worker_cfg)
        for idx, row in unique.iterrows()
        if force or str(row[smiles_column]) not in available
    ]
    retained = cached.loc[~cached["input_smiles"].isin(requested_smiles)] if force else cached
    rows: list[dict[str, str]] = retained.to_dict("records")
    failures: list[dict[str, str]] = []

    def collect(results):
        for row, failure in tqdm(results, total=len(jobs), desc="conformers"):
            if row is not None:
                rows.append(row)
            if failure is not None:
                failures.append(failure)

    if int(workers) == 1:
        collect(map(_generate_one, jobs))
    else:
        with Pool(processes=int(workers)) as pool:
            collect(pool.imap_unordered(_generate_one, jobs))
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest_path, index=False)
    failure_path = output_dir / "failures.csv"
    if failure_path.exists():
        try:
            previous = pd.read_csv(failure_path)
        except pd.errors.EmptyDataError:
            previous = pd.DataFrame(columns=FAILURE_COLUMNS)
        failures.extend(previous.loc[~previous["smiles"].isin(requested_smiles)].to_dict("records"))
    pd.DataFrame(failures, columns=FAILURE_COLUMNS).to_csv(failure_path, index=False)
    print(f"Wrote {len(rows)} conformer records to {manifest_path}")
    if failures:
        print(f"Wrote {len(failures)} failures to {output_dir / 'failures.csv'}")
    return manifest_path
