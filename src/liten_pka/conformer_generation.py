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
    canonical = canonicalize_smiles(smiles)
    row = {
        "conformer_id": conformer_id,
        "input_smiles": smiles,
        "canonical_smiles": canonical,
        "order_smiles": smiles,
        "conformer_npz": str(npz_path),
    }
    if npz_path.exists():
        return row, None
    try:
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
    """Generate a shared RDKit conformer manifest for one or more CSV tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    if manifest_path.exists() and not force:
        return manifest_path

    data_cfg = config["data"]
    conf_cfg = config["conformers"]
    tables = [pd.read_csv(path) for path in table_paths]
    table = pd.concat(tables, ignore_index=True)
    smiles_column = data_cfg["smiles_column"]
    if smiles_column not in table.columns:
        raise ValueError(f"Input table must contain '{smiles_column}'.")
    unique = table[[smiles_column]].drop_duplicates().reset_index(drop=True)
    worker_cfg = {
        "output_dir": str(output_dir),
        "n_conformers": int(conf_cfg["n_conformers"]),
        "seed": int(conf_cfg["random_seed"]),
        "max_embed_attempts": int(conf_cfg["max_embed_attempts"]),
        "prune_rms_thresh": float(conf_cfg["prune_rms_thresh"]),
    }
    jobs = [(idx, str(row[smiles_column]), worker_cfg) for idx, row in unique.iterrows()]
    rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    with Pool(processes=int(workers)) as pool:
        for row, failure in tqdm(pool.imap_unordered(_generate_one, jobs), total=len(jobs), desc="conformers"):
            if row is not None:
                rows.append(row)
            if failure is not None:
                failures.append(failure)
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False)
    print(f"Wrote {len(rows)} conformer records to {manifest_path}")
    if failures:
        print(f"Wrote {len(failures)} failures to {output_dir / 'failures.csv'}")
    return manifest_path
