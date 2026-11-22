from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable
import json
import os
import tempfile
import warnings
import zipfile

import numpy as np
import pandas as pd
from tqdm import tqdm

import hashlib
from rdkit import Chem, rdBase

from .chem import atom_numbers, canonicalize_smiles, embed_rdkit_conformers

MANIFEST_COLUMNS = ["conformer_id", "input_smiles", "canonical_smiles", "order_smiles", "conformer_npz"]
MANIFEST_COLUMNS += ["cache_schema", "generation_fingerprint", "actual_seed", "sha256", "cache_status"]
FAILURE_COLUMNS = ["conformer_id", "smiles", "error"]
CACHE_SCHEMA = 2


def generation_parameters(config: dict[str, Any]) -> dict[str, Any]:
    conf = config["conformers"]
    params = dict(schema=CACHE_SCHEMA, rdkit_version=rdBase.rdkitVersion,
                  n_conformers=int(conf["n_conformers"]), random_seed=int(conf["random_seed"]),
                  max_embed_attempts=int(conf["max_embed_attempts"]),
                  prune_rms_thresh=float(conf["prune_rms_thresh"]),
                  embedding="ETKDGv3", use_random_coords=True, explicit_h=True,
                  atom_order="input_smiles_rdkit_parse_then_AddHs",
                  force_field="MMFF_if_available_else_UFF", minimization_iterations=500,
                  shortage_retry="unpruned_single_embed_seed_plus_17",
                  seed_derivation="sha256(base_seed,exact_smiles)_31bit_v1")
    if params["n_conformers"] < 1 or params["max_embed_attempts"] < 1:
        raise ValueError("Conformer count and embedding limit must be positive.")
    if not np.isfinite(params["prune_rms_thresh"]) or not 0 <= params["random_seed"] < 2**31:
        raise ValueError("Invalid conformer RNG seed or RMS pruning threshold.")
    return params


def fingerprint(params: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def molecule_seed(smiles: str, base_seed: int) -> int:
    raw = json.dumps([base_seed, smiles], ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big") % (2**31 - 1)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_cache(path: Path, smiles: str, params: dict[str, Any], legacy: bool = False) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            expected = atom_numbers(Chem.AddHs(mol))
            if (not np.array_equal(data["z"], expected)
                    or data["pos"].shape != (params["n_conformers"], len(expected), 3)
                    or not np.isfinite(data["pos"]).all()
                    or data["energy"].shape != (params["n_conformers"],)
                    or str(data["smiles"].item()) != smiles):
                return False
            if legacy and "generation_metadata" not in data:
                return True
            metadata = json.loads(str(data["generation_metadata"].item()))
            return metadata == dict(parameters=params, input_smiles=smiles,
                                    actual_seed=molecule_seed(smiles, params["random_seed"]))
    except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile):
        return False


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
    params = cfg["parameters"]
    generation_id = fingerprint(params)
    conformer_id = hashlib.sha256((smiles + "\0" + generation_id).encode()).hexdigest()[:32]
    actual_seed = molecule_seed(smiles, params["random_seed"])
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
            "cache_schema": CACHE_SCHEMA,
            "generation_fingerprint": generation_id,
            "actual_seed": actual_seed,
            "cache_status": "validated",
        }
        if not cfg.get("force", False) and valid_cache(npz_path, smiles, params):
            row["sha256"] = file_hash(npz_path)
            return row, None
        mol, pos, energy = embed_rdkit_conformers(
            smiles,
            n_conformers=params["n_conformers"],
            seed=actual_seed,
            max_attempts=params["max_embed_attempts"],
            prune_rms_thresh=params["prune_rms_thresh"],
        )
        z = atom_numbers(mol)
        _write_multiconf_sdf(mol, raw_sdf, params["n_conformers"])
        used_backend = "rdkit"
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".npz", delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            np.savez_compressed(
                temp_path,
                z=z.astype(np.int64),
                pos=pos.astype(np.float32),
                energy=energy.astype(np.float32),
                smiles=np.asarray(smiles),
                canonical_smiles=np.asarray(canonical),
                backend=np.asarray(used_backend),
                generation_metadata=np.asarray(json.dumps(dict(parameters=params, input_smiles=smiles,
                                                               actual_seed=actual_seed), sort_keys=True)),
            )
            if not valid_cache(temp_path, smiles, params):
                raise ValueError("Generated cache did not pass atom-order/geometry/metadata validation.")
            os.replace(temp_path, npz_path)
        finally:
            temp_path.unlink(missing_ok=True)
        row["sha256"] = file_hash(npz_path)
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
    params = generation_parameters(config)
    policy = config["conformers"].get("legacy_cache_policy", "regenerate")
    if policy not in {"regenerate", "allow", "error"}:
        raise ValueError("legacy_cache_policy must be regenerate, allow or error")
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
    invalid_smiles = set()
    if not cached.empty:
        cached = cached.drop_duplicates("input_smiles", keep="last")
        valid_rows = []
        for row in cached.to_dict("records"):
            smiles = str(row["input_smiles"])
            path = Path(str(row["conformer_npz"]))
            recorded_hash = str(row.get("sha256", ""))
            hash_matches = path.is_file() and (
                recorded_hash in {"", "nan"} or recorded_hash == file_hash(path)
            )
            verified = valid_cache(path, smiles, params)
            if verified and not hash_matches:
                verified = False
            legacy = "generation_fingerprint" not in row or pd.isna(row["generation_fingerprint"])
            if not verified and legacy and path.is_file() and smiles in requested_smiles and not force:
                if policy == "error":
                    raise ValueError(f"Legacy cache metadata missing: {path}; explicitly allow or regenerate it.")
                if policy == "allow" and hash_matches and valid_cache(path, smiles, params, legacy=True):
                    warnings.warn(f"Using legacy conformers with unverified generation settings: {path}")
                    row["cache_status"] = "legacy_unverified"
                    row["sha256"] = file_hash(path)
                    verified = True
            if verified:
                valid_rows.append(row)
            else:
                invalid_smiles.add(smiles)
        cached = pd.DataFrame(valid_rows, columns=MANIFEST_COLUMNS)
    available = set(cached["input_smiles"].astype(str))
    worker_cfg = {
        "output_dir": str(output_dir),
        "parameters": params,
        "force": force,
    }
    jobs = [
        (idx, str(row[smiles_column]), dict(worker_cfg, force=force or str(row[smiles_column]) in invalid_smiles))
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
    # Keep the first legacy manifest for tracing published geometry; do not rewrite its NPZs.
    if manifest_path.exists():
        legacy_backup = output_dir / "manifest.before_schema2.csv"
        if not legacy_backup.exists():
            legacy_backup.write_bytes(manifest_path.read_bytes())
    with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".csv", delete=False) as handle:
        temp_manifest = Path(handle.name)
    try:
        pd.DataFrame(rows, columns=MANIFEST_COLUMNS).sort_values("input_smiles").to_csv(temp_manifest, index=False)
        os.replace(temp_manifest, manifest_path)
    finally:
        temp_manifest.unlink(missing_ok=True)
    (output_dir / "generation_config.json").write_text(
        json.dumps(dict(parameters=params, fingerprint=fingerprint(params), legacy_cache_policy=policy),
                   indent=2) + "\n", encoding="utf-8")
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
