from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

from .chem import canonicalize_smiles, stable_mol_id


def standardize_table(
    input_csv: str | Path,
    output_csv: str | Path,
    smiles_column: str = "smiles",
    target_column: str = "pka",
    center_column: str = "center_atom_idx",
    task_column: str = "task",
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    rows = []
    for _, row in df.iterrows():
        smiles = canonicalize_smiles(str(row[smiles_column]))
        item = row.to_dict()
        item["mol_id"] = stable_mol_id(smiles)
        item["smiles"] = smiles
        item["pka"] = float(row[target_column]) if target_column in row and pd.notna(row[target_column]) else None
        item["center_atom_idx"] = int(row[center_column]) if center_column in row and pd.notna(row[center_column]) else -1
        item["task"] = str(row[task_column]).lower() if task_column in row and pd.notna(row[task_column]) else "unknown"
        rows.append(item)
    out = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def _fingerprints(smiles: list[str]):
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    return fps


def butina_split(
    table: pd.DataFrame,
    output_dir: str | Path,
    smiles_column: str = "smiles",
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    cutoff: float = 0.55,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = _fingerprints(table[smiles_column].tolist())
    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - x for x in sims])
    clusters = Butina.ClusterData(dists, len(fps), cutoff, isDistData=True)
    clusters = sorted(clusters, key=len, reverse=True)

    n_total = len(table)
    n_test = max(1, int(round(n_total * test_fraction)))
    n_val = max(1, int(round(n_total * val_fraction)))
    split = ["train"] * n_total
    count = 0
    for cluster in clusters:
        if count >= n_test:
            break
        for idx in cluster:
            split[idx] = "test"
        count += len(cluster)
    count = 0
    for cluster in clusters:
        available = [idx for idx in cluster if split[idx] == "train"]
        if not available:
            continue
        if count >= n_val:
            break
        for idx in available:
            split[idx] = "val"
        count += len(available)
    table = table.copy()
    table["split"] = split
    table.to_csv(output_dir / "all.csv", index=False)
    for name in ["train", "val", "test"]:
        table[table["split"] == name].to_csv(output_dir / f"{name}.csv", index=False)

