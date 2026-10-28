from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .chem import pkalearn_atom_features


@dataclass
class PKASample:
    mol_id: str
    z: torch.Tensor
    pos: torch.Tensor
    atom_features: torch.Tensor
    y: torch.Tensor
    center_atom_idx: torch.Tensor
    task_id: torch.Tensor


TASK_TO_ID = {"acid": 0, "base": 1, "neutral": 2, "unknown": 3}


class PKADataset(Dataset[PKASample]):
    def __init__(
        self,
        table_path: str | Path,
        manifest_path: str | Path,
        id_column: str = "mol_id",
        target_column: str = "pka",
        center_column: str = "center_atom_idx",
        task_column: str = "task",
        smiles_column: str = "smiles",
        atom_feature_set: str = "none",
        include_center_feature: bool = False,
    ) -> None:
        labels = pd.read_csv(table_path)
        manifest = pd.read_csv(manifest_path)
        if "input_smiles" in manifest.columns:
            manifest_columns = ["input_smiles", "conformer_npz"]
            self.table = labels.merge(
                manifest[manifest_columns],
                left_on=smiles_column,
                right_on="input_smiles",
                how="left",
            )
        else:
            manifest_columns = [id_column, "conformer_npz"]
            self.table = labels.merge(manifest[manifest_columns], on=id_column, how="left")
        missing = self.table["conformer_npz"].isna()
        if missing.any():
            missing_ids = self.table.loc[missing, id_column].head(5).tolist()
            print(
                f"[WARN] Dropping {int(missing.sum())} rows without conformers, "
                f"examples: {missing_ids}."
            )
            self.table = self.table.loc[~missing].reset_index(drop=True)
        if self.table.empty:
            raise ValueError(f"No usable rows in {table_path}; all rows are missing conformers.")
        self.id_column = id_column
        self.target_column = target_column
        self.center_column = center_column
        self.task_column = task_column
        self.smiles_column = smiles_column
        self.atom_feature_set = str(atom_feature_set or "none")
        self.include_center_feature = bool(include_center_feature)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> PKASample:
        row = self.table.iloc[idx]
        cached = np.load(row["conformer_npz"], allow_pickle=True)
        task = str(row.get(self.task_column, "unknown")).lower()
        center_idx = int(row.get(self.center_column, -1))
        atom_features = pkalearn_atom_features(
            str(row[self.smiles_column]),
            feature_set=self.atom_feature_set,
            center_atom_idx=center_idx,
            include_center_feature=self.include_center_feature,
        )
        if len(cached["z"]) != atom_features.shape[0]:
            raise ValueError(
                f"Atom feature count mismatch for {row[self.id_column]} before collation: "
                f"{atom_features.shape[0]} features vs {len(cached['z'])} cached atoms."
            )
        return PKASample(
            mol_id=str(row[self.id_column]),
            z=torch.as_tensor(cached["z"], dtype=torch.long),
            pos=torch.as_tensor(cached["pos"], dtype=torch.float32),
            atom_features=torch.as_tensor(atom_features, dtype=torch.float32),
            y=torch.tensor(float(row[self.target_column]), dtype=torch.float32),
            center_atom_idx=torch.tensor(center_idx, dtype=torch.long),
            task_id=torch.tensor(TASK_TO_ID.get(task, TASK_TO_ID["unknown"]), dtype=torch.long),
        )


def collate_pka(
    batch: list[PKASample],
    atom_mask_ratio: float = 0.0,
    conformer_mode: str = "all",
) -> dict[str, Any]:
    conformer_mode = str(conformer_mode or "all").lower()
    if conformer_mode not in {"all", "random_one"}:
        raise ValueError(f"Unknown conformer_mode: {conformer_mode}")

    z_chunks: list[torch.Tensor] = []
    pos_chunks: list[torch.Tensor] = []
    feat_chunks: list[torch.Tensor] = []
    conf_batch: list[torch.Tensor] = []
    mol_for_conf: list[int] = []
    center_indices: list[int] = []
    raw_center_indices: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    task_ids: list[torch.Tensor] = []

    atom_offset = 0
    conf_idx = 0
    for mol_idx, sample in enumerate(batch):
        n_confs, n_atoms, _ = sample.pos.shape
        if sample.atom_features.shape[0] != n_atoms:
            raise ValueError(
                f"Atom feature count mismatch for {sample.mol_id}: "
                f"{sample.atom_features.shape[0]} features vs {n_atoms} atoms."
            )
        if conformer_mode == "random_one":
            conf_indices = [int(torch.randint(n_confs, (1,)).item())]
        else:
            conf_indices = list(range(n_confs))
        for conf in conf_indices:
            z_chunks.append(sample.z)
            pos_chunks.append(sample.pos[conf])
            feat_chunks.append(sample.atom_features)
            conf_batch.append(torch.full((n_atoms,), conf_idx, dtype=torch.long))
            mol_for_conf.append(mol_idx)
            center = int(sample.center_atom_idx.item())
            center_indices.append(atom_offset + (center if center >= 0 else 0))
            atom_offset += n_atoms
            conf_idx += 1
        raw_center_indices.append(sample.center_atom_idx)
        ys.append(sample.y)
        task_ids.append(sample.task_id)

    z = torch.cat(z_chunks, dim=0)
    center_tensor = torch.tensor(center_indices, dtype=torch.long)
    if atom_mask_ratio > 0:
        mask = torch.rand(z.shape) < float(atom_mask_ratio)
        if len(center_tensor):
            mask[center_tensor] = False
        z = z.clone()
        z[mask] = 0

    return {
        "z": z,
        "pos": torch.cat(pos_chunks, dim=0),
        "atom_features": torch.cat(feat_chunks, dim=0),
        "conf_batch": torch.cat(conf_batch, dim=0),
        "mol_for_conf": torch.tensor(mol_for_conf, dtype=torch.long),
        "center_indices": center_tensor,
        "center_atom_idx": torch.stack(raw_center_indices, dim=0),
        "y": torch.stack(ys, dim=0),
        "task_id": torch.stack(task_ids, dim=0),
        "mol_ids": [sample.mol_id for sample in batch],
    }
