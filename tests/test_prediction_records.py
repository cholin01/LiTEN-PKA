import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from liten_pka import predict


class ToyPredictor(torch.nn.Module):
    def __init__(self, *_args):
        super().__init__()

    def forward(self, batch):
        return batch["center_atom_idx"].float() * 2 + 1


class ToyJointPredictor(ToyPredictor):
    def forward(self, batch):
        return {
            "pka_pred": super().forward(batch),
            "pred_center_idx": batch["center_atom_idx"],
            "site_correct_count": torch.tensor(len(batch["y"])),
            "site_count": torch.tensor(len(batch["y"])),
        }


class PredictionRecordTests(unittest.TestCase):
    def test_repeated_ids_preserve_each_usable_record_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = root / "input.csv"
            pd.DataFrame({
                "mol_id": ["shared", "shared", "single", "missing"],
                "smiles": ["CCO", "CCO", "CC", "O"],
                "pka": [1.0, 4.0, 2.0, 9.0],
                "center_atom_idx": [0, 1, 0, 0],
                "task": ["acid"] * 4,
            }).to_csv(table, index=False)
            entries = []
            for i, smiles in enumerate(["CCO", "CC"]):
                molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
                z = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
                cache = root / f"conformer_{i}.npz"
                np.savez(cache, z=np.array(z), pos=np.zeros((1, len(z), 3)))
                entries.append({"input_smiles": smiles, "conformer_npz": str(cache)})
            manifest = root / "manifest.csv"
            pd.DataFrame(entries).to_csv(manifest, index=False)
            checkpoint = root / "model.pt"
            torch.save({"model": {}, "config": {
                "data": {"id_column": "mol_id", "smiles_column": "smiles",
                         "target_column": "pka", "center_column": "center_atom_idx",
                         "task_column": "task"},
                "liten": {}, "model": {},
            }}, checkpoint)

            for mode, model_name, model_type in [
                ("known_ic", "LiTENPKA", ToyPredictor),
                ("joint", "LiTENJointSitePKA", ToyJointPredictor),
            ]:
                with self.subTest(mode=mode), patch.object(predict, model_name, model_type):
                    output = root / f"{mode}.csv"
                    result = predict.predict_table(checkpoint, table, manifest, output,
                                                   batch_size=2, device="cpu", mode=mode)
                    self.assertEqual(result["mol_id"].tolist(), ["shared", "shared", "single"])
                    self.assertEqual(result["y_true"].tolist(), [1.0, 4.0, 2.0])
                    self.assertEqual(result["y_pred"].tolist(), [1.0, 3.0, 1.0])
                    self.assertEqual(result.attrs["n"], 3)
                    self.assertAlmostEqual(result.attrs["mae"], 2 / 3)
                    self.assertAlmostEqual(result.attrs["rmse"], np.sqrt(2 / 3))
                    self.assertEqual(len(pd.read_csv(output)), 3)
                    if mode == "joint":
                        self.assertEqual(result["pred_center_idx"].tolist(), [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
