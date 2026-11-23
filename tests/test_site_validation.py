import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from liten_pka.dataset import PKADataset
from liten_pka.predict import prepare_prediction_table
from liten_pka.site_validation import checked_center
from liten_pka import model as models
from test_heavy_candidates import ToyEncoder


class SiteValidationTests(unittest.TestCase):
    config = {"data": {"id_column": "mol_id", "smiles_column": "smiles", "target_column": "pka",
                       "center_column": "center_atom_idx", "task_column": "task"}}

    def test_rejects_noninteger_nonfinite_out_of_range_and_hydrogen(self):
        for value in (999, 3, -2, -1, 1.7, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                checked_center(value, [6, 7, 1])
        with self.assertRaisesRegex(ValueError, "heavy atom"):
            checked_center(2, [6, 7, 1])
        self.assertEqual(checked_center(1.0, [6, 7, 1]), 1)
        self.assertEqual(checked_center(-1, [6, 7, 1], allow_missing=True), -1)

    def test_csv_validation_before_cache_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.csv"
            for value in (999, -1, 1.7, 9, float("nan"), float("inf")):
                pd.DataFrame({"smiles": ["CCN"], "center_atom_idx": [value]}).to_csv(source, index=False)
                with self.subTest(value=value), self.assertRaises(ValueError):
                    prepare_prediction_table(source, self.config, "known_ic")
            pd.DataFrame({"smiles": ["CCN"]}).to_csv(source, index=False)
            _, table, cleanup = prepare_prediction_table(source, self.config, "joint")
            self.assertEqual(table.center_atom_idx.tolist(), [-1])
            cleanup.cleanup()

    def test_dataset_does_not_truncate_fractional_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root / "conf.npz", z=[6, 6, 7], pos=np.zeros((1, 3, 3)))
            pd.DataFrame({"mol_id": ["a"], "smiles": ["CCN"], "pka": [9.],
                          "center_atom_idx": [1.7], "task": ["base"]}).to_csv(root / "input.csv", index=False)
            pd.DataFrame({"input_smiles": ["CCN"], "conformer_npz": [str(root / "conf.npz")]}).to_csv(root / "manifest.csv", index=False)
            dataset = PKADataset(root / "input.csv", root / "manifest.csv")
            with self.assertRaisesRegex(ValueError, "finite integer"):
                dataset[0]

    def test_model_rejects_invalid_indices_instead_of_clamping(self):
        batch = {"z": torch.tensor([6, 7, 1]), "pos": torch.zeros(3, 3),
                 "conf_batch": torch.tensor([0, 0, 0]), "mol_for_conf": torch.tensor([0]),
                 "center_indices": torch.tensor([1]), "center_atom_idx": torch.tensor([1]),
                 "task_id": torch.tensor([0])}
        for model_type in (models.LiTENPKA, models.LiTENJointSitePKA):
            with patch.object(models, "LiTENNodeEncoder", ToyEncoder):
                model = model_type({}, {"dropout": 0}).eval()
            for center in (999, 2, 1.7, -2):
                bad = dict(batch, center_indices=torch.tensor([center]), center_atom_idx=torch.tensor([center]))
                with self.subTest(model=model_type.__name__, center=center), self.assertRaises(ValueError):
                    model(bad)
            model(batch)


if __name__ == "__main__":
    unittest.main()
