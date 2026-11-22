import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from liten_pka import conformer_generation as cg


class CacheMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = {"data": {"smiles_column": "smiles"},
                       "conformers": {"n_conformers": 2, "random_seed": 2026,
                                      "max_embed_attempts": 100, "prune_rms_thresh": -1.0}}

    def generate(self, name, smiles, config=None, workers=1):
        csv = self.root / f"{name}.csv"
        pd.DataFrame({"smiles": smiles}).to_csv(csv, index=False)
        return pd.read_csv(cg.generate_conformers_for_tables(
            [csv], self.root / name, config or self.config, workers=workers)).set_index("input_smiles")

    def test_order_and_worker_independent_geometry(self):
        first = self.generate("a", ["CCO", "N"])
        second = self.generate("b", ["N", "CCO"], workers=2)
        for smiles in first.index:
            self.assertEqual(first.loc[smiles, "actual_seed"], second.loc[smiles, "actual_seed"])
            with np.load(first.loc[smiles, "conformer_npz"]) as a, np.load(second.loc[smiles, "conformer_npz"]) as b:
                np.testing.assert_array_equal(a["z"], b["z"])
                np.testing.assert_array_equal(a["pos"], b["pos"])

    def test_parameter_changes_invalidate_without_overwriting_old_npz(self):
        first = self.generate("cache", ["CCO"])
        original = Path(first.loc["CCO", "conformer_npz"])
        old = original.read_bytes()
        for key, value in [("random_seed", 17), ("n_conformers", 3),
                           ("max_embed_attempts", 200), ("prune_rms_thresh", 0.1)]:
            config = copy.deepcopy(self.config)
            config["conformers"][key] = value
            changed = self.generate("cache", ["CCO"], config)
            self.assertNotEqual(changed.loc["CCO", "conformer_npz"], str(original))
            self.assertEqual(original.read_bytes(), old)
        original_params = cg.generation_parameters(self.config)
        with patch.object(cg.rdBase, "rdkitVersion", "different-version"):
            self.assertNotEqual(cg.fingerprint(original_params), cg.fingerprint(cg.generation_parameters(self.config)))

    def test_corruption_is_regenerated_and_legacy_requires_opt_in(self):
        first = self.generate("cache", ["C"])
        npz = Path(first.loc["C", "conformer_npz"])
        npz.write_bytes(b"broken ZIP")
        repaired = self.generate("cache", ["C"])
        self.assertTrue(cg.valid_cache(npz, "C", cg.generation_parameters(self.config)))
        with np.load(npz) as data:
            legacy = {key: data[key] for key in data.files if key != "generation_metadata"}
        np.savez_compressed(npz, **legacy)
        repaired.reset_index()[cg.MANIFEST_COLUMNS[:5]].to_csv(self.root / "cache/manifest.csv", index=False)
        config = copy.deepcopy(self.config)
        config["conformers"]["legacy_cache_policy"] = "error"
        with self.assertRaisesRegex(ValueError, "Legacy"):
            self.generate("cache", ["C"], config)
        config["conformers"]["legacy_cache_policy"] = "allow"
        with self.assertWarnsRegex(UserWarning, "unverified"):
            accepted = self.generate("cache", ["C"], config)
        self.assertEqual(accepted.loc["C", "cache_status"], "legacy_unverified")
        legacy["pos"] = legacy["pos"] + 1
        np.savez_compressed(npz, **legacy)
        replaced = self.generate("cache", ["C"], config)
        self.assertEqual(replaced.loc["C", "cache_status"], "validated")


if __name__ == "__main__":
    unittest.main()
