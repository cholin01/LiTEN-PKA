import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from liten_pka.attention import capture_attention, site_attention_tables
from liten_pka.config import load_config
from test_reproduction_entrypoints import ROOT, script


class AttentionTests(unittest.TestCase):
    def test_direction_heads_self_edges_and_missing_conformers(self):
        record = {"layer": 6, "edge_index": torch.tensor([[2, 2, 2, 5, 5], [0, 1, 2, 4, 5]]),
                  "distance": torch.ones(5),
                  "attention": torch.tensor([[-2., 2.], [1., 1.], [100., 100.], [4., 4.], [100., 100.]])}
        batch = {"task_id": torch.tensor([0]), "mol_for_conf": torch.tensor([0, 0]),
                 "conf_batch": torch.tensor([0, 0, 0, 1, 1, 1]), "z": torch.tensor([6, 1, 7] * 2),
                 "center_indices": torch.tensor([2, 5])}
        raw, summary = site_attention_tables(record, batch)
        self.assertEqual(len(raw), 10)
        self.assertEqual(raw.self_edge.sum(), 4)
        np.testing.assert_allclose(summary.attention_strength, [1 / 3, 2 / 3])
        np.testing.assert_allclose(summary.attention_signed_mean, [0, 2.5])
        self.assertEqual(summary.edge_conformer_count.tolist(), [1, 2])
        self.assertEqual(summary.atomic_number.tolist(), [6, 1])

    def test_real_checkpoint_attention_is_captured_without_prediction_changes(self):
        extract = script("extract_figure5")
        cfg = load_config(ROOT / "configs/pkalearn_liten.yaml")
        manifest = ROOT / "data/conformers_entry2_row_order/manifest.csv"
        table = pd.read_csv(ROOT / "data/article_protocols/article_entry2_noC_test.csv")
        available = set(pd.read_csv(manifest).input_smiles)
        table = table[table.smiles.isin(available)].head(1)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.csv"
            table.to_csv(source, index=False)
            model = extract.build_model_from_checkpoint(ROOT / "checkpoints/known_ic_entry2_noC/best.pt", cfg, torch.device("cpu"))
            batch = next(iter(extract.build_loader(source, manifest, cfg, 1, 0)))
            keys = tuple(model.state_dict())
            with torch.no_grad():
                before = model(batch)
                with capture_attention(model) as records:
                    during = model(batch)
                after = model(batch)
            torch.testing.assert_close(before, during, rtol=0, atol=0)
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            self.assertEqual(tuple(model.state_dict()), keys)
            self.assertNotIn("message", model.encoder.backbone.Interaction[5].__dict__)
            self.assertEqual(len(records), 1)
            raw, summary = site_attention_tables(records[0], batch)
            self.assertEqual(summary.n_conformers.iloc[0], 5)
            self.assertAlmostEqual(summary.attention_strength.sum(), 1.0)
            self.assertTrue(np.isfinite(raw.attention_signed).all())
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                with capture_attention(model):
                    raise RuntimeError("fixture")
            self.assertNotIn("message", model.encoder.backbone.Interaction[5].__dict__)
            model.train()
            with self.assertRaisesRegex(ValueError, "eval"):
                with capture_attention(model):
                    pass

    def test_plot_rejects_fake_or_mismatched_sources(self):
        plot = script("reproduce_figure5")
        row = pd.Series({"mol_id": "a", "smiles": "CCN", "task": "base", "center_atom_idx": 2,
                         "pred_pre": 9.0, "pred_scratch": 8.0})
        with self.assertRaisesRegex(ValueError, "actual model attention"):
            plot.case_attention(pd.DataFrame({"score": [1.]}), row)
        rows = []
        for mode, prediction in (("pretrained_ft", 9.), ("scratch", 8.)):
            for atom in (0, 1):
                rows.append(dict(mode=mode, mol_id="a", smiles="CCN", task="base", center_atom_idx=2,
                                 atom_idx=atom, symbol="C", attention_strength=.5, prediction=prediction,
                                 layer=6, n_conformers=5, n_heads=8, checkpoint_sha256=mode, conformer_sha256="same"))
        frame = pd.DataFrame(rows)
        _, selected = plot.case_attention(frame, row)
        self.assertEqual(selected, [0, 1])
        frame.loc[0, "prediction"] = 1.
        with self.assertRaisesRegex(ValueError, "predictions"):
            plot.case_attention(frame, row)


if __name__ == "__main__":
    unittest.main()
