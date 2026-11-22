import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


class ReproductionEntrypointTests(unittest.TestCase):
    def test_plans_reuse_pool_and_fixed_split_and_test_validation_best(self):
        launcher = script("reproduce_five_seeds")
        for entry, n_tests in (("entry1", 5), ("entry2", 15), ("entry3", 5)):
            args = SimpleNamespace(entry=entry, model="ft", mode="known_ic", output_dir="results/plans",
                                   phase="train", seeds=list(range(5)), split_seed=3, epochs=100,
                                   workers=1, device="cpu")
            training = list(launcher.commands(args))
            self.assertEqual(len(training), 5)
            self.assertTrue(all(command[command.index("--train-file") + 1].endswith("_train_pool.csv")
                                for command in training))
            self.assertTrue(all(command[command.index("--split-seed") + 1] == "3" for command in training))
            self.assertFalse(any("_test.csv" in item for command in training for item in command))
            args.phase = "test"
            testing = list(launcher.commands(args))
            self.assertEqual(len(testing), n_tests)
            self.assertTrue(all(command[command.index("--known-ic-checkpoint") + 1].endswith("/best.pt")
                                for command in testing))

    def make_runs(self, root, helper):
        records = pd.DataFrame({"input_row_index": [0, 1, 2, 3], "mol_id": ["same", "b", "same", "d"],
                                "smiles": ["CCN", "CCO", "CCN", "CCS"], "pka": [9., 7., 10., 8.],
                                "center_atom_idx": [2] * 4, "task": ["base", "acid", "base", "acid"],
                                "center_symbol": ["N", "O", "N", "S"], "source_name": ["a", "b", "c", "d"],
                                "source_dataset": ["unit_test"] * 4})
        for model in ("ft", "scratch"):
            tag = f"entry1_{model}_known_ic"
            for seed in range(5):
                run = root / tag / f"seed_{seed}"
                checkpoint = run / "models" / tag / f"seed_{seed}/best.pt"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(f"fixture-{model}-{seed}".encode())
                report = run / "reports" / tag / f"seed_{seed}"
                summary = {"seed": seed, "best_epoch": 2, "best_val_mae": .2,
                           "top_checkpoints": [{"epoch": 2, "val_mae": .2}, {"epoch": 1, "val_mae": .5}]}
                write_json(report / "summary.json", summary)
                history = [{"epoch": 1, "val_metrics": {"mae": .5}, "test_mae": .01},
                           {"epoch": 2, "val_metrics": {"mae": .2}, "test_mae": .8}]
                (report / "history.jsonl").write_text("\n".join(map(json.dumps, history)))
                write_json(run / "reports/reproduce_train_summary.json",
                           {"split": {"input_sha256": {"train_file": "train-sha", "val_file": "val-sha"}}})
                folder = run / "test/entry1"
                folder.mkdir(parents=True)
                record_path = folder / "evaluation_records.csv"
                records.to_csv(record_path, index=False)
                errors = np.asarray([.1, -.2, .3, -.4]) * (1 if model == "ft" else 2) + seed * .01
                pd.DataFrame({"mol_id": records.mol_id, "y_true": records.pka,
                              "y_pred": records.pka + errors}).to_csv(folder / "known_ic_predictions.csv", index=False)
                write_json(folder / "evaluation_protocol.json", {
                    "checkpoints": {"known_ic": {"sha256": helper.sha256(checkpoint)}},
                    "records_file": record_path.name, "records_sha256": helper.sha256(record_path)})

    def test_collect_uses_all_runs_preserves_duplicate_ids_and_rejects_test_best(self):
        helper = script("reproduce_results")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_runs(root, helper)
            data, metrics = helper.collect(root, "entry1", ["ft", "scratch"], "known_ic", list(range(5)))
            self.assertEqual(len(metrics), 10)
            self.assertEqual(len(data["datasets"]["Testing"]["records"]), 4)
            self.assertTrue((metrics.epoch == 2).all())
            summary_path = root / "entry1_ft_known_ic/seed_0/reports/entry1_ft_known_ic/seed_0/summary.json"
            summary = json.loads(summary_path.read_text())
            summary.update(best_epoch=1, best_val_mae=.5)
            write_json(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "validation history"):
                helper.collect(root, "entry1", ["ft", "scratch"], "known_ic", list(range(5)))

    def test_collect_rejects_mismatched_checkpoint_hash(self):
        helper = script("reproduce_results")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_runs(root, helper)
            protocol_path = root / "entry1_ft_known_ic/seed_0/test/entry1/evaluation_protocol.json"
            protocol = json.loads(protocol_path.read_text())
            protocol["checkpoints"]["known_ic"]["sha256"] = "different-checkpoint"
            write_json(protocol_path, protocol)
            with self.assertRaisesRegex(ValueError, "validation-best"):
                helper.collect(root, "entry1", ["ft", "scratch"], "known_ic", list(range(5)))

    def test_bootstrap_is_deterministic_and_groups_repeated_structures(self):
        helper, bootstrap = script("reproduce_results"), script("reproduce_bootstrap")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_runs(root, helper)
            payload, _ = helper.collect(root, "entry1", ["ft", "scratch"], "known_ic", list(range(5)))
            bootstrap.OUT, bootstrap.BOOTSTRAPS = root, 100
            a = bootstrap.analyze("Testing", payload["datasets"]["Testing"], "entry1", 42)
            b = bootstrap.analyze("Testing", payload["datasets"]["Testing"], "entry1", 42)
            self.assertEqual(a, b)
            self.assertEqual(a["n_bootstrap_clusters"], 3)
            self.assertEqual(a["seed_resampling"], "matched pairs")

    def test_evaluation_metadata_keeps_row_order_and_duplicate_ids(self):
        helper = script("test_checkpoints")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.csv"
            cache = root / "manifest.csv"
            pd.DataFrame({"mol_id": ["same", "missing", "same"], "smiles": ["CCN", "CCO", "CCN"],
                          "pka": [9., 7., 10.]}).to_csv(source, index=False)
            pd.DataFrame({"input_smiles": ["CCN"], "conformer_npz": ["fixture.npz"]}).to_csv(cache, index=False)
            cfg = {"data": {"smiles_column": "smiles", "id_column": "mol_id", "target_column": "pka"}}
            records = helper.evaluation_records(source, cache, cfg)
            self.assertEqual(records.input_row_index.tolist(), [0, 2])
            helper.check_prediction_order(pd.DataFrame({"mol_id": ["same", "same"], "y_true": [9., 10.]}), records, cfg)
            with self.assertRaisesRegex(ValueError, "labels"):
                helper.check_prediction_order(pd.DataFrame({"mol_id": ["same", "same"], "y_true": [10., 9.]}), records, cfg)

    def test_figure5_prediction_tables_preserve_alignment(self):
        extract, plot = script("extract_figure5"), script("reproduce_figure5")
        table = pd.DataFrame({"mol_id": ["same", "same"], "smiles": ["CCN", "CCO"],
                              "pka": [9., 7.], "task": ["base", "acid"], "center_atom_idx": [2, 2]})
        dataset = SimpleNamespace(table=table, id_column="mol_id", smiles_column="smiles", target_column="pka",
                                  task_column="task", center_column="center_atom_idx")
        rows = pd.DataFrame({"mode": ["pretrained_ft"] * 2 + ["scratch"] * 2,
                             "mol_id": ["same"] * 4, "y": [9., 7., 9., 7.], "y_pred": [8.9, 7.2, 8.7, 7.4]})
        with tempfile.TemporaryDirectory() as tmp:
            extract.write_prediction_tables(rows, dataset, Path(tmp))
            plot.PRED = Path(tmp)
            cases = plot.prediction_cases([0, 1])
            self.assertEqual(len(cases), 2)
            self.assertEqual(cases.center_symbol.tolist(), ["N", "O"])
            self.assertEqual(cases.pred_scratch.tolist(), [8.7, 7.4])

    def test_figure5_probe_fit_uses_only_training_arrays(self):
        import ast
        tree = ast.parse((ROOT / "scripts/extract_figure5.py").read_text())
        probe = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "linear_probe_metrics")
        fits = [node for node in ast.walk(probe) if isinstance(node, ast.Call) and ast.unparse(node.func).endswith(".fit")]
        self.assertEqual([ast.unparse(node) for node in fits], ["probe.fit(x_train, y_train)"])

    def test_released_model_embedding_readout_matches_forward(self):
        import os
        import torch
        from liten_pka.config import load_config
        from liten_pka.model import move_batch_to_device
        extract = script("extract_figure5")
        cwd, threads = Path.cwd(), torch.get_num_threads()
        os.chdir(ROOT)
        torch.set_num_threads(1)
        self.addCleanup(os.chdir, cwd)
        self.addCleanup(torch.set_num_threads, threads)
        config = load_config(ROOT / "configs/pkalearn_liten.yaml")
        manifest = ROOT / "data/conformers_entry2_row_order/manifest.csv"
        available = set(pd.read_csv(manifest).input_smiles)
        table = pd.read_csv(ROOT / "data/article_protocols/article_entry2_noC_test.csv")
        table = table[table.smiles.isin(available)].head(2)
        self.assertEqual(len(table), 2)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.csv"
            table.to_csv(source, index=False)
            model = extract.build_model_from_checkpoint(ROOT / "checkpoints/known_ic_entry2_noC/best.pt", config, torch.device("cpu"))
            loader = extract.build_loader(source, manifest, config, 2, 0)
            frame = extract.extract_embeddings(model, loader, torch.device("cpu"), "pretrained_ft", "test")
            with torch.no_grad():
                pred = model(move_batch_to_device(next(iter(loader)), torch.device("cpu"))).numpy()
            np.testing.assert_allclose(frame.y_pred, pred, rtol=1e-5, atol=1e-5)

    def test_schnet_inference_shape_and_finite_output(self):
        import torch
        from liten_pka.schnet import SiteSchNet
        batch = {"z": torch.tensor([6, 1, 1, 6, 1, 1]),
                 "pos": torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]] * 2),
                 "conf_batch": torch.tensor([0, 0, 0, 1, 1, 1]), "center_indices": torch.tensor([0, 3]),
                 "mol_for_conf": torch.tensor([0, 0]), "task_id": torch.tensor([0])}
        model = SiteSchNet().eval()
        with torch.no_grad():
            pred = model(batch)
        self.assertEqual(tuple(pred.shape), (1,))
        self.assertTrue(torch.isfinite(pred).all())


if __name__ == "__main__":
    unittest.main()
