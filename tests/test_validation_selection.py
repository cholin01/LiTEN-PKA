import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from liten_pka import train, train_joint_site_pka


class ToyModel(torch.nn.Module):
    def __init__(self, *_args):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        return self.value.expand_as(batch["y"])


class ToyJointModel(ToyModel):
    def forward(self, batch):
        return {"pka_pred": super().forward(batch), "site_loss": self.value.square(),
                "site_acc": self.value.new_tensor(1.0),
                "site_correct_count": torch.tensor(len(batch["y"])),
                "site_count": torch.tensor(len(batch["y"]))}


class ToyLoader(list):
    pass


class ValidationSelectionTests(unittest.TestCase):
    def test_both_trainers_rank_and_stop_on_validation_mae(self):
        for module, model_name, model_type in [
            (train, "LiTENPKA", ToyModel),
            (train_joint_site_pka, "LiTENJointSitePKA", ToyJointModel),
        ]:
            with self.subTest(trainer=module.__name__), tempfile.TemporaryDirectory() as tmp:
                config = {
                    "project": {"device": "cpu"},
                    "data": {"train_file": "train.csv", "val_file": "val.csv",
                             "conformer_dir": tmp},
                    "model": {}, "liten": {},
                    "train": {"lr": 0.01, "epochs": 10, "patience": 2,
                              "keep_top_k": 2, "checkpoint_dir": tmp + "/checkpoints",
                              "report_dir": tmp + "/reports"},
                }
                prediction = pd.DataFrame({"mol_id": ["validation"], "y_true": [1.0],
                                           "y_pred": [1.0]})
                evaluations = []
                for mae in [0.5, 0.4, 0.6, 0.7]:
                    metric = {"mae": mae}
                    evaluations.append((0.0, metric, prediction) if module is train
                                       else (metric, prediction))
                loader = ToyLoader([{"y": torch.tensor([1.0])}])
                with patch.object(module, "build_loader", return_value=loader) as build, \
                     patch.object(module, model_name, model_type), \
                     patch.object(module, "evaluate", side_effect=evaluations), \
                     patch.object(module, "tqdm", side_effect=lambda rows, **_kwargs: rows):
                    summary = module.train_one_seed(config, seed=0, experiment_name="check")
                self.assertEqual([call.args[0] for call in build.call_args_list],
                                 ["train.csv", "val.csv"])
                self.assertEqual(summary["best_epoch"], 2)
                self.assertEqual(summary["best_val_mae"], 0.4)
                self.assertEqual([item["epoch"] for item in summary["top_checkpoints"]], [2, 1])
                best = torch.load(summary["checkpoint"], map_location="cpu")
                last = torch.load(summary["last_checkpoint"], map_location="cpu")
                self.assertEqual(best["epoch"], 2)
                self.assertEqual(last["epoch"], 4)
                self.assertEqual(best["val_metrics"]["mae"], 0.4)
                self.assertTrue(all(Path(item["checkpoint"]).exists()
                                    for item in summary["top_checkpoints"]))


if __name__ == "__main__":
    unittest.main()
