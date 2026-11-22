import unittest
from unittest.mock import patch

import torch

from liten_pka import model as models


class ToyEncoder(torch.nn.Module):
    hidden_channels = 8

    def __init__(self, *_):
        super().__init__()

    def forward_nodes(self, z, pos, *_args, **_kwargs):
        return pos[:, :1].expand(-1, self.hidden_channels)


class Scores(torch.nn.Module):
    def forward(self, h):
        return h[:, :1]


class CaptureHead(torch.nn.Module):
    def forward(self, h):
        self.last = h.detach().clone()
        return h[:1]


class HeavyCandidateTests(unittest.TestCase):
    def make_model(self, policy="heavy_inference"):
        with patch.object(models, "LiTENNodeEncoder", ToyEncoder):
            model = models.LiTENJointSitePKA({}, {"dropout": 0, "site_candidate_policy": policy})
        model.site_head = Scores()
        model.pka_head = CaptureHead()
        return model

    def batch(self):
        return {
            "z": torch.tensor([6, 7, 1] * 2),
            "pos": torch.tensor([[1., 0., 0.], [2., 0., 0.], [100., 0., 0.]] * 2),
            "conf_batch": torch.tensor([0, 0, 0, 1, 1, 1]),
            "mol_for_conf": torch.tensor([0, 0]),
            "center_indices": torch.tensor([1, 4]),
            "center_atom_idx": torch.tensor([1]), "task_id": torch.tensor([0]),
        }

    def test_inference_masks_h_but_keeps_h_in_graph_and_loss(self):
        model, batch = self.make_model(), self.batch()
        training = model.train()(batch)
        self.assertEqual(training["pred_center_idx"].item(), 2)
        inference = model.eval()(batch)
        self.assertEqual(inference["pred_center_idx"].item(), 1)
        torch.testing.assert_close(training["site_loss"], inference["site_loss"])
        self.assertAlmostEqual(model.pka_head.last[0].item(), 103 / 3, places=5)
        self.assertEqual(model.pka_head.last[8].item(), 2)

    def test_legacy_mode_is_explicit_and_h_only_input_fails(self):
        batch = self.batch()
        self.assertEqual(self.make_model("legacy_all").eval()(batch)["pred_center_idx"].item(), 2)
        batch["z"].fill_(1)
        with self.assertRaisesRegex(ValueError, "heavy atom"):
            self.make_model().eval()(batch)


if __name__ == "__main__":
    unittest.main()
