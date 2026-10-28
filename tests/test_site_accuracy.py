import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from liten_pka import model as models
from liten_pka import train_joint_site_pka as train


class ToyEncoder(torch.nn.Module):
    hidden_channels = 8

    def __init__(self, *_args):
        super().__init__()

    def forward_nodes(self, z, *_args, **_kwargs):
        return torch.zeros(len(z), self.hidden_channels)


class CountingModel(torch.nn.Module):
    def __init__(self, *_args):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        labels = batch['center_atom_idx']
        valid = labels >= 0
        return {
            'pka_pred': self.value.expand_as(batch['y']),
            'pred_center_idx': torch.zeros_like(labels),
            'site_loss': self.value.square(),
            'site_correct_count': ((labels == 0) & valid).sum(),
            'site_count': valid.sum(),
        }


def batches(batch_size, labels):
    return [{'center_atom_idx': torch.tensor(labels[i:i + batch_size]),
             'y': torch.arange(i, min(i + batch_size, len(labels)), dtype=torch.float),
             'mol_ids': [str(j) for j in range(i, min(i + batch_size, len(labels)))]}
            for i in range(0, len(labels), batch_size)]


class SiteAccuracyTests(unittest.TestCase):
    def test_model_counts_exclude_unlabeled_sites(self):
        with patch.object(models, 'LiTENNodeEncoder', ToyEncoder):
            model = models.LiTENJointSitePKA({}, {'dropout': 0})
        for p in model.site_head.parameters():
            torch.nn.init.zeros_(p)
        batch = {
            'z': torch.tensor([6, 1] * 3), 'pos': torch.zeros(6, 3),
            'conf_batch': torch.tensor([0, 0, 1, 1, 2, 2]),
            'mol_for_conf': torch.tensor([0, 1, 2]),
            'center_indices': torch.tensor([0, 2, 4]),
            'center_atom_idx': torch.tensor([0, 1, -1]), 'task_id': torch.zeros(3, dtype=torch.long),
        }
        result = model(batch)
        self.assertEqual(result['site_correct_count'].item(), 1)
        self.assertEqual(result['site_count'].item(), 2)

    def test_evaluation_accuracy_does_not_depend_on_batch_size(self):
        # One mistake in the first full batch, plus an unlabeled final record.
        labels = [1] + [0] * 34 + [-1]
        for batch_size in [8, 32, 36]:
            with self.subTest(batch_size=batch_size):
                metrics, predictions = train.evaluate(CountingModel(), batches(batch_size, labels), torch.device('cpu'))
                self.assertEqual(metrics['site_correct_count'], 34)
                self.assertEqual(metrics['site_count'], 35)
                self.assertAlmostEqual(metrics['site_acc'], 34 / 35)
                self.assertEqual(len(predictions), 36)

    def test_training_log_uses_total_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                'project': {'device': 'cpu'},
                'data': {'train_file': 'train.csv', 'val_file': 'val.csv', 'conformer_dir': tmp},
                'liten': {}, 'model': {},
                'train': {'lr': 0.001, 'epochs': 1, 'checkpoint_dir': tmp + '/models',
                          'report_dir': tmp + '/reports'},
            }
            loader = batches(32, [1] + [0] * 34)
            with patch.object(train, 'LiTENJointSitePKA', CountingModel), \
                 patch.object(train, 'build_loader', return_value=loader), \
                 patch.object(train, 'evaluate', return_value=({'mae': 1.0}, pd.DataFrame())), \
                 patch.object(train, 'tqdm', side_effect=lambda rows, **kwargs: rows):
                train.train_one_seed(config, 0, 'counts')
            report = json.loads((Path(tmp) / 'reports/counts/seed_0/history.jsonl').read_text())
            self.assertEqual(report['train_site_correct_count'], 34)
            self.assertEqual(report['train_site_count'], 35)
            self.assertAlmostEqual(report['train_site_acc'], 34 / 35)


if __name__ == '__main__':
    unittest.main()
