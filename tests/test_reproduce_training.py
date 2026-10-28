import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from liten_pka.dataset import PKADataset

spec = importlib.util.spec_from_file_location(
    'reproduce_train', Path(__file__).resolve().parents[1] / 'scripts/reproduce_train.py')
reproduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduce)


class ReproduceTrainingTests(unittest.TestCase):
    def test_failed_conformers_are_counted_and_skipped_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root / 'pool.csv'
            pd.DataFrame({
                'mol_id': [str(i) for i in range(20)],
                'smiles': ['C' * (i + 1) for i in range(20)],
                'pka': [1.0] * 20, 'center_atom_idx': [0] * 20, 'task': ['acid'] * 20,
            }).to_csv(pool, index=False)
            config_path = root / 'config.yaml'
            config_path.write_text(yaml.safe_dump({
                'data': {'train_pool_file': str(pool), 'smiles_column': 'smiles'},
                'conformers': {},
            }))

            def generate(tables, directory, *_args, **_kwargs):
                # Emulate one embedding failure while retaining the other molecules.
                rows = pd.read_csv(tables[0])
                manifest = Path(directory) / 'manifest.csv'
                pd.DataFrame({'input_smiles': rows.smiles.iloc[1:],
                              'conformer_npz': ['unused.npz'] * 19}).to_csv(manifest, index=False)
                return manifest

            def fit(config, *_args):
                manifest = Path(config['data']['conformer_dir']) / 'manifest.csv'
                train = PKADataset(config['data']['train_file'], manifest)
                val = PKADataset(config['data']['val_file'], manifest)
                self.assertEqual(len(train) + len(val), 19)
                return {'trained': True}

            args = ['reproduce_train.py', '--config', str(config_path), '--mode', 'known_ic',
                    '--conformer-dir', str(root / 'cache'), '--report-dir', str(root / 'reports'),
                    '--checkpoint-dir', str(root / 'models'), '--device', 'cpu']
            with patch('sys.argv', args), \
                 patch.object(reproduce, 'generate_conformers_for_tables', side_effect=generate), \
                 patch.object(reproduce, 'train_known_ic', side_effect=fit) as trainer:
                reproduce.main()
            trainer.assert_called_once()
            summary = json.loads((root / 'reports/reproduce_train_summary.json').read_text())
            counts = summary['conformers'].values()
            self.assertEqual(sum(c['total_records'] for c in counts), 20)
            self.assertEqual(sum(c['skipped_records'] for c in counts), 1)
            skipped = pd.read_csv(summary['skipped_conformer_records'])
            self.assertEqual(skipped.smiles.tolist(), ['C'])


if __name__ == '__main__':
    unittest.main()
