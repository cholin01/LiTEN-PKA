import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from liten_pka import conformer_generation as cg
from liten_pka.chem import embed_rdkit_conformers


class ConformerRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.table = self.root / 'pool.csv'
        self.cache = self.root / 'cache'
        self.config = {
            'data': {'smiles_column': 'smiles'},
            'conformers': {'n_conformers': 1, 'random_seed': 2026,
                           'max_embed_attempts': 100, 'prune_rms_thresh': -1.0},
        }

    def generate(self, smiles, **kwargs):
        pd.DataFrame({'smiles': smiles}).to_csv(self.table, index=False)
        return cg.generate_conformers_for_tables(
            [self.table], self.cache, self.config, workers=1, **kwargs)

    def test_reuses_existing_and_regenerates_missing_files(self):
        manifest = self.generate(['C', 'CC'])
        original = pd.read_csv(manifest).set_index('input_smiles')
        retained_file = Path(original.loc['C', 'conformer_npz'])
        retained_bytes = retained_file.read_bytes()
        Path(original.loc['CC', 'conformer_npz']).unlink()
        with patch.object(cg, 'embed_rdkit_conformers', wraps=embed_rdkit_conformers) as embed:
            self.generate(['C', 'CC', 'O'])
        self.assertEqual([c.args[0] for c in embed.call_args_list], ['CC', 'O'])
        self.assertEqual(retained_file.read_bytes(), retained_bytes)
        result = pd.read_csv(manifest)
        self.assertEqual(set(result.input_smiles), {'C', 'CC', 'O'})
        self.assertTrue(all(Path(p).is_file() for p in result.conformer_npz))

    def test_generation_failure_is_logged_and_retried(self):
        def fail_oxygen(smiles, **kwargs):
            if smiles == 'O':
                raise ValueError('embedding failed')
            return embed_rdkit_conformers(smiles, **kwargs)

        with patch.object(cg, 'embed_rdkit_conformers', side_effect=fail_oxygen):
            manifest = self.generate(['C', 'O', 'O'])
        self.assertEqual(pd.read_csv(manifest).input_smiles.tolist(), ['C'])
        failures = pd.read_csv(self.cache / 'failures.csv')
        self.assertEqual(failures.smiles.tolist(), ['O'])
        self.assertIn('embedding failed', failures.error.iloc[0])
        self.generate(['C', 'O'])
        self.assertEqual(len(pd.read_csv(manifest)), 2)
        self.assertTrue(pd.read_csv(self.cache / 'failures.csv').empty)

    def test_all_failures_leave_readable_manifest(self):
        with patch.object(cg, 'embed_rdkit_conformers', side_effect=ValueError('failed')):
            manifest = self.generate(['C', 'O'])
        self.assertTrue(pd.read_csv(manifest).empty)
        self.assertEqual(pd.read_csv(manifest).columns.tolist(), cg.MANIFEST_COLUMNS)
        self.assertEqual(len(pd.read_csv(self.cache / 'failures.csv')), 2)


if __name__ == '__main__':
    unittest.main()
