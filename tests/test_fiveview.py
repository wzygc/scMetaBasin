import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scmetabasin.fiveview import build_fiveview, initialize_gat_weights, main, VIEW_NAMES

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Install the fiveview extra to test attention construction.')
class FiveViewTests(unittest.TestCase):
    def setUp(self):
        self.expression = np.array([[1, 0], [0, 1]], dtype=np.float32)
        self.embedding = np.array([[1, 2], [3, 4]], dtype=np.float32)
        self.weights = {'W': np.eye(2, dtype=np.float32),
                        'a_self': np.zeros((2, 1), dtype=np.float32),
                        'a_neighs': np.zeros((2, 1), dtype=np.float32)}

    def test_attention_and_propagation_have_distinct_residuals(self):
        bank = build_fiveview(self.expression, self.embedding, gat_weights=self.weights)
        self.assertEqual(tuple(bank), VIEW_NAMES)
        np.testing.assert_array_equal(bank[VIEW_NAMES[0]], self.embedding)
        np.testing.assert_array_equal(bank[VIEW_NAMES[1]], [[4, 7], [8, 11]])
        for key in VIEW_NAMES[2:]:
            np.testing.assert_allclose(bank[key], [[3, 5], [5, 7]], rtol=0, atol=1e-6)

    def test_seed_is_explicit_repeatable_and_does_not_change_cpu_rng(self):
        before = torch.random.get_rng_state().clone()
        first = initialize_gat_weights(3, seed=42)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        second = initialize_gat_weights(3, seed=42)
        third = initialize_gat_weights(3, seed=43)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertFalse(np.array_equal(first['W'], third['W']))

    def test_negative_attention_uses_the_recorded_model_slope(self):
        z = np.array([[-1, 1], [1, 1]], dtype=np.float32)
        self.weights['a_neighs'] = np.array([[1], [0]], dtype=np.float32)
        bank = build_fiveview(self.expression, z, gat_weights=self.weights)
        positive_weight = 1 / (1 + np.exp(-1.3))
        attended = np.array([[2 * positive_weight - 1, 1]], dtype=np.float32)
        np.testing.assert_allclose(bank[VIEW_NAMES[1]], attended + 2 * z, rtol=0, atol=5e-7)

    def test_parameter_and_cell_mismatch_are_rejected(self):
        with self.assertRaises(ValueError):
            build_fiveview(self.expression[:1], self.embedding, gat_weights=self.weights)
        with self.assertRaises(ValueError):
            build_fiveview(self.expression, self.embedding, gat_weights={'W': np.eye(3)})
        with self.assertRaises(ValueError):
            build_fiveview(self.expression, self.embedding, gat_weights=self.weights, batch_size=0)

    def test_cli_keeps_duplicate_cell_ids_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / 'x.npy', self.expression)
            np.save(path / 'z.npy', self.embedding)
            np.save(path / 'ids.npy', np.array(['same', 'same']))
            np.savez(path / 'weights.npz', **self.weights)
            args = ['--expression', str(path / 'x.npy'), '--embedding', str(path / 'z.npy'),
                    '--gat-weights', str(path / 'weights.npz'), '--cell-ids', str(path / 'ids.npy'),
                    '--output', str(path / 'bank.npz')]
            main(args)
            with np.load(path / 'bank.npz', allow_pickle=False) as data:
                np.testing.assert_array_equal(data['cell_ids'], ['same', 'same'])
            info = json.loads((path / 'bank.fiveview.json').read_text())
            self.assertEqual(info['gat_parameter_source'], 'supplied')
            self.assertFalse(info['encoder_training'])
            self.assertEqual(len(info['view_hashes']), 5)
            with self.assertRaises(SystemExit):
                main(args)


if __name__ == '__main__':
    unittest.main()
