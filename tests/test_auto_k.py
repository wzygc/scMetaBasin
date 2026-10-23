import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scmetabasin import scMetaBasin, load_representations
from scmetabasin.cli import main, make_parser
from scmetabasin.result import save_scmetabasin_results
from scmetabasin._auto_k_score import select_supported_area


def decision(k):
    return dict(target_K=k, info={'status': 'estimated' if k else 'no_supported_split'},
                ranking=pd.DataFrame(columns=['K','supported_area','supported_levels']),
                windows=pd.DataFrame(columns=['K','start_index','end_index','consecutive_levels']),
                paths=pd.DataFrame(columns=['batch','representation','K','noise_index','stadion','between']),
                between=pd.DataFrame(columns=['batch','representation','K','noise_index','between']))


class AutomaticKTests(unittest.TestCase):
    def setUp(self):
        x=np.vstack([np.ones((10,3))*center for center in [-4,0,4]])
        self.bank={'a':x,'b':x*2,'c':x+1}

    def test_K_is_optional_in_cli(self):
        args=make_parser().parse_args(['--representations','bank.npz','--output','out'])
        self.assertIsNone(args.k)

    def test_manual_mode_never_estimates_K(self):
        with patch('scmetabasin.automatic.estimate_k',side_effect=AssertionError('automatic path called')):
            model=scMetaBasin(n_clusters=3,seeds=range(6))
            y=model.fit_predict(self.bank)
        self.assertEqual(model.k_mode_,'manual')
        self.assertEqual(len(np.unique(y)),3)
        self.assertIsNone(model.estimated_K_)

    def test_repeated_auto_fit_reestimates_and_clears_old_output(self):
        model=scMetaBasin(seeds=range(6))
        with patch('scmetabasin.automatic.estimate_k',side_effect=[decision(3),decision(None)]) as estimate:
            self.assertIsNotNone(model.fit_predict(self.bank))
            self.assertEqual(model.estimated_K_,3)
            self.assertIsNone(model.fit_predict(self.bank))
            self.assertEqual(estimate.call_count,2)
        self.assertIsNone(model.labels_)
        self.assertIsNone(model.realized_K_)
        self.assertFalse(hasattr(model,'selected_meta_id_'))
        self.assertEqual(model.status_,'no_supported_split')

    def test_no_labels_cannot_be_serialized_as_a_constant_partition(self):
        model=scMetaBasin()
        with patch('scmetabasin.automatic.estimate_k',return_value=decision(None)):
            model.fit_predict(self.bank)
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError,'No cell labels'):
            save_scmetabasin_results(model,folder)

    def test_unsupported_auto_cli_returns_status_and_no_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'input.npz'; output=Path(folder)/'out'
            np.savez(path,**self.bank)
            with patch('scmetabasin.automatic.estimate_k',return_value=decision(None)), contextlib.redirect_stdout(io.StringIO()):
                code=main(['--representations',str(path),'--output',str(output)])
            self.assertEqual(code,3)
            self.assertFalse((output/'cluster_labels.csv').exists())
            self.assertEqual(json.loads((output/'scmetabasin_summary.json').read_text())['status'],'no_supported_split')

    def test_manual_core_failure_still_raises(self):
        model=scMetaBasin(n_clusters=3)
        with patch.object(model,'_fit_known_k',side_effect=RuntimeError('No selectable meta-basin with >=2 representations.')):
            with self.assertRaisesRegex(RuntimeError,'No selectable'):
                model.fit_predict(self.bank)

    def test_directory_input_and_cell_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            for name,x in self.bank.items():
                np.save(Path(folder)/f'{name}.npy',x)
            ids=np.array([f'cell_{i}' for i in range(30)])
            np.save(Path(folder)/'cell_ids.npy',ids)
            bank,loaded=load_representations(folder)
            self.assertEqual(sorted(bank),sorted(self.bank))
            np.testing.assert_array_equal(ids,loaded)
            np.save(Path(folder)/'truth.npy',np.zeros(30))
            with self.assertRaisesRegex(ValueError,'2-D'):
                load_representations(folder)

    def test_invalid_K_workers_and_cell_count(self):
        for k in [True,0,1,2.5]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                scMetaBasin(n_clusters=k)
        for count in [0,False]:
            with self.subTest(workers=count), self.assertRaises(ValueError):
                scMetaBasin(auto_k_workers=count)
        with self.assertRaisesRegex(ValueError,'exceeds'):
            scMetaBasin(n_clusters=31).fit_predict(self.bank)
        with self.assertRaisesRegex(ValueError,'>=10'):
            scMetaBasin().fit_predict({'a':np.ones((8,2)),'b':np.ones((8,2))})

    def test_no_split_path_and_one_point_spike(self):
        rows=[dict(batch=b,representation='v',K=k,noise_index=t,between=1.,
                   stadion=.9 if k==2 and t==1 else 0.)
              for b in [0,1] for k in [1,2] for t in range(5)]
        k,ranking,windows=select_supported_area(pd.DataFrame(rows))
        self.assertEqual(k,1)
        self.assertTrue(windows.empty)
        self.assertEqual(ranking.iloc[0].supported_area,0)

    def test_auto_serialization_does_not_call_estimated_K_known(self):
        model=scMetaBasin(seeds=range(6))
        with patch('scmetabasin.automatic.estimate_k',return_value=decision(3)):
            model.fit_predict(self.bank)
        with tempfile.TemporaryDirectory() as folder:
            save_scmetabasin_results(model,folder)
            summary=json.loads((Path(folder)/'scmetabasin_summary.json').read_text())
            self.assertIsNone(summary['known_K'])
            self.assertEqual(summary['estimated_K'],3)
            self.assertEqual(summary['realized_K'],3)


if __name__=='__main__':
    unittest.main()
