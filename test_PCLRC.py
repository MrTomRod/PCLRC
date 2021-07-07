from unittest import TestCase
import pandas as pd
import numpy as np
from PCLRC import PCLRC


class TestPCLRC(TestCase):
    def test_r_vs_py(self):
        test_df = pd.read_csv('data/input.tsv', sep='\t', index_col=0)

        result_py = PCLRC.pclrc(
            test_df,
            method='bicor',
            n_iter=1000,
            frac=0.75,
            rank_thr=0.3
        )
        assert all(result_py.index == result_py.columns)

        result_r = pd.read_csv('data/r_pclrc_100_iterations.tsv', sep='\t', index_col=0)
        assert all(result_r.index == result_r.columns)
        assert all(result_py.index == result_r.index)

        x, y = result_r.shape

        differences = result_py != result_r
        differences = np.sum(differences.to_numpy())
        print(f'{round(differences / (x * y) * 100, 2)} % different')
