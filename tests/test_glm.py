# coding=utf-8

import unittest

import os
import pandas as pd
import pymc3 as pm
from simpanel.glm import Glm


ROOT = os.path.dirname(__file__)


class TestGlm(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv(os.path.join(ROOT, 'data', 'testdata.csv'))
        self.y, self.X = data.iloc[:, 0], data.iloc[:, 1:-1]
        self.mdata = data.iloc[:10]
        self.data = data

    def test_init(self):
        import patsy
        formula = patsy.ModelDesc(
            [patsy.Term([patsy.LookupFactor(self.y.name)])],
            [patsy.Term([patsy.LookupFactor(p)]) for p in self.X.columns]
        )
        with pm.Model():
            _ = Glm(formula, self.mdata)

    def test_init_from_xy(self):
        import numpy as np
        with pm.Model():
            _ = Glm.from_xy(self.X, self.y)
        with pm.Model():
            _ = Glm.from_xy(np.asarray(self.X), self.y)
        with pm.Model():
            _ = Glm.from_xy(self.X, np.asarray(self.y))
        with pm.Model():
            _ = Glm.from_xy(np.asarray(self.X), np.asarray(self.y))


if __name__ == '__main__':
    unittest.main()
