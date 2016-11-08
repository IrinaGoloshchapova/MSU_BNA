# coding=utf-8
from collections import OrderedDict
import pandas as pd
import pymc3 as pm
from simpanel.glm import Glm


class SimPanel(object):
    def __init__(self, name, X, y, sim, idindex=None, idcol=None):
        if (not isinstance(X, pd.DataFrame) or
                not isinstance(y, pd.Series)):
            raise TypeError('Please provide X as pd.DataFrame and y as pd.Series')
        if idcol is None or idindex is None:
            raise ValueError('Please provide idcol or idindex')
        self.name = name
        self.groups = OrderedDict()
        self.advifits = OrderedDict()
        full = pd.concat([X, y], 1)  # type: pd.DataFrame
        level = idindex
        if idcol:
            full.set_index(idcol, append=True, inplace=True)
            level = idcol
        for label, df in full.groupby(level=level):
            self.groups[label] = df
        self.sim = sim

    @property
    def data(self):
        return pd.concat(self.groups.values())
