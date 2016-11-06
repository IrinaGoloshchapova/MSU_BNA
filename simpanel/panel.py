# coding=utf-8
import pandas as pd
from simpanel.glm import Glm


class SimPanel(object):
    def __init__(self, X, y, groupid, sim):
        if (not isinstance(X, pd.DataFrame) or
                not isinstance(y, pd.Series)):
            raise TypeError('Please provide X as pd.DataFrame and y as pd.Series')
        full = pd.concat([X, y], 1)  # type: pd.DataFrame
        full.set_index(groupid, append=True, inplace=True)
        self.groups = dict()
        for label, df in full.groupby(level=groupid):
            self.groups[label] = df

    @property
    def data(self):
        return pd.concat(self.groups.values())
