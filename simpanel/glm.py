# coding=utf-8

from pymc3 import glm


class Glm(object):
    """A more convenient wrapper for glm"""
    def __init__(self, formula, data,
                 priors=None,
                 intercept_prior=None,
                 regressor_prior=None,
                 init_vals=None,
                 family='normal',
                 model=None,
                 **kwargs):

        families = dict(
            normal=glm.families.Normal(),
            student=glm.families.StudentT(),
            binomial=glm.families.Binomial(),
            poisson=glm.families.Poisson()
        )
        if isinstance(family, str):
            family = families[family]

        res = glm.glm(formula,
                      data,
                      priors=priors,
                      intercept_prior=intercept_prior,
                      regressor_prior=regressor_prior,
                      init_vals=init_vals,
                      family=family,
                      model=model,
                      **kwargs)
        self.yest = res[0]
        self.coefs = res[1:]

    @classmethod
    def from_xy(cls, X, y,
                priors=None,
                intercept_prior=None,
                regressor_prior=None,
                init_vals=None,
                family='normal',
                model=None,
                **kwargs):
        import patsy
        import pandas as pd

        name = 'y_target'
        if hasattr(y, 'name'):
            name = y.name or name
        y = pd.Series(y, name=name)
        if not isinstance(X, pd.DataFrame):
            cols = ['x%d' % i for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=cols)
        data = pd.concat([y, X], 1)
        formula = patsy.ModelDesc(
            [patsy.Term([patsy.LookupFactor(y.name)])],
            [patsy.Term([patsy.LookupFactor(p)]) for p in X.columns]
        )
        return cls(formula=formula,
                   data=data,
                   priors=priors,
                   intercept_prior=intercept_prior,
                   regressor_prior=regressor_prior,
                   init_vals=init_vals,
                   family=family,
                   model=model,
                   **kwargs)
