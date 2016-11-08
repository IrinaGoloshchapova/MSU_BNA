# coding=utf-8

import numpy as np
import pymc3 as pm
import simpanel._glm as glm
from simpanel._families import Normal, StudentT, Poisson, Binomial


class Glm(object):
    """A more convenient wrapper for glm"""
    advifit = None
    trace = None

    def __init__(self, formula, data,
                 priors=None,
                 intercept_prior=None,
                 regressor_prior=None,
                 init_vals=None,
                 family='normal',
                 model=None,
                 name='glm_model'
                 ):
        families = dict(
            normal=Normal,
            student=StudentT,
            binomial=Binomial,
            poisson=Poisson
        )
        if isinstance(family, str):
            family = families[family]()

        y_est, coefs = glm.glm(
                name=name,
                formula=formula,
                data=data,
                priors=priors,
                intercept_prior=intercept_prior,
                regressor_prior=regressor_prior,
                init_vals=init_vals,
                family=family,
                model=model,
                )
        self.y_est = y_est
        self.coefs = coefs

    @classmethod
    def from_xy(cls, X, y,
                priors=None,
                intercept_prior=None,
                regressor_prior=None,
                init_vals=None,
                family='normal',
                model=None,
                name='glm_model'):
        import patsy
        import pandas as pd

        _name = 'y_target'
        if hasattr(y, 'name'):
            _name = y.name or _name
        y = pd.Series(y, name=_name)
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
                   name=name)

    def advi(self, **kwargs):
        self.advifit = pm.advi(**kwargs)
        return self.advifit

    def nuts(self, draws=300, njobs=4, model=None, **kwargs):
        import numpy as np
        if not self.advifit:
            fit = self.advi(verbose=False)
        else:
            fit = self.advifit
        model = pm.modelcontext(model)
        step = pm.NUTS(scaling=np.power(model.dict_to_array(fit.stds), 2), is_cov=True)
        trace = pm.sample(draws=draws, njobs=njobs, step=step, start=fit.means, **kwargs)
        self.trace = trace
        return trace
