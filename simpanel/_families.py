import numbers
from copy import copy

import theano.tensor as tt
from pymc3.model import modelcontext
from pymc3 import distributions as pm_dists

__all__ = ['Normal', 'StudentT', 'Binomial', 'Poisson']

# Define link functions

# Hack as assigning a function in the class definition automatically binds
# it as a method.


class Identity(object):

    def __call__(self, x):
        return x

identity = Identity()
logit = tt.nnet.sigmoid
inverse = tt.inv
exp = tt.exp


class Family(object):
    """Base class for Family of likelihood distribution and link functions.
    """
    priors = {}
    link = None
    parent = None
    likelihood = None

    def __init__(self, **kwargs):
        # Overwrite defaults
        for key, val in kwargs.items():
            if key == 'priors':
                self.priors = copy(self.priors)
                self.priors.update(val)
            else:
                setattr(self, key, val)

    def _get_priors(self, name, model=None):
        """Return prior distributions of the likelihood.

        Returns
        -------
        dict : mapping name -> pymc3 distribution
        """
        model = modelcontext(model)
        priors = {}
        for key, val in self.priors.items():
            if isinstance(val, numbers.Number):
                priors[key] = val
            else:
                priors[key] = model.Var('{}_{}'.format(name, key), val)

        return priors

    def create_likelihood(self, y_est, y_data, name, model=None):
        """Create likelihood distribution of observed data.

        Parameters
        ----------
        y_est : theano.tensor
            Estimate of dependent variable
        y_data : array
            Observed dependent variable
        """
        priors = self._get_priors(model=model, name=name)
        # Wrap y_est in link function
        priors[self.parent] = self.link(y_est)
        return self.likelihood('{}_y'.format(name), observed=y_data, **priors)

    def __repr__(self):
        return """Family {klass}:
    Likelihood   : {likelihood}({parent})
    Priors       : {priors}
    Link function: {link}.""".format(
            klass=self.__class__,
            likelihood=self.likelihood.__name__,
            parent=self.parent, priors=self.priors,
            link=self.link)


class StudentT(Family):
    link = identity
    likelihood = pm_dists.StudentT
    parent = 'mu'
    priors = {'lam': pm_dists.HalfCauchy.dist(beta=10, testval=1.),
              'nu': 1}


class Normal(Family):
    link = identity
    likelihood = pm_dists.Normal
    parent = 'mu'
    priors = {'sd': pm_dists.HalfCauchy.dist(beta=10, testval=1.)}


class Binomial(Family):
    link = logit
    likelihood = pm_dists.Bernoulli
    parent = 'p'
    priors = {'p': pm_dists.Beta.dist(alpha=1, beta=1)}


class Poisson(Family):
    link = exp
    likelihood = pm_dists.Poisson
    parent = 'mu'
    priors = {'mu': pm_dists.HalfCauchy.dist(beta=10, testval=1.)}
