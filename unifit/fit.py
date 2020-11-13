import math
import warnings
from unifit.distributions import distributions


def fit(data):
    '''
    The best-fitting distribution as measured by the Bayesian Information Criterion.
    '''
    with warnings.catch_warnings(record=True) as recorder:
        result = min(
            [
                distribution(*distribution.fit(data))
                for distribution in distributions.values()
            ],
            key=lambda distribution: bic(data, distribution)
        )
    for warning in recorder:
        if not issubclass(warning.category, RuntimeWarning):
            warnings.warn(warning.message, warning.category)
    return result


def bic(data, distribution):
    '''
    Bayesian information criterion.
    See https://en.wikipedia.org/wiki/Bayesian_information_criterion
    '''
    parameter_loss = len(distribution.args) * math.log(len(data))
    likelihood_loss = 2 * distribution.logpdf(data).sum()
    return parameter_loss + likelihood_loss
