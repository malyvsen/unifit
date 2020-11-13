import math
from unifit.distributions import distributions


def fit(data):
    '''
    The best-fitting distribution as measured by the Bayesian Information Criterion.
    '''
    return min(
        [
            distribution(*distribution.fit(data))
            for distribution in distributions.values()
        ],
        key=lambda distribution: bic(data, distribution)
    )


def bic(data, distribution):
    '''
    Bayesian information criterion.
    See https://en.wikipedia.org/wiki/Bayesian_information_criterion
    '''
    parameter_loss = len(distribution.args) * math.log(len(data))
    likelihood_loss = 2 * distribution.logpdf(data).sum()
    return parameter_loss + likelihood_loss
