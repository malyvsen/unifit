import math
import warnings
from tqdm.auto import tqdm
from unifit.distributions import distributions


def fit(data, *, distributions=distributions, loading_bar=True, return_name=False):
    '''
    The best-fitting distribution as measured by the Bayesian Information Criterion.
    '''
    named_distributions = (
        distributions
        if isinstance(distributions, dict)
        else {
            distribution.__class__.__name__.replace('_gen', ''): distribution
            for distribution in distributions
        }
    ).items()
    iterator = tqdm(named_distributions) if loading_bar else named_distributions

    with warnings.catch_warnings(record=True) as recorder:
        fits = {}
        for name, distribution in iterator:
            if loading_bar:
                iterator.set_description(f'Fitting {name}')
            try:
                fits[name] = distribution(*distribution.fit(data))
            except ValueError:
                pass
        best_name, best_fit = min(
            fits.items(),
            key=lambda named_fit: bic(data, named_fit[1])
        )

    for warning in recorder:
        if not issubclass(warning.category, RuntimeWarning):
            warnings.warn(warning.message, warning.category)

    if return_name:
        return best_name, best_fit
    return best_fit


def bic(data, distribution):
    '''
    Bayesian information criterion.
    See https://en.wikipedia.org/wiki/Bayesian_information_criterion
    '''
    parameter_loss = len(distribution.args) * math.log(len(data))
    likelihood_loss = -2 * distribution.logpdf(data).sum()
    return parameter_loss + likelihood_loss
