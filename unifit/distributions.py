import scipy.stats


impossible = [
    'erlang',
    'frechet_l',
    'frechet_r',
    'ksone',
    'kstwo',
    'kstwobign',
]
difficult = [
    'alpha',
    'arcsine',
    'argus',
    'beta',
    'betaprime',
    'burr12',
    'exponnorm',
    'exponpow',
    'exponweib',
    'f',
    'foldcauchy',
    'foldnorm',
    'gausshyper',
    'genextreme',
    'genhalflogistic',
    'geninvgauss',
    'genlogistic',
    'halfcauchy',
    'invgamma',
    'invweibull',
    'levy',
    'levy_l',
    'levy_stable',
    'loggamma',
    'loglaplace',
    'lomax',
    'mielke',
    'moyal',
    'nakagami',
    'ncf',
    'nct',
    'pareto',
    'powerlognorm',
    'rice',
    'tukeylambda',
]
distributions = {
    name: distribution
    for name, distribution in {
        attr: getattr(scipy.stats, attr)
        for attr in dir(scipy.stats)
    }.items()
    if isinstance(distribution, scipy.stats.rv_continuous)
    if name not in impossible + difficult
}
