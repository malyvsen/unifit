import scipy.stats


distributions = {
    name: distribution
    for name, distribution in {
        attr: getattr(scipy.stats, attr)
        for attr in dir(scipy.stats)
    }.items()
    if isinstance(distribution, scipy.stats.rv_continuous)
}
