import scipy.stats
import unifit


class TestFit:
    def test_cauchy(self):
        data = scipy.stats.cauchy.rvs(size=256)
        unifit.fit(data)
