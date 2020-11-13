import scipy.stats
import unifit


class TestFit:
    data = scipy.stats.cauchy.rvs(size=256)


    def test_basic(self):
        unifit.fit(self.data)


    def test_unnamed(self):
        unifit.fit(
            self.data,
            distributions=unifit.distributions.values()
        )
