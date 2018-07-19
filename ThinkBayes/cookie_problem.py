from bayes import Pmf


class Cookie(Pmf):

    def __init__(self, hypos):
        super(Cookie, self).__init__()

        self.mixes = {
            'Bowl 1': dict(vanilla=0.75, chocolate=0.25),
            'Bowl 2': dict(vanilla=0.5, chocolate=0.5)
        }

        for hypo in hypos:
            self.Set(hypo, 1)
        self.Normalize()

    def Update(self, data):
        for hypo in self.Values():
            like = self.Likelihood(data, hypo)
            self.Mult(hypo, like)
        self.Normalize()

    def Likelihood(self, data, hypo):
        mix = self.mixes[hypo]
        like = mix[data]
        return like


if __name__ == '__main__':
    hypos = ['Bowl 1', 'Bowl 2']
    pmf = Cookie(hypos)

    # pmf.Update('vanilla')
    # pmf.Update('chocolate')
    dataset = ['vanilla', 'chocolate', 'vanilla']
    for data in dataset:
        pmf.Update(data)

    for hypo, prob in pmf.Items():
        print("{}: {}".format(hypo, prob))
