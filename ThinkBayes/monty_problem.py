from bayes import Pmf


class Monty(Pmf):

    def __init__(self, hypos):
        super(Monty, self).__init__()
        for hypo in hypos:
            self.Set(hypo, 1)
        self.Normalize()

    def Update(self, data):
        for hypo in self.Values():
            like = self.Likelihood(data, hypo)
            self.Mult(hypo, like)
        self.Normalize()

    def Likelihood(self, data, hypo):
        if hypo == data:
            return 0
        elif hypo == 'A':
            return 0.5
        else:
            return 1


if __name__ == '__main__':
   hypos = 'ABC'
   pmf = Monty(hypos)

   data = 'B'
   pmf.Update(data)

   for hypo, prob in pmf.Items():
       print("{}: {}".format(hypo, prob))