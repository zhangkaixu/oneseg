import sys
class Online :
    def __init__(self, decoder, weights = {}, learner = None, Eval = None):
        self.decoder = decoder
        self.learner = learner
        self.Eval = Eval
        self.weights = weights

    def fit(self, train_x, train_y,
            train_Y = None, subset = None,
            dev_x = None, dev_y = None,
            iterations = 5):
        self.learner.reset()
        for it in range(iterations) :
            if self.Eval : evaluator = self.Eval()
            c = 0
            for c in range(len(train_x)):
                x = train_x[c]
                y = train_y[c]
                if c % 100 == 0 : print("%d (%.1f%%)"%(c, c/len(train_x)*100), end='\r', file = sys.stderr)
                z = self.decoder(x, self.weights)

                if train_Y :
                    Y = train_Y[c]
                    if subset(z, Y) :
                        y = z
                    else :
                        self.decoder(x, self.weights, subset = Y)

                self.learner(x, y, z, self.weights)
                if self.Eval : evaluator(y, z)
            if self.Eval : evaluator.report()

            averaged = self.learner.average(self.weights)

            if self.Eval : evaluator = self.Eval()
            for x, y in zip(dev_x, dev_y) :
                z = self.decoder(x, averaged)
                if self.Eval : evaluator(y, z)
            if self.Eval : evaluator.report()

        self.weights = averaged
        self.learner.reset()

    def predict(self, test_x):
        result_y = []
        for x in test_x :
            y = self.decoder(x, self.weights)
            result_y.append(y)
        return result_y

    def predict_margin(self, test_x):
        results = []
        for x in test_x :
            margins = self.decoder.cal_margins(x, self.weights)
            results.append(margins)
        return results
