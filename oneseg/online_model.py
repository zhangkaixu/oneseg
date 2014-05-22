from oneseg.utils import show_progress 

class Online(object) :
    def __init__(self, decoder, weights = {}, learner = None, Eval = None):
        self.decoder = decoder
        self.learner = learner
        self.Eval = Eval
        self.evaluator = Eval
        self.weights = weights

    def fit(self, train_x, train_y,
            train_Y = None, subset = None,
            dev_x = None, dev_y = None,
            iterations = 5):
        self.learner.reset()
        for it in range(iterations) :
            if self.evaluator : self.evaluator.reset()
            for c in show_progress(len(train_x)):
                x = train_x[c]
                y = train_y[c]
                z = self.decoder(x, self.weights)

                if train_Y : 
                    Y = train_Y[c]
                    if subset(z, Y) :
                        y = z
                    else :
                        y = self.decoder(x, self.weights, subset = Y)

                self.learner(x, y, z, self.weights)
                if self.evaluator : self.evaluator(y, z)
            if self.evaluator : self.evaluator.report()

            averaged = self.learner.average(self.weights)
            #averaged = self.weights

            if dev_x is not None :
                if self.evaluator : self.evaluator.reset()
                for x, y in show_progress(zip(dev_x, dev_y), len(dev_x)) :
                    z = self.decoder(x, averaged)
                    if self.evaluator : self.evaluator(y, z)
                if self.evaluator : self.evaluator.report()

        #self.evaluator = evaluator

        self.weights = averaged
        self.learner.reset()

    def predict(self, test_x):
        result_y = []
        for x in show_progress(test_x) :
            y = self.decoder(x, self.weights)
            result_y.append(y)
        return result_y

