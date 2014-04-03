from oneseg.utils import show_progress # 训练、解码比较慢，显式进度，增加耐心

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
            for c in show_progress(len(train_x)):
                x = train_x[c]
                y = train_y[c]
                z = self.decoder(x, self.weights)

                if train_Y : # 训练时
                    Y = train_Y[c]
                    if subset(z, Y) :
                        y = z
                    else :
                        y = self.decoder(x, self.weights, subset = Y)

                self.learner(x, y, z, self.weights)
                if self.Eval : evaluator(y, z)
            if self.Eval : evaluator.report()

            averaged = self.learner.average(self.weights)
            #averaged = self.weights

            if dev_x is not None :
                if self.Eval : evaluator = self.Eval()
                for x, y in show_progress(zip(dev_x, dev_y), len(dev_x)) :
                    z = self.decoder(x, averaged)
                    if self.Eval : evaluator(y, z)
                if self.Eval : evaluator.report()

        self.evaluator = evaluator

        self.weights = averaged
        self.learner.reset()

    def predict(self, test_x):
        result_y = []
        for x in show_progress(test_x) :
            y = self.decoder(x, self.weights)
            result_y.append(y)
        return result_y

