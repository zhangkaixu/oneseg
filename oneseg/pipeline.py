class Pipeline :
    def __init__(self,items):
        self.items = items
    def fit(self, xs, ys):
        pass
    def predict(self, xs):
        data_in = xs
        for name, model in self.items :
            data_out = None

