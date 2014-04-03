from oneseg.sequence_labeling import Decoder, Feature_Generator
from oneseg.online_learner import Learner
from oneseg.online_model import Online
import pickle

def wrap_eval(Eval, codec):

    class Wrapped_Eval(Eval):
        def __init__(self):
            super().__init__()
        def __call__(self, y, z):
            x = '.' * len(y)
            y = codec.decode(x, y)
            z = codec.decode(x, z)
            super().__call__(y, z)
    return Wrapped_Eval


class Character_Labeler(Online) :
    def __init__(self, 
            Codec, Evaluator,
            bigrams = None, bigram_vectors = None):
        self.codec = Codec()
        feature_generator = Feature_Generator(bigrams = bigrams, bigram_vectors = bigram_vectors)
        decoder = Decoder(feature_generator)
        learner = Learner(feature_generator)

        #self.Eval = wrap_eval(Evaluator, self.codec)
        class Wrapped_Eval(Evaluator):
            def __init__(self, codec):
                self.codec = codec
                super().__init__()
            def __call__(self, y, z):
                x = '.' * len(y)
                y = self.codec.decode(x, y)
                z = self.codec.decode(x, z)
                super().__call__(y, z)

        self.Eval = lambda : Wrapped_Eval(self.codec)

        super().__init__(decoder, learner = learner, 
                Eval = self.Eval,
                weights = {})

    def fit(self, xs, ys, 
            dev_x = None, dev_y = None, 
            **args):
        y_seq = [self.codec.encode(y) for y in ys]

        self.decoder.tag_size = self.codec.get_size()
        self.learner.tag_size = self.codec.get_size()

        dev_y_seq = [self.codec.encode(y) for y in dev_y] if dev_y else None

        super().fit(xs, y_seq, dev_x = dev_x, dev_y = dev_y_seq, **args)

    def predict(self, xs, **args):
        yys = super().predict(xs, **args)
        yys = [self.codec.decode(x, y) for x, y in zip(xs,yys)]
        return yys

    def predict_with_margin(self, test_x, as_sequence = False):
        results = []
        margins = []
        for x in show_progress(test_x) :
            emissions, transitions, alphas, betas, result = self.decoder(x, self.weights, with_details = True)
            margin = self.decoder.cal_margin(alphas, betas, emissions, as_sequence = as_sequence)
            results.append(self.codec.decode(x, result))
            margins.append(margin)
        return results, margins
