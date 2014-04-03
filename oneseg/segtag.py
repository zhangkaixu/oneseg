from oneseg.character_labeling import Character_Labeler
from oneseg.utils import Indexer
from oneseg.segtag_evaluator import SegTag_Evaluator

def load_std_examples(filename):
    xs = []
    ys = []
    for line in open(filename):
        line = line.split()
        line = [tuple(x.split('_')) for x in line]
        raw = ''.join(x[0] for x in line)
        xs.append(raw)
        ys.append(line)
    return xs, ys 

class Codec :
    def __init__(self):
        self.indexer = Indexer()
    def get_size(self):
        return len(self.indexer.list)
    def encode(self, y):
        tags = []
        for word, tag in y :
            if len(word) == 1 :
                tags.append(('S',tag))
            else :
                tags += [('B',tag)] + [('M',tag)] * (len(word)-2) + [('E',tag)]
        tags = [self.indexer(tag) for tag in tags]
        return tags
    def decode(self, xs, tags):
        tags = [self.indexer.list[tag] for tag in tags]
        sentence = []
        cache = []
        for i in range(len(xs)):
            x = xs[i]
            tag = tags[i]
            cache.append(x)
            if (tag[0] == 'E') or (tag[0] == 'S') or (i == len(xs) - 1) :
                sentence.append((''.join(cache),tag[1]))
                cache = []
        return sentence

class Base_SegTag(Character_Labeler):
    def __init__(self, **args):
        codec = Codec()
        evaluator = SegTag_Evaluator(codec)
        super().__init__(codec, evaluator, **args)
