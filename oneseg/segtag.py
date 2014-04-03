import time

from oneseg.online_model import Online
from oneseg.utils import Indexer
from oneseg.online_learner import Learner
from oneseg.sequence_labeling import Decoder, Feature_Generator
from oneseg.utils import show_progress # 训练、解码比较慢，显式进度，增加耐心

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
    def encode(self, y):
        tags = []
        for word, tag in y :
            #tag = 'x'
            if len(word) == 1 :
                tags.append(('S',tag))
            else :
                tags += [('B',tag)] + [('M',tag)] * (len(word)-2) + [('E',tag)]
        tags = [self.indexer(tag) for tag in tags]
        return tags
    def decode(self, xs, tags):
        tags = [self.indexer.list[tag] for tag in tags]
        #print(tags)
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

class SegTag_Evaluator :
    def __init__(self, codec):
        self.std,self.rst,self.cor=0,0,0
        self.seg_std,self.seg_rst,self.seg_cor=0,0,0
        self.codec = codec
        self.start_time=time.time()

    def _gen_set(self,words, seg_only = False):
        offset=0
        word_set=set()
        for word in words:
            word_set.add((offset,word))
            if not seg_only :
                offset+=len(word[0])
            else :
                offset+=len(word)
        return word_set
        
    def __call__(self, y, z) :
        #print(y)
        #print(z)
        x = '.' * len(y)
        y = self.codec.decode(x, y)
        z = self.codec.decode(x, z)
        std,rst=self._gen_set(y),self._gen_set(z)
        #print(list(sorted(std)))
        #print(list(sorted(rst)))
        
        self.std+=len(std)
        self.rst+=len(rst)
        self.cor+=len(std&rst)
        d = len(std&rst)
        std,rst=self._gen_set([yy[0] for yy in y], seg_only=True), self._gen_set([zz[0]for zz in z], seg_only=True)
        #print(list(sorted(std)))
        #print(list(sorted(rst)))
        self.seg_std+=len(std)
        self.seg_rst+=len(rst)
        self.seg_cor+=len(std&rst)
        seg_d = len(std&rst)
        #if d != seg_d :
        #    print(self.std,self.rst,self.cor)
        #    print(self.seg_std,self.seg_rst,self.seg_cor)
        return

    #def report(self) :
    def report(self, quiet = False):
        precision=self.cor/self.rst if self.rst else 0
        recall=self.cor/self.std if self.std else 0
        f1=2*precision*recall/(precision+recall) if precision+recall!=0 else 0
        precision=self.seg_cor/self.seg_rst if self.seg_rst else 0
        recall=self.seg_cor/self.seg_std if self.seg_std else 0
        seg_f1=2*precision*recall/(precision+recall) if precision+recall!=0 else 0
        if not quiet :
            print("历时: %.2f秒 答案词数: %i 结果词数: %i 正确词数: %i F值: %.4f seg_f: %.4f"
                    %(time.time()-self.start_time,self.std,self.rst,self.cor,f1,seg_f1))
        #input()
        return f1
    pass

class Eval_Gen :
    def __init__(self, codec):
        self.codec = codec
    def __call__(self):
        return SegTag_Evaluator(self.codec)

class Base_SegTag(Online) :
    def __init__(self, bigrams = None, bigram_vectors = None):
        tag_size = 4 # for cws only
        feature_generator = Feature_Generator(bigrams = bigrams, bigram_vectors = bigram_vectors)
        decoder = Decoder(feature_generator, tag_size = tag_size)
        learner = Learner(feature_generator, tag_size = tag_size)
        self.codec = Codec()
        super(Base_SegTag, self).__init__(decoder, learner = learner, Eval = Eval_Gen(self.codec),
                weights = {})

    def fit(self, xs, ys, 
            dev_x = None, dev_y = None, 
            **args):
        y_seq = [self.codec.encode(y) for y in ys]

        self.decoder.tag_size = len(self.codec.indexer.list)
        self.learner.tag_size = len(self.codec.indexer.list)

        dev_y_seq = [self.codec.encode(y) for y in dev_y] if dev_y else None

        super(Base_SegTag, self).fit(xs, y_seq, dev_x = dev_x, dev_y = dev_y_seq, **args)

    def predict(self, xs, **args):
        yys = super(Base_Segger, self).predict(xs, **args)
        return Base_Segger.decode(xs, yys)
