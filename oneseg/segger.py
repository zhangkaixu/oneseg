import numpy as np
from collections import Counter
import time
import sys

from oneseg.online_model import Online
from oneseg.online_learner import Learner
from oneseg.sequence_labeling import Decoder, Feature_Generator

def load_seg_file(filename):
    """
    """
    xs = []
    ys = []
    for line in open(filename):
        y = line.split()
        ys.append(y)
        xs.append(''.join(y))
    return xs, ys


"""评价"""
class CWS_Evaluator : # 评价
    def __init__(self):
        self.std,self.rst,self.cor=0,0,0
        self.start_time=time.time()
    def _gen_set(self,words):
        offset=0
        word_set=set()
        for word in words:
            word_set.add((offset,word))
            offset+=len(word)
        return word_set
    def __call__(self,std,rst): # 根据答案std和结果rst进行统计
        std,rst=self._gen_set(std),self._gen_set(rst)
        self.std+=len(std)
        self.rst+=len(rst)
        self.cor+=len(std&rst)
    def report(self):
        precision=self.cor/self.rst if self.rst else 0
        recall=self.cor/self.std if self.std else 0
        f1=2*precision*recall/(precision+recall) if precision+recall!=0 else 0
        print("历时: %.2f秒 答案词数: %i 结果词数: %i 正确词数: %i F值: %.4f"
                %(time.time()-self.start_time,self.std,self.rst,self.cor,f1))
    def eval_all(self,test_x, test_y):
        for x, y in zip(test_x, test_y):
            self(x,y)


'''输出高频bigram'''
def count_bigrams(train_x, max_size = 10000):
    counter = Counter()
    min_freq = 1
    for x in (train_x) :
        line = '##'+''.join(x)+'#'
        for i in range(len(line)-1) :
            b = line[i:i+2]
            if b not in counter :
                while max_size and len(counter) > max_size :
                    counter = Counter({k:v for k,v in counter.items() if v > min_freq})
                    min_freq += 1
            counter.update({b:1})
    counter = Counter({k:v for k,v in counter.items() if v >= min_freq})
    return counter



class Tag_Evaluator(CWS_Evaluator) : # 评价
    def __call__(self,std,rst): # 根据答案std和结果rst进行统计
        std = Base_Segger.decode(['0'*len(std)],[std])[0]
        rst = Base_Segger.decode(['0'*len(rst)],[rst])[0]
        super(Tag_Evaluator,self).__call__(std,rst)

class Base_Segger(Online) :
    @staticmethod
    def encode(words_list):
        ys = []
        for words in words_list :
            y=[]
            for word in words :
                if len(word)==1 : y.append(3)
                else : y.extend([0]+[1]*(len(word)-2)+[2])
            ys.append(y)
        return ys

    @staticmethod
    def decode(xs, ys):
        words_list = []
        for x, y in zip(xs, ys):
            cache=''
            words=[]
            for i in range(len(x)) :
                cache+=x[i]
                if y[i]==2 or y[i]==3 :
                    words.append(cache)
                    cache=''
            if cache : words.append(cache)
            words_list.append(words)
        return words_list

    def __init__(self, bigrams = None):
        tag_size = 4 # for cws
        feature_generator = Feature_Generator(bigrams = bigrams)
        decoder = Decoder(feature_generator, tag_size = tag_size)
        learner = Learner(feature_generator, tag_size = tag_size)
        super(Base_Segger, self).__init__(decoder, learner = learner, Eval = Tag_Evaluator)

    def fit(self, xs, ys, dev_x, dev_y, **args):
        y_seq = Base_Segger.encode(ys)
        dev_y_seq = Base_Segger.encode(dev_y)
        super(Base_Segger, self).fit(xs, y_seq, dev_x = dev_x, dev_y = dev_y_seq, **args)

    def predict(self, xs, **args):
        yys = super(Base_Segger, self).predict(xs, **args)
        return Base_Segger.decode(xs, yys)
