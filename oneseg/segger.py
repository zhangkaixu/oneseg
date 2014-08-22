from __future__ import print_function
from collections import Counter
from oneseg.character_labeling import Character_Labeler
from oneseg.segtag_evaluator import SegTag_Evaluator

def load_seg_file(filename):
    xs = []
    ys = []
    for line in open(filename):
        #print(line)
        line = line.decode('utf8',"replace") # special for python2
        y = line.split()
        ys.append(y)
        xs.append(''.join(y))
    return xs, ys

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



class Codec :
    TAG_B = 3
    TAG_M = 2
    TAG_E = 1
    TAG_S = 0
    def get_size(self):
        return 4
    def encode(self, words):
        y=[]
        for word in words :
            if len(word)==1 : y.append(Codec.TAG_S)
            else : y.extend([Codec.TAG_B]+[Codec.TAG_M]*(len(word)-2)+[Codec.TAG_E])
        return y
    def decode(self, x, y):
        cache=''
        words=[]
        for i in range(len(x)) :
            cache+=x[i]
            if y[i]==Codec.TAG_E or y[i]==Codec.TAG_S :
                words.append(cache)
                cache=''
        if cache : words.append(cache)
        return words


class Base_Segger(Character_Labeler):
    def __init__(self, **args):
        codec = Codec()
        evaluator = SegTag_Evaluator(codec)
        super(Base_Segger, self).__init__(codec, evaluator, **args)
