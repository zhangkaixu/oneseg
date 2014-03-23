import numpy as np
from collections import Counter
import time
import sys
import pickle
import gzip

"""数据的读入与读出"""
def load_example(words): # 词数组，得到x，y
    y=[]
    for word in words :
        if len(word)==1 : y.append(3)
        else : y.extend([0]+[1]*(len(word)-2)+[2])
    return ''.join(words),y

def dump_example(x,y) : # 根据x，y得到词数组
    cache=''
    words=[]
    for i in range(len(x)) :
        cache+=x[i]
        if y[i]==2 or y[i]==3 :
            words.append(cache)
            cache=''
    if cache : words.append(cache)
    return words

def gen_samples_from_file(filename):
    xs = []
    ys = []
    for i,line in enumerate(open(filename)):
        if i%100 == 0 :
            print(i,end='\r',file = sys.stderr)
        x, y = load_example(line.split())
        xs.append(x)
        ys.append(y)
    return xs, ys

"""评价"""
class Tag_Evaluator : # 评价
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
        std = dump_example('0'*len(std),std)
        rst = dump_example('0'*len(rst),rst)
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
        self.report()

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

"""解码算法"""
class Decoder :
    def __init__(self, feature_generator, tag_size = 4):
        self.feature_generator = feature_generator
        self.tag_size = tag_size

    def put_value(self, sentence, weights): # 根据输入与参数， 得到'发射概率'与'转移概率'
        values = [np.zeros(self.tag_size) for i in range(len(sentence))]
        features = self.feature_generator(sentence)
        for i in range(len(values)) :
            values[i] += sum(weights.get(key,0) for key in features[i])
        values = [x.tolist() for x in values]
        return values, weights.get('trans',np.zeros((self.tag_size, self.tag_size))).tolist()

    def __call__(self, sentence, weights):
        emissions,transitions = self.put_value(sentence, weights)

        alphas=[[[e,None] for e in emissions[0]]]
        for i in range(len(emissions)-1) :
            alphas.append([max([alphas[i][j][0]+transitions[j][k]+emissions[i+1][k],j]
                                        for j in range(self.tag_size))
                                        for k in range(self.tag_size)])
        # 根据alphas中的“指针”得到最优序列
        alpha=max([alphas[-1][j],j] for j in range(self.tag_size))
        i=len(emissions)
        tags=[]
        while i :
            tags.append(alpha[1])
            i-=1
            alpha=alphas[i][alpha[1]]
        return list(reversed(tags))

'''根据模板生成特征'''
class Feature_Generator :
    def __init__(self, bigrams = None):
        self.bigrams = bigrams
    def __call__(self,sentence):
        ext = '##' + sentence + '##'
        bigrams = [ext[i:i+2] for i in range(len(ext)-1)]
        if self.bigrams :
            bigrams = [(b if b in self.bigrams else '~~') for b in bigrams]
        features = []
        for i in range(len(sentence)) :
            features.append([ext[i+1]+'r', ext[i+2]+'M', ext[i+3]+'l',
                bigrams[i]+'R', bigrams[i+1]+'r', bigrams[i+2]+'l', bigrams[i+3] + 'L'])
        return features


"""参数更新"""
class Learner :
    def reset(self):
        self.step = 0
        self.acc = {}

    def __init__(self, feature_generator, tag_size = 4):
        self.feature_generator = feature_generator
        self.tag_size = tag_size
        self.reset()

    def get_grad(self, sentence, y, z):
        grad = dict()
        features = self.feature_generator(sentence)
        for i in range(len(sentence)) :
            vec = np.zeros(self.tag_size)
            vec[y[i]] += 1
            vec[z[i]] -= 1
            for key in features[i]:
                x = grad.setdefault(key, np.zeros(self.tag_size))
                x += vec

        trans = np.zeros([self.tag_size, self.tag_size])
        for i in range(len(y)-1):
            trans[y[i]][y[i+1]] += 1
            trans[z[i]][z[i+1]] -= 1
        grad['trans'] = trans
        return grad

    def update(self, weights, grad):
        for k,v in grad.items():
            if k not in weights :
                weights[k] = v
            else :
                weights[k] += v
            if k not in self.acc :
                self.acc[k] = v * self.step
            else :
                self.acc[k] += v * self.step

    def __call__(self, sentence, y, z, weights):
        self.step += 1
        if y == z : return
        grad = self.get_grad(sentence, y, z) # 求梯度
        self.update(weights, grad) # 根据梯度更新权重

    def average(self, weights):
        averaged = {}
        for k, v in weights.items():
            averaged[k] = v - self.acc[k] / self.step
        return averaged

class Online :
    def __init__(self, decoder, weights = {}, learner = None, Eval = None):
        self.decoder = decoder
        self.learner = learner
        self.Eval = Eval
        self.weights = weights

    def fit(self, train_x, train_y, 
            dev_x = None, dev_y = None,
            iterations = 15):
        self.learner.reset()
        for it in range(iterations) :
            if self.Eval : evaluator = self.Eval()
            for x, y in zip(train_x, train_y) :
                z = self.decoder(x, self.weights)
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

class Base_Segger(Online) :
    def __init__(self, bigrams = None):
        tag_size = 4 # for cws
        feature_generator = Feature_Generator(bigrams = bigrams)
        decoder = Decoder(feature_generator, tag_size = tag_size)
        learner = Learner(feature_generator, tag_size = tag_size)
        super(Base_Segger, self).__init__(decoder, learner = learner, Eval = Tag_Evaluator)
