#!/usr/bin/python3
from oneseg.segger import *
from oneseg.pipeline import *

import pickle
import gzip
import random
import numpy as np

def gen_Ys(ys):
    """ 随机将唯一的标准答案污染成一个可行解集合 """
    ys = Base_Segger.encode(ys)
    for i in range(len(ys)) :
        y = ys[i]
        y = [[x] for x in y]
        for j in range(len(y)):
            n = random.randint(0, 3)
            if n != y[j][0] : y[j].append(n)
            v = np.array([-np.inf, - np.inf, - np.inf, - np.inf])
            for ind in y[j]:
                v[ind] = 0
            y[j] = v
        ys[i] = y
    return ys

def subset(y, Y):
    """判断一个结果是否在可行的子集之中"""
    return all(Y[i][y[i]] == 0 for i in range(len(y)))


if __name__ == '__main__':
    # some refs
    train_file = 'ctb5.training.seg'
    dev_file = 'ctb5.dev.seg'
    test_file = 'ctb5.test.seg'

    # load copora
    train_x, train_y = load_seg_file(train_file)
    dev_x, dev_y = load_seg_file(dev_file)
    test_x, test_y = load_seg_file(test_file)

    train_x, train_y = dev_x, dev_y # for debug

    # init the model
    bigrams = count_bigrams(train_x, max_size = 100000)
    print('bigram size',len(bigrams))
    #bigrams = None
    segger = Base_Segger(bigrams = bigrams)

    # train the model
    Ys = gen_Ys(train_y) # 将完全标注语料变为部分标注语料
    train_y = [[] for y in train_y] # 删除完全标注结果
    segger.fit(train_x, train_y, # 使用部分标注语料训练, 给定train_Y, train_y就没有用了
            train_Y = Ys, subset = subset,
            dev_x = test_x, dev_y = test_y,
            iterations = 5)

    # save it and reload it
    gzip.open('model.gz','w').write(pickle.dumps(segger))
    segger = pickle.load(gzip.open('model.gz'))

    # use the model and evaluate outside
    evaluator = CWS_Evaluator()
    output = segger.predict(test_x)
    evaluator.eval_all(test_y, output)
    evaluator.report()
