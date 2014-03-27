#!/usr/bin/python3
from oneseg.segger import *
from oneseg.pipeline import *
from oneseg.utils import co_shuffle

import pickle
import gzip
"""
主动学习
"""

if __name__ == '__main__':
    # some refs
    train_file = 'ctb5.training.seg'
    dev_file = 'ctb5.dev.seg'
    test_file = 'ctb5.test.seg'

    # load copora
    train_x, train_y = load_seg_file(train_file)
    test_x, test_y = load_seg_file(test_file)

    co_shuffle(train_x, train_y)
    first_size = 500
    dev_x = train_x[: first_size]
    dev_y = train_y[: first_size]
    train_x = train_x[first_size:]
    train_y = train_y[first_size::]

    # init the model
    #bigrams = count_bigrams(train_x, max_size = 100000)
    #print('bigram size',len(bigrams))
    bigrams = None
    segger = Base_Segger(bigrams = bigrams)

    # train the model
    segger.fit(dev_x, dev_y, 
            dev_x = test_x, dev_y = test_y,
            iterations = 5)

    # "fake" active learning
    results, margins = segger.predict_with_margin(train_x)
    
    # select new examples
    r = sorted(list(zip(margins, train_x, train_y)))
    for _,x,y in r[:500] :
        dev_x.append(x)
        dev_y.append(y)
    co_shuffle(dev_x, dev_y)

    # re-train the model using additional new examples
    segger = Base_Segger(bigrams = bigrams)
    segger.fit(dev_x, dev_y, 
            dev_x = test_x, dev_y = test_y,
            iterations = 5)
