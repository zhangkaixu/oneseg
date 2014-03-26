#!/usr/bin/python3
from oneseg.segger import *
from oneseg.pipeline import *

import pickle
import gzip
import numpy as np

"""
在特征中引入bigram vector，可提高效果
"""

def load_vector(filename):
    bigrams = {}
    for line in open(filename):
        bigram, *vec = line.split()
        vec = np.array(list(map(float,vec)))
        bigrams[bigram] = vec
    return bigrams
    

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
    bigram_vectors = load_vector("bigrams.vec")
    print('bigram_vector size',len(bigram_vectors))

    #bigrams = None
    segger = Base_Segger(bigrams = bigrams,
            bigram_vectors = bigram_vectors,
            )


    # train the model
    segger.fit(train_x, train_y, 
            dev_x = test_x, dev_y = test_y,
            iterations = 10)

    # save it and reload it
    gzip.open('model.gz','w').write(pickle.dumps(segger))
    segger = pickle.load(gzip.open('model.gz'))

    # use the model and evaluate outside
    evaluator = CWS_Evaluator()
    output = segger.predict(test_x)
    evaluator.eval_all(test_y, output)
