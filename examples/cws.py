#!/usr/bin/python3
from oneseg.segger import *
from oneseg.pipeline import *

import pickle
import gzip

if __name__ == '__main__':
    # some refs
    train_file = 'ctb5.training.seg'
    dev_file = 'ctb5.dev.seg'
    test_file = 'ctb5.test.seg'

    # load copora
    train_x, train_y = load_seg_file(train_file)
    dev_x, dev_y = load_seg_file(dev_file)
    test_x, test_y = load_seg_file(test_file)

    #train_x, train_y = dev_x, dev_y # for debug

    # init the model
    bigrams = count_bigrams(train_x, max_size = 100000)
    print('bigram size',len(bigrams))
    #bigrams = None
    segger = Base_Segger(bigrams = bigrams)


    # train the model
    segger.fit(train_x, train_y, 
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
