#!/usr/bin/python3
import random
import numpy as np

from oneseg.segger import *
from oneseg.pipeline import *
from oneseg.online_learner import average_weights
"""
re-average method for regularization
"""

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

    weights_list = []

    for i in range(5) :
        segger = Base_Segger(bigrams = bigrams)

        # shuffle
        seed = random.random()
        random.seed(seed)
        random.shuffle(train_x)
        random.seed(seed)
        random.shuffle(train_y)

        # train the model
        segger.fit(train_x, train_y, 
                dev_x = test_x, dev_y = test_y,
                iterations = 5)

        print('save')
        weights_list.append(segger.weights)

    averaged = average_weights(weights_list)

    segger.weights, averged = averaged, segger.weights
    evaluator = CWS_Evaluator()
    output = segger.predict(test_x)
    evaluator.eval_all(test_y, output)
    evaluator.report()
    segger.weights, averged = averaged, segger.weights


    averaged = average_weights(weights_list, only_non_zeros = True)

    segger.weights, averged = averaged, segger.weights
    evaluator = CWS_Evaluator()
    output = segger.predict(test_x)
    evaluator.eval_all(test_y, output)
    evaluator.report()
    segger.weights, averged = averaged, segger.weights
