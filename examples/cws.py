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
    train_x, train_y = load_seg_file(dev_file)
    dev_x, dev_y = load_seg_file(test_file)

    # init the model
    segger = Base_Segger()

    # train the model
    segger.fit(train_x, train_y, 
            dev_x = dev_x, dev_y = dev_y,
            iterations = 5)

    # save it and reload it
    gzip.open('model.gz','w').write(pickle.dumps(segger))
    segger = pickle.load(gzip.open('model.gz'))

    # use the model and evaluate outside
    evaluator = CWS_Evaluator()
    output = segger.predict(dev_x)
    evaluator.eval_all(dev_y,output)
    evaluator.report()
