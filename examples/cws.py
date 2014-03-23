#!/usr/bin/python3
from oneseg import *

if __name__ == '__main__':
    # corpus
    train_file = 'ctb5.training.seg'
    dev_file = 'ctb5.dev.seg'
    test_file = 'ctb5.test.seg'
    train_x, train_y = gen_samples_from_file(dev_file)
    dev_x, dev_y = gen_samples_from_file(test_file)

    # subset of the bigrams
    bigrams = count_bigrams(train_x, max_size = 50000)
    print('bigram feature size',len(bigrams))

    segger = Base_Segger(bigrams = bigrams)
    segger.fit(train_x, train_y, 
            iterations = 5,
            dev_x = dev_x, dev_y = dev_y)

    gzip.open('model.gz','w').write(pickle.dumps(segger))

    segger = pickle.load(gzip.open('model.gz'))
    evaluator = Tag_Evaluator()
    y = segger.predict(dev_x)
    evaluator.eval_all(dev_y, y)
