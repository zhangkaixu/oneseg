import os
import sys
import gzip
import pickle

from unittest import TestCase
from oneseg.segger import *
from nose.tools import assert_almost_equals



class TestCWS(TestCase):
    def setUp(self):
        # some refs
        train_file = 'ctb5.training.seg'
        dev_file = 'ctb5.dev.seg'
        test_file = 'ctb5.test.seg'

        print('check files: dev.seg train.seg test.seg ...')
        # load copora
        self.train_x, self.train_y = load_seg_file(train_file)
        self.dev_x, self.dev_y = load_seg_file(dev_file)
        self.test_x, self.test_y = load_seg_file(test_file)

    def tearDown(self):
        os.system('\\rm test_model.gz')


    def test_分词(self):
        self.bigrams = count_bigrams(self.dev_x, max_size = 100000)
        print('bigram size',len(self.bigrams))
        self.assertEqual(len(self.bigrams), 6308)

        train_x, train_y = self.dev_x, self.dev_y # for debug
        test_x, test_y = self.test_x, self.test_y

        # init the model
        segger = Base_Segger(bigrams = self.bigrams)

        # train the model
        segger.fit(train_x, train_y, 
                dev_x = test_x, dev_y = test_y,
                iterations = 3)

        f1 = segger.evaluator.report(quiet = True)
        assert_almost_equals(f1, 0.8299, places = 4)

        # save it and reload it
        gzip.open('test_model.gz','w').write(pickle.dumps(segger))
        segger = pickle.load(gzip.open('test_model.gz'))

        # use the model and evaluate outside
        evaluator = CWS_Evaluator()
        output = segger.predict(test_x)
        evaluator.eval_all(test_y, output)
        evaluator.report()

        f1 = segger.evaluator.report(quiet = True)
        assert_almost_equals(f1, 0.8299, places = 4)
