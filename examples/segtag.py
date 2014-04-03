#!/usr/bin/python3
from oneseg.segger import *
from oneseg.segtag import load_std_examples, Codec, Base_SegTag

def load_vector(filename):
    bigrams = {}
    for line in open(filename):
        bigram, *vec = line.split()
        vec = np.array(list(map(float,vec)))
        bigrams[bigram] = vec
    return bigrams

if __name__ == '__main__':
    test_x, test_y = load_std_examples('ctb5.test.tag')
    dev_x, dev_y = load_std_examples('ctb5.dev.tag')
    #train_x, train_y = load_std_examples('ctb5.training.tag')

    bigrams = None
    #bigrams = count_bigrams(train_x, max_size = 50000)
    #print('bigram size',len(bigrams))
    bigram_vectors = None
    #bigram_vectors = load_vector("bigrams.vec")
    #print('bigram_vector size',len(bigram_vectors))

    segtag = Base_SegTag(bigrams = bigrams,
            bigram_vectors = bigram_vectors,
            )



    segtag.fit(dev_x, dev_y, dev_x = test_x, dev_y = test_y)
    #segtag.fit(train_x, train_y, dev_x = test_x, dev_y = test_y, iterations = 10)
