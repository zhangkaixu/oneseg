#!/usr/bin/python3

from oneseg.segger import *
from oneseg.utils import *

import pickle
import gzip
import json
import sys
import numpy as np

def load_vector(filename):
    bigrams = {}
    for line in open(filename):
        bigram, *vec = line.split()
        vec = np.array(list(map(float,vec)))
        bigrams[bigram] = vec
    return bigrams

def seq_to_anno(y):
    anno = []
    for word in y :
        anno+=(['-']*(len(word)-1)+['|'])
    return ''.join(anno[:-1])

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


TAG_B = Codec.TAG_B
TAG_M = Codec.TAG_M
TAG_E = Codec.TAG_E
TAG_S = Codec.TAG_S
tag_map = {
        ('?','?'): [TAG_B, TAG_M, TAG_E, TAG_S],
        ('-','?'): [TAG_M, TAG_E],
        ('?','-'): [TAG_B, TAG_M],
        ('?','|'): [TAG_E, TAG_S],
        ('|','?'): [TAG_B, TAG_S],
        ('|','-'): [TAG_B],
        ('-','-'): [TAG_M],
        ('-','|'): [TAG_E],
        ('|','|'): [TAG_S],
        }

if __name__ == '__main__':
    if len(sys.argv)<2 :
        exit()

    #filename = 'dev.anno.json'
    filename = sys.argv[1]
    
    anno_x, anno_y = [], []
    corpus = []
    total_x = []
    partial_x, partial_Y = [], []
    for line in open(filename):
        rec = json.loads(line)
        raw = rec['raw']
        anno = rec['anno']
        if '?' not in anno :
            y = []
            cache = []
            for ch, tag in zip(raw, anno):
                cache.append(ch)
                if tag == '|' :
                    y.append(''.join(cache))
                    cache = []
            cache.append(ch)
            y.append(''.join(cache))
            assert(len(raw)==sum(len(x) for x in y))
            anno_x.append(raw)
            anno_y.append(y)
        if not all(x == '?' for x in anno):
            #if '?' in anno : continue
            partial_x.append(raw)
            tags = '|' + ''.join(anno) + '|'
            Y = [np.array([-np.inf, - np.inf, - np.inf, - np.inf]) for i in range(len(raw))]
            for i in range(len(Y)):
                for ind in tag_map[(tags[i],tags[i+1])] : Y[i][ind] = 0
            partial_Y.append(Y)
                


        total_x.append(raw)
        corpus.append(rec)

    co_shuffle(partial_x, partial_Y)
    print('语料库总句子数目',len(corpus),'完整标注',len(anno_x), '部分或完整标注',len(partial_x))
    print('完成度 %.3f%%'%(100*len(anno_x)/len(corpus)))
    print('部分完成度 %.3f%%'%(100*len(partial_x)/len(corpus)))
    #input()


    # some refs
    
    test_file = '100.seg'
    test_x, test_y = load_seg_file(test_file)


    # init the model
    #bigrams = count_bigrams(train_x, max_size = 100000)
    #print('bigram size',len(bigrams))
    bigrams = None
    bigram_vectors = load_vector("bigrams.vec")
    print('bigram_vector size',len(bigram_vectors))
    segger = Base_Segger(bigrams = bigrams,
            bigram_vectors = bigram_vectors,
            )

    # train the model
    #segger.fit(anno_x, anno_y, 
            #dev_x = test_x, dev_y = test_y,
            #iterations = 5)
    segger.fit(partial_x, partial_x, 
            train_Y = partial_Y, subset = subset,
            dev_x = test_x, dev_y = test_y,
            iterations = 10)

    f = open(filename+'.output','w')
    for rec, raw in show_progress(zip(corpus, total_x)):
        y, margins = segger.predict_with_margin([raw], as_sequence = True)
        #print(y[0])
        #print(seq_to_anno(y[0]))
        rec['margins'] = margins[0]
        rec['margin'] = min(margins[0])
        rec['predicted'] = seq_to_anno(y[0])

    corpus = sorted(corpus, key = lambda x : x['margin'])
    for rec in corpus :
        j = json.dumps(rec, ensure_ascii = False)
        print(j,file =f)

