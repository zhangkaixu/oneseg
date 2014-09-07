#!/usr/bin/env python2
#coding=utf8

import sys
import json

def seq_to_anno(y):
    anno = []
    for word in y :
        anno+=(['-']*(len(word)-1)+['|'])
    return ''.join(anno[:-1])

if __name__ == '__main__':
    for line in sys.stdin :
        line = line.strip().replace(' ','　')
        #line = line.split()
        #predicted = seq_to_anno(line)
        #line = ''.join(line)
        anno = ['?' for i in range(len(line)-1)]
        record = {'raw':line,'anno':anno, 
                #'predicted' : predicted
                }
        print(json.dumps(record, ensure_ascii = False))

