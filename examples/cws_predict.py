#!/usr/bin/python3
from __future__ import print_function
import pickle
import gzip
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2 :
        print("""oneseg Chinese Word Segmentor
    example: %s model.gz"""%(sys.argv[0]))
        exit()

    model_filename = sys.argv[1]
    segger = pickle.load(gzip.open(model_filename))
    print('ok',file = sys.stderr)

    for line in sys.stdin :
        segged = segger.predict([line.strip()])
        print(*segged[0])

        #result, margin = segger.predict_with_margin([line.strip()])
        #print(margin[0])
