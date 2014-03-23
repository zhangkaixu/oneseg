#!/usr/bin/python3
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

    for line in sys.stdin :
        segged = segger.predict([line.strip()])
        print(*segged[0])
