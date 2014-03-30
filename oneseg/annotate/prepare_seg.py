#!/usr/bin/python3

import sys
import json

if __name__ == '__main__':
    for line in sys.stdin :
        line = line.split()
        line = ''.join(line)
        anno = ['?' for i in range(len(line)-1)]
        record = {'raw':line,'anno':anno}
        print(json.dumps(record, ensure_ascii = False))


