#!/usr/bin/python3
import sys
import json


def to_seg(raw, anno):
    sentence = []
    cache = []
    for c, a in zip(raw, anno):
        cache.append(c)
        if a == '|' :
            sentence.append(''.join(cache))
            cache = []
    if cache : sentence.append(''.join(cache))
    return sentence


if __name__ == '__main__':
    with_id = False
    if len(sys.argv)>1 :
        if sys.argv[1] == '+id' :
            with_id = True

    for line in sys.stdin :
        rec = json.loads(line)
        raw = rec['raw'].replace(' ','　')
        anno = rec['anno']
        id = rec['id']
        sentence = to_seg(raw,anno)
        print(*sentence)

        pass

