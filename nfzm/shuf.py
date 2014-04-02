#!/usr/bin/python3
import random
import sys

if __name__ == '__main__':
    lines = list(sys.stdin)
    random.seed(100)
    random.shuffle(lines)
    print(*lines[:100],sep='',end= '')
    
