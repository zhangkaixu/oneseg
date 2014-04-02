#!/usr/bin/python3
import time
import random
import os
import sys
"""
http://www.infzm.com/content/99296
"""
if __name__ == '__main__':
    a = 90000
    b = 99000

    for i in range(a, b):
        filename = os.path.join('data',str(i))
        if os.path.exists(filename): continue
        print(i)
        os.system('curl http://www.infzm.com/content/'+str(i)+' -o '+filename)
        time.sleep(random.randint(4,7))
    pass
