import sys
import random

class Indexer :
    def __init__(self):
        self.list = []
        self.dict = {}
        pass
    def __call__(self, key):
        if key not in self.dict :
            self.dict[key] = len(self.list)
            self.list.append(key)
        return self.dict[key]

def co_shuffle(*lists):
    seed = random.random()
    for i in range(len(lists)):
        random.seed(seed)
        random.shuffle(lists[i])
    

def show_progress(l, n = None):
    def show_numbers(c, n):
        if c % 100 == 0 : print("%d (%.1f%%)"%(c, c/n*100), end='\r', file = sys.stderr)

    if type(l) == int :
        if n == None : n = l
        for c in range(n):
            show_numbers(c, n)
            yield c
    if type(l) == list :
        if n == None : n = len(l)
        for c in range(n):
            show_numbers(c, n)
            yield l[c]
    if type(l) == zip :
        if n == None : n = 1
        c = 0
        for item in l:
            show_numbers(c, n)
            c += 1
            yield item
