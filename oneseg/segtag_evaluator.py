import time

class SegTag_Evaluator :
    def reset(self):
        self.std,self.rst,self.cor=0,0,0
        self.seg_std,self.seg_rst,self.seg_cor=0,0,0
        self.start_time=time.time()

    def __init__(self, codec = None):
        self.codec = codec
        self.reset()

    def _gen_set(self, words):
        offset=0
        word_set=set()
        for word in words:
            word_set.add((offset,word))
            if type(word) == tuple :
                offset+=len(word[0])
            else :
                offset+=len(word)
        return word_set
        
    def __call__(self, y, z, use_decoder = True) :
        if use_decoder and self.codec is not None :
            x = '.' * len(y)
            y = self.codec.decode(x, y)
            z = self.codec.decode(x, z)

        std,rst=self._gen_set(y),self._gen_set(z)
        self.std+=len(std)
        self.rst+=len(rst)
        self.cor+=len(std&rst)

        std = self._gen_set([yy[0] for yy in y])
        rst = self._gen_set([zz[0]for zz in z])
        self.seg_std+=len(std)
        self.seg_rst+=len(rst)
        self.seg_cor+=len(std&rst)

    def report(self, quiet = False):
        precision=1.0 * self.cor/self.rst if self.rst else 0
        recall=1.0 * self.cor/self.std if self.std else 0
        #print(precision,recall)
        f1=2*precision*recall/(precision+recall) if precision+recall!=0 else 0
        precision=self.seg_cor/self.seg_rst if self.seg_rst else 0
        recall=self.seg_cor/self.seg_std if self.seg_std else 0
        seg_f1=2*precision*recall/(precision+recall) if precision+recall!=0 else 0
        if not quiet :
            print("time: %.2f std: %i rst: %i cor: %i F: %.4f seg_f: %.4f"
                    %(time.time()-self.start_time,self.std,self.rst,self.cor,f1,seg_f1))
        return f1

    def eval_all(self,test_x, test_y):
        for x, y in zip(test_x, test_y):
            self(x,y, use_decoder = False)
