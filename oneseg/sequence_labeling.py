import numpy as np
"""解码算法"""
class Decoder :
    def __init__(self, feature_generator, tag_size = 4):
        self.feature_generator = feature_generator
        self.tag_size = tag_size

    def put_value(self, sentence, weights): # 根据输入与参数， 得到'发射概率'与'转移概率'
        values = [np.zeros(self.tag_size) for i in range(len(sentence))]
        features = self.feature_generator(sentence)
        for i in range(len(values)) :
            values[i] += sum(weights.get(key,0) for key in features[i])
        values = [x.tolist() for x in values]
        return values, weights.get('trans',np.zeros((self.tag_size, self.tag_size))).tolist()

    def __call__(self, sentence, weights):
        emissions,transitions = self.put_value(sentence, weights)

        alphas=[[[e,None] for e in emissions[0]]]
        for i in range(len(emissions)-1) :
            alphas.append([max([alphas[i][j][0]+transitions[j][k]+emissions[i+1][k],j]
                                        for j in range(self.tag_size))
                                        for k in range(self.tag_size)])
        # 根据alphas中的“指针”得到最优序列
        alpha=max([alphas[-1][j],j] for j in range(self.tag_size))
        i=len(emissions)
        tags=[]
        while i :
            tags.append(alpha[1])
            i-=1
            alpha=alphas[i][alpha[1]]
        return list(reversed(tags))

'''根据模板生成特征'''
class Feature_Generator :
    def __init__(self, bigrams = None):
        self.bigrams = bigrams
    def __call__(self,sentence):
        ext = '##' + sentence + '##'
        bigrams = [ext[i:i+2] for i in range(len(ext)-1)]
        if self.bigrams :
            bigrams = [(b if b in self.bigrams else '~~') for b in bigrams]
        features = []
        for i in range(len(sentence)) :
            features.append([ext[i+1]+'r', ext[i+2]+'M', ext[i+3]+'l',
                bigrams[i]+'R', bigrams[i+1]+'r', bigrams[i+2]+'l', bigrams[i+3] + 'L'])
        return features
