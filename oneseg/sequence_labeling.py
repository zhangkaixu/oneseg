import numpy as np
class Decoder :
    """解码算法"""
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

        alphas=[[[e,None] for e in emissions[0]]] # [分数, 上一状态]
        for i in range(len(emissions)-1) :
            alphas.append([max([alphas[i][j][0]+transitions[j][k]+emissions[i+1][k], j]
                                        for j in range(self.tag_size)) # 枚举前一状态
                                        for k in range(self.tag_size)]) # 枚举后一状态
        # 根据alphas中的“指针”得到最优序列
        alpha=max([alphas[-1][j],j] for j in range(self.tag_size))
        i=len(emissions)
        tags=[]
        while i :
            tags.append(alpha[1])
            i-=1
            alpha=alphas[i][alpha[1]]
        return list(reversed(tags))

    def cal_margins(self,sentence, weights):
        emissions,transitions = self.put_value(sentence, weights)

        alphas=[[[e,None] for e in emissions[0]]] # [分数, 上衣状态]
        for i in range(len(emissions)-1) :
            alphas.append([max([alphas[i][j][0]+transitions[j][k]+emissions[i+1][k], j]
                                        for j in range(self.tag_size)) # 枚举前一状态
                                        for k in range(self.tag_size)]) # 枚举后一状态

        max_score = max([alphas[-1][j],j] for j in range(self.tag_size))[0][0]

        betas = [[[e, None] for e in emissions[-1]]]
        for i in range(len(emissions)-2,-1,-1):
            betas.append([max([betas[-1][j][0]+transitions[k][j]+emissions[i][k], j]
                                        for j in range(self.tag_size)) # 枚举右边状态
                                        for k in range(self.tag_size)]) # 枚举左边状态
        betas = list(reversed(betas))
        margins = [ 
                [max_score - alphas[i][j][0] - betas[i][j][0] + emissions[i][j] for j in range(self.tag_size)] 
            for i in range(len(betas))]
        return margins


class Feature_Generator :
    '''根据模板生成特征'''
    def __init__(self, bigrams = None):
        self.bigrams = bigrams # 可用bigrams来限定能够产生的bigram
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
