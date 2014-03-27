import numpy as np


def average_weights(weights_list, only_non_zeros = False):
    """
    将多个weights平均成一个，通常会有更好的效果
    only_non_zeros : 只平均非零权重，一般来说有更好的效果
    """
    # 加和
    averaged = {}
    if only_non_zeros : N = {}
    for weights in weights_list :
        for k, v in weights.items() :
            if k in averaged : averaged[k] += v
            else : averaged[k] = v
            if only_non_zeros :
                if k in N : N[k] += np.abs(np.sign(v))
                else : N[k] = np.abs(np.sign(v))
    # 相除
    n = len(weights_list)
    for k, v in averaged.items():
        if not only_non_zeros : 
            v /= n
        else :
            mask = np.where(N[k], N[k], N[k]+1)
            averaged[k] = np.where(N[k], averaged[k]/mask, averaged[k])
    return averaged


class Learner :
    def reset(self):
        self.step = 0
        self.acc = {}

    def __init__(self, feature_generator, tag_size = 4):
        self.feature_generator = feature_generator
        self.tag_size = tag_size
        self.reset()

    def get_grad(self, sentence, y, z):
        grad = dict()
        features = self.feature_generator(sentence)
        for i in range(len(sentence)) :
            if y[i] == z[i] : continue
            vec = np.zeros(self.tag_size)
            vec[y[i]] += 1
            vec[z[i]] -= 1
            for key in features[i]:
                if type(key) is not list :
                    x = grad.setdefault(key, np.zeros(self.tag_size))
                    x += vec
                else :
                    weight_key, feature_vector = key
                    if feature_vector is not None :
                        x = grad.setdefault(weight_key, np.zeros((self.tag_size, feature_vector.shape[0])))
                        x += vec[:,np.newaxis] * feature_vector / 100
                    pass

        trans = np.zeros([self.tag_size, self.tag_size])
        for i in range(len(y)-1):
            trans[y[i]][y[i+1]] += 1
            trans[z[i]][z[i+1]] -= 1
        grad['trans'] = trans
        return grad

    def update(self, weights, grad):
        for k,v in grad.items():
            if k not in weights :
                weights[k] = v
            else :
                weights[k] += v
            if k not in self.acc :
                self.acc[k] = v * self.step
            else :
                self.acc[k] += v * self.step

    def __call__(self, sentence, y, z, weights):
        self.step += 1
        if y == z : return
        grad = self.get_grad(sentence, y, z) # 求梯度
        self.update(weights, grad) # 根据梯度更新权重

    def average(self, weights):
        averaged = {}
        for k, v in weights.items():
            averaged[k] = v - self.acc[k] / self.step
        return averaged
