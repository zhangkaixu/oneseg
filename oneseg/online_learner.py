import numpy as np
"""参数更新"""
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
            vec = np.zeros(self.tag_size)
            vec[y[i]] += 1
            vec[z[i]] -= 1
            for key in features[i]:
                x = grad.setdefault(key, np.zeros(self.tag_size))
                x += vec

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


    def predict(self, test_x):
        result_y = []
        for x in test_x :
            y = self.decoder(x, self.weights)
            result_y.append(y)
        return result_y
