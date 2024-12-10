import numpy as np
from tqdm import tqdm

class LinearSVM:
    def __init__(self, learning_rate=0.01, n_epochs=100):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in tqdm(range(self.n_epochs), desc='Training Progress', ncols=100):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if not condition:
                    # 只在不满足约束条件时更新参数
                    self.w += self.lr * (x_i * y[idx])
                    self.b += self.lr * y[idx]
    
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)
    

class KernelSVM:
    pass # TODO


class NormSVM:
    pass # TODO