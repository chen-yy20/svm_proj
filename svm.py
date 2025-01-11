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
    

class xKernelSVM:
    def __init__(self, kernel='rbf', learning_rate=0.01, n_epochs=100, gamma=1.0):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = None
        self.b = None
        self.X = None
        self.y = None
    
    def _kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.sum((x1 - x2) ** 2))
        elif self.kernel == 'poly':
            return (np.dot(x1, x2) + 1) ** 2
        else:
            raise ValueError("Kernel not supported")
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        self.X = X
        self.y = y
        self.alpha = np.zeros(n_samples)
        self.b = 0
        
        for _ in tqdm(range(self.n_epochs), desc='Training Progress', ncols=100):
            for i in range(n_samples):
                kernel_sum = 0
                for j in range(n_samples):
                    kernel_sum += self.alpha[j] * y[j] * self._kernel_function(X[i], X[j])
                
                condition = y[i] * (kernel_sum + self.b) >= 1
                if not condition:
                    self.alpha[i] += self.lr
                    self.b += self.lr * y[i]
    
    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            prediction = 0
            for i in range(len(self.X)):
                prediction += self.alpha[i] * self.y[i] * self._kernel_function(self.X[i], x)
            prediction += self.b
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)


class NormSVM:
    def __init__(self, learning_rate=0.01, n_epochs=100, C=1.0):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.C = C  # 正则化参数
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
                    # 更新参数时加入正则化项
                    self.w = self.w * (1 - self.lr) + self.lr * self.C * (x_i * y[idx])
                    self.b += self.lr * y[idx]
                else:
                    # 即使满足条件也要更新w以施加正则化
                    self.w = self.w * (1 - self.lr)
    
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)


class KernelSVM:
    def __init__(self, kernel='rbf', learning_rate=0.01, n_epochs=100, gamma=1.0):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = None
        self.b = None
        self.X = None
        self.y = None
    
    def _kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.sum((x1 - x2) ** 2))
        elif self.kernel == 'poly':
            return (np.dot(x1, x2) + 1) ** 2
        else:
            raise ValueError("Kernel not supported")
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        self.X = X
        self.y = y
        self.alpha = np.zeros(n_samples)
        self.b = 0
        
        total_iterations = self.n_epochs * n_samples
        progress_bar = tqdm(total=total_iterations, desc='Training Progress')
        
        for epoch in range(self.n_epochs):
            for i in range(n_samples):
                kernel_sum = 0
                # 内部循环的计算
                for j in range(n_samples):
                    kernel_sum += self.alpha[j] * y[j] * self._kernel_function(X[i], X[j])
                
                condition = y[i] * (kernel_sum + self.b) >= 1
                if not condition:
                    self.alpha[i] += self.lr
                    self.b += self.lr * y[i]
                
                # 更新进度条
                progress_bar.update(1)
                # 显示当前epoch和样本信息
                progress_bar.set_description(f'Epoch {epoch+1}/{self.n_epochs}, Sample {i+1}/{n_samples}')
        
        progress_bar.close()
    
    def predict(self, X_test):
        predictions = []
        # 添加预测进度条
        for x in tqdm(X_test, desc='Predicting'):
            prediction = 0
            for i in range(len(self.X)):
                prediction += self.alpha[i] * self.y[i] * self._kernel_function(self.X[i], x)
            prediction += self.b
            predictions.append(np.sign(prediction))
        return np.array(predictions)