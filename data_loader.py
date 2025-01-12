import numpy as np

class MNISTDataLoader:
    def __init__(self, batch_size=32, digits=None, shuffle=True):
        """
        初始化数据加载器
        
        参数:
        batch_size: 每批数据的大小
        digits: 需要的数字列表，例如[4,9]
        shuffle: 是否打乱数据
        """
        self.batch_size = batch_size
        self.digits = digits
        self.shuffle = shuffle
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.train_index = 0
        self.test_index = 0
        
        # 加载并处理数据
        self._load_data()
        
    def _load_data(self):
        """加载MNIST数据集并进行预处理"""
        with np.load('mnist.npz') as data:
            x_train = data['x_train'].astype('float32') / 255.0
            y_train = data['y_train']
            x_test = data['x_test'].astype('float32') / 255.0
            y_test = data['y_test']
        
        # 如果指定了digits，只选择这些数字的数据
        if self.digits is not None:
            train_mask = np.isin(y_train, self.digits)
            test_mask = np.isin(y_test, self.digits)
            
            x_train = x_train[train_mask]
            y_train = y_train[train_mask]
            x_test = x_test[test_mask]
            y_test = y_test[test_mask]
            
        self.train_data = x_train.reshape(x_train.shape[0], -1)
        self.test_data = x_test.reshape(x_test.shape[0], -1)
        
        # 存储标签
        self.train_labels = y_train
        self.test_labels = y_test
        
        # 如果需要打乱数据
        if self.shuffle:
            self._shuffle_train_data()
    
    def _shuffle_train_data(self):
        """打乱训练数据"""
        indices = np.random.permutation(len(self.train_data))
        self.train_data = self.train_data[indices]
        self.train_labels = self.train_labels[indices]
    
    def get_train_batch(self):
        """获取一批训练数据"""
        if self.train_index + self.batch_size >= len(self.train_data):
            # 一个epoch结束，重置索引
            self.train_index = 0
            if self.shuffle:
                self._shuffle_train_data()
        
        batch_data = self.train_data[self.train_index:self.train_index + self.batch_size]
        batch_labels = self.train_labels[self.train_index:self.train_index + self.batch_size]
        
        self.train_index += self.batch_size
        return batch_data, batch_labels
    
    def get_test_batch(self):
        """获取一批测试数据"""
        if self.test_index + self.batch_size >= len(self.test_data):
            # 一个epoch结束，重置索引
            self.test_index = 0
        
        batch_data = self.test_data[self.test_index:self.test_index + self.batch_size]
        batch_labels = self.test_labels[self.test_index:self.test_index + self.batch_size]
        
        self.test_index += self.batch_size
        return batch_data, batch_labels
    
    def get_full_train_data(self):
        """获取完整训练集"""
        return self.train_data, self.train_labels
    
    def get_full_test_data(self):
        """获取完整测试集"""
        return self.test_data, self.test_labels
    
    @property
    def train_size(self):
        """返回训练集大小"""
        return len(self.train_data)
    
    @property
    def test_size(self):
        """返回测试集大小"""
        return len(self.test_data)

# 使用示例
def example_usage():
    # 创建数据加载器，只加载数字4和9
    loader = MNISTDataLoader(batch_size=32, digits=[4, 9], shuffle=True)
    
    # 打印数据集信息
    print(f"训练集大小: {loader.train_size}")
    print(f"测试集大小: {loader.test_size}")
    
    # 获取一个批次的训练数据
    batch_x, batch_y = loader.get_train_batch()
    print(f"批次数据形状: {batch_x.shape}")
    print(f"批次标签形状: {batch_y.shape}")
    
    # 获取完整测试集
    test_x, test_y = loader.get_full_test_data()
    print(f"测试集数据形状: {test_x.shape}")
    print(f"测试集标签形状: {test_y.shape}")
    
    # 遍历整个训练集的例子
    n_batches = loader.train_size // loader.batch_size
    for i in range(n_batches):
        batch_x, batch_y = loader.get_train_batch()
        # 在这里处理批次数据
        print(f"处理第 {i+1}/{n_batches} 批数据")

if __name__ == "__main__":
    example_usage()