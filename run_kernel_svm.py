import numpy as np
from svm import KernelSVM
from data_loader import MNISTDataLoader
from visualizer import visualize_predictions, visualize_decision_boundary

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def train_and_evaluate_kernel_svm(a, b, lr, epochs, kernel, gamma, visualize):
    # 初始化数据加载器
    loader = MNISTDataLoader(digits=[a, b], shuffle=True)
    
    # 获取完整训练集和测试集
    X_train, y_train = loader.get_full_train_data()
    X_test, y_test = loader.get_full_test_data()
    
    # 将标签转换为+1/-1
    y_train = np.where(y_train == a, 1, -1)
    y_test = np.where(y_test == a, 1, -1)
    
    # 初始化SVM
    svm = KernelSVM(learning_rate=lr, n_epochs=epochs, kernel=kernel, gamma=gamma)
    
    # 训练模型
    print(f"Training Kernel SVM with {kernel} kernel...")
    svm.fit(X_train, y_train)
    
    # 计算训练集和测试集准确率
    train_pred = svm.predict(X_train)
    test_pred = svm.predict(X_test)
    
    train_acc = calculate_accuracy(y_train, train_pred)
    test_acc = calculate_accuracy(y_test, test_pred)
    
    print("-" * 40)
    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    
    # 可视化
    if False and visualize: # some bug happened when visualize kernel svm
        visualize_predictions(a, b, X_test, y_test, svm)
        print("正在可视化svm, 需要大约15s...")
        visualize_decision_boundary(a, b, X_test, y_test, svm, "Test Data Decision Boundary")
    
    return svm

if __name__ == "__main__":
    trained_svm = train_and_evaluate_kernel_svm(
        a=4,               # 第一个数字
        b=9,               # 第二个数字
        lr=0.0001,         # 学习率
        epochs=1,        # 训练轮数，核函数特别慢，1个epoch跑跑得了
        kernel='linear',      # 核函数类型：'linear', 'rbf', 'poly'
        gamma=0.1,         # RBF核参数
        visualize=True     # 是否可视化
    )