import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_decision_boundary(a, b, X, y, svm, title="Decision Boundary Visualization"):
    """
    将高维数据降到2维并可视化决策边界
    """
    # 使用PCA降维到2维
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    # 创建网格点
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
    
    # 对网格点进行预测
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_original = pca.inverse_transform(grid_points)
    Z = svm.predict(grid_points_original)
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界和数据点
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], c='red', marker='o', label=f'Class {a}')
    plt.scatter(X_2d[y == -1, 0], X_2d[y == -1, 1], c='blue', marker='s', label=f'Class {b}')
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(title)
    plt.legend()
    plt.show()


def visualize_predictions(a, b, X_test, y_test, svm, num_samples=10):
    """可视化预测结果"""
    # 随机选择样本
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for idx, i in enumerate(indices):
        img = X_test[i].reshape(28, 28)
        true_label = str(a) if y_test[i] == 1 else str(b)
        pred_label = str(a) if svm.predict(X_test[i].reshape(1, -1)) == 1 else str(b)
        
        axes[idx].imshow(img, cmap='gray')
        axes[idx].axis('off')
        axes[idx].set_title(f'Truth: {true_label}\nPredict: {pred_label}')
    
    plt.tight_layout()
    plt.show()