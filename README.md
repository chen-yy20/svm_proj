# 最优化方法 大作业

## 下载到本地
已有此文件则跳过。
```
git clone https://github.com/chen-yy20/svm_proj.git
cd svm_proj
```

## 安装依赖
```
pip install -r requirements.txt
```

## 目录结构
`data_loader.py` 加载数据

`svm.py` 支持向量机实现，包括初始的LinearSVM，加入核函数的xKernelSVM以及正则化的NormSVM

`visualizer.py` 数据可视化

`run_linear_svm.py` 运行实验，加载并处理数据、初始化并训练LinearSVM、验证并可视化结果。
`run_kernel_svm.py` 运行实验，加载并处理数据、初始化并训练xKernelSVM、验证并可视化结果。
`run_norm_svm.py` 运行实验，加载并处理数据、初始化并训练NormSVM、验证并可视化结果。

## 运行
修改`run_linear_svm.py`中的最后一行，选择需要的参数。
```
if __name__ == "__main__":
    trained_svm = train_and_evaluate_linear_svm(
        a=4,               # 第一个数字
        b=9,               # 第二个数字
        lr=0.0001,         # 学习率
        epochs=100,        # 训练轮数
        visualize=True     # 是否可视化
    )
```

然后运行：
```
python run_linear_svm.py
```
---
修改`run_kernel_svm.py`中的最后一行，选择需要的参数。
```

if __name__ == "__main__":
    trained_svm = train_and_evaluate_kernel_svm(
        a=4,               # 第一个数字
        b=9,               # 第二个数字
        lr=0.0001,         # 学习率
        epochs=1,          # 训练轮数，核函数特别慢，1个epoch跑跑得了
        kernel='linear',   # 核函数类型：'linear', 'rbf', 'poly'
        gamma=0.1,         # RBF核参数
        visualize=True     # 是否可视化
    )
```

然后运行：
```
python run_kernel_svm.py
```
---

修改`run_norm_svm.py`中的最后一行，选择需要的参数。
```

if __name__ == "__main__":
    trained_svm = train_and_evaluate_norm_svm(
        a=4,               # 第一个数字
        b=9,               # 第二个数字
        lr=0.0001,         # 学习率
        epochs=100,        # 训练轮数
        C=1.0,            # 正则化参数
        visualize=True     # 是否可视化
    )
```

然后运行：
```
python run_norm_svm.py
```