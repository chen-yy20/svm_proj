# 最优化方法 大作业

## 下载到本地
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

`svm.py` 支持向量机实现，在这里实现你们的改进代码

`visualizer.py` 数据可视化


`exp.py` 运行实验，加载并处理数据、初始化并训练svm、验证并可视化结果。

## 运行
修改exp.py中的最后一行，选择需要的参数。
```
if __name__ == "__main__":
    # a,b: 需要分类的数字;
    # lr: 学习率; 
    # epochs: 轮数; 
    # visualize: 是否可视化
    trained_svm = train_and_evaluate_svm(a=4, b=9, lr=0.0001, epochs=100, visualize = True)
```

然后运行：
```
python exp.py
```
