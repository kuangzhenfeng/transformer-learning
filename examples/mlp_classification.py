"""
多层感知机(MLP)分类示例
使用自制的神经网络库训练 MLP 解决分类问题

任务：
- 实现 MLP 模型
- 在 XOR 问题上训练
- 在多类分类问题上训练
- 使用交叉熵损失和梯度下降优化
"""

import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minitf.tensor import Tensor
from minitf.nn import Linear, ReLU, Sigmoid, Sequential, cross_entropy_loss


def train_xor():
    """
    训练 MLP 解决 XOR 问题
    XOR 是一个经典的非线性分类问题，单层感知机无法解决
    """
    print("\n" + "=" * 70)
    print("训练 MLP 解决 XOR 问题")
    print("=" * 70)
    
    # XOR 数据集
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ], dtype=np.float32)
    
    y = np.array([0, 1, 1, 0], dtype=np.int32)  # XOR 输出
    
    # 创建 MLP 模型: 2 -> 4 -> 2
    # 输入层: 2 个特征
    # 隐藏层: 4 个神经元，ReLU 激活
    # 输出层: 2 个类别（0 或 1）
    model = Sequential(
        Linear(2, 4, bias=True),
        ReLU(),
        Linear(4, 2, bias=True)
    )
    
    print(f"\n模型结构:")
    print(model)
    
    # 训练参数
    learning_rate = 0.1
    epochs = 1000
    print_every = 100
    
    print(f"\n训练参数:")
    print(f"  学习率: {learning_rate}")
    print(f"  训练轮数: {epochs}")
    print(f"  批次大小: {len(X)} (全批次)")
    
    print(f"\n开始训练...")
    print("-" * 70)
    
    for epoch in range(epochs):
        # 前向传播
        x_tensor = Tensor(X, requires_grad=False)
        logits = model(x_tensor)
        
        # 计算损失
        loss = cross_entropy_loss(logits, y)
        
        # 反向传播
        loss.backward()
        
        # 梯度下降
        for param in model.parameters():
            param.data -= learning_rate * param.grad
        
        # 清零梯度
        model.zero_grad()
        
        # 打印训练信息
        if (epoch + 1) % print_every == 0 or epoch == 0:
            # 计算准确率
            predictions = np.argmax(logits.data, axis=1)
            accuracy = np.mean(predictions == y)
            print(f"Epoch [{epoch+1:4d}/{epochs}] | Loss: {loss.data:.6f} | Accuracy: {accuracy:.4f}")
    
    print("-" * 70)
    print("训练完成!")
    
    # 最终评估
    print(f"\n最终评估:")
    x_tensor = Tensor(X, requires_grad=False)
    logits = model(x_tensor)
    predictions = np.argmax(logits.data, axis=1)
    
    print(f"\n输入 -> 预测 (真实)")
    for i in range(len(X)):
        print(f"  {X[i]} -> {predictions[i]} ({y[i]})")
    
    accuracy = np.mean(predictions == y)
    print(f"\n最终准确率: {accuracy:.4f}")
    
    return model


def generate_spiral_data(n_samples=100, n_classes=3, noise=0.1):
    """
    生成螺旋形状的多类分类数据
    
    Args:
        n_samples: 每个类别的样本数
        n_classes: 类别数
        noise: 噪声水平
    
    Returns:
        X: 特征数据 (n_samples * n_classes, 2)
        y: 标签 (n_samples * n_classes,)
    """
    np.random.seed(42)
    X = np.zeros((n_samples * n_classes, 2), dtype=np.float32)
    y = np.zeros(n_samples * n_classes, dtype=np.int32)
    
    for class_idx in range(n_classes):
        ix = range(n_samples * class_idx, n_samples * (class_idx + 1))
        r = np.linspace(0.0, 1, n_samples)  # 半径
        t = np.linspace(class_idx * 4, (class_idx + 1) * 4, n_samples) + np.random.randn(n_samples) * noise
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = class_idx
    
    return X, y


def train_spiral():
    """
    训练 MLP 解决螺旋形多类分类问题
    """
    print("\n" + "=" * 70)
    print("训练 MLP 解决螺旋形多类分类问题")
    print("=" * 70)
    
    # 生成螺旋数据
    n_samples = 100
    n_classes = 3
    X, y = generate_spiral_data(n_samples, n_classes, noise=0.2)
    
    print(f"\n数据集信息:")
    print(f"  样本数: {len(X)}")
    print(f"  特征数: {X.shape[1]}")
    print(f"  类别数: {n_classes}")
    
    # 创建 MLP 模型: 2 -> 16 -> 16 -> 3
    model = Sequential(
        Linear(2, 16, bias=True),
        ReLU(),
        Linear(16, 16, bias=True),
        ReLU(),
        Linear(16, n_classes, bias=True)
    )
    
    print(f"\n模型结构:")
    print(model)
    
    # 训练参数
    learning_rate = 0.5
    epochs = 500
    print_every = 50
    
    print(f"\n训练参数:")
    print(f"  学习率: {learning_rate}")
    print(f"  训练轮数: {epochs}")
    
    print(f"\n开始训练...")
    print("-" * 70)
    
    for epoch in range(epochs):
        # 前向传播
        x_tensor = Tensor(X, requires_grad=False)
        logits = model(x_tensor)
        
        # 计算损失
        loss = cross_entropy_loss(logits, y)
        
        # 反向传播
        loss.backward()
        
        # 梯度下降
        for param in model.parameters():
            param.data -= learning_rate * param.grad
        
        # 清零梯度
        model.zero_grad()
        
        # 打印训练信息
        if (epoch + 1) % print_every == 0 or epoch == 0:
            # 计算准确率
            predictions = np.argmax(logits.data, axis=1)
            accuracy = np.mean(predictions == y)
            print(f"Epoch [{epoch+1:4d}/{epochs}] | Loss: {loss.data:.6f} | Accuracy: {accuracy:.4f}")
    
    print("-" * 70)
    print("训练完成!")
    
    # 最终评估
    x_tensor = Tensor(X, requires_grad=False)
    logits = model(x_tensor)
    predictions = np.argmax(logits.data, axis=1)
    accuracy = np.mean(predictions == y)
    
    print(f"\n最终准确率: {accuracy:.4f}")

    def visualize(model: Sequential, X: np.ndarray, y: np.ndarray):
        import matplotlib.pyplot as plt
        
        # 创建网格点用于绘制决策边界
        h = 0.01
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # 预测网格点
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_tensor = Tensor(grid_points, requires_grad=False)
        grid_logits = model(grid_tensor)
        grid_predictions = np.argmax(grid_logits.data, axis=1)
        grid_predictions = grid_predictions.reshape(xx.shape)
        
        # 绘制决策边界和数据点
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, grid_predictions, alpha=0.3, cmap=plt.cm.viridis)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.viridis)
        plt.colorbar(scatter)
        plt.title(f"Spiral Classification Results (Accuracy: {accuracy:.4f})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    visualize(model, X, y)
    
    return model


def main():
    """主函数"""
    print("\n多层感知机(MLP)分类示例")
    
    # 训练 XOR 问题
    train_xor()
    
    # 训练螺旋分类问题
    train_spiral()
    
    print("\n" + "=" * 70)
    print("所有训练完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()

