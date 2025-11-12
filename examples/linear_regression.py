"""
线性回归训练示例
使用自制的 micrograd 自动微分库训练线性回归模型

任务：
- 定义线性模型 y_pred = w * x + b
- 实现均方误差损失函数 (MSE)
- 使用梯度下降优化参数 w 和 b
"""

import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minitf.tensor import Tensor


def generate_data(n_samples=100, true_w=2.5, true_b=1.5, noise=0.1):
    """
    生成线性回归的训练数据
    y = true_w * x + true_b + noise
    
    Args:
        n_samples: 样本数量
        true_w: 真实的权重
        true_b: 真实的偏置
        noise: 噪声标准差
    
    Returns:
        x_data: 输入数据
        y_data: 目标数据
    """
    np.random.seed(42)
    x_data = np.random.randn(n_samples, 1).astype(np.float32)
    y_data = true_w * x_data + true_b + noise * np.random.randn(n_samples, 1).astype(np.float32)
    return x_data, y_data


def mse_loss(y_pred, y_true):
    """
    计算均方误差损失
    MSE = mean((y_pred - y_true)^2)
    
    Args:
        y_pred: 预测值 (Tensor)
        y_true: 真实值 (Tensor)
    
    Returns:
        loss: 损失值 (Tensor)
    """
    diff = y_pred + (y_true * Tensor(-1.0))  # y_pred - y_true
    squared_diff = diff ** 2
    loss = squared_diff.sum() * Tensor(1.0 / len(y_true.data))
    return loss


def train_linear_regression(x_data, y_data, learning_rate=0.01, epochs=100, print_every=10):
    """
    训练线性回归模型
    
    Args:
        x_data: 输入数据
        y_data: 目标数据
        learning_rate: 学习率
        epochs: 训练轮数
        print_every: 每隔多少轮打印一次
    
    Returns:
        w: 训练后的权重
        b: 训练后的偏置
    """
    # 初始化参数
    w = Tensor(np.random.randn(1, 1).astype(np.float32), requires_grad=True)
    b = Tensor(np.zeros((1, 1), dtype=np.float32), requires_grad=True)
    
    print("=" * 60)
    print("开始训练线性回归模型")
    print("=" * 60)
    print(f"初始参数: w = {w.data[0, 0]:.4f}, b = {b.data[0, 0]:.4f}")
    print(f"学习率: {learning_rate}, 训练轮数: {epochs}")
    print("=" * 60)
    
    for epoch in range(epochs):
        # 前向传播: y_pred = w * x + b
        x = Tensor(x_data, requires_grad=False)
        y_true = Tensor(y_data, requires_grad=False)
        
        # 计算预测值
        y_pred = x * w + b
        
        # 计算损失
        loss = mse_loss(y_pred, y_true)
        
        # 反向传播
        loss.backward()
        
        # 梯度下降更新参数
        w.data -= learning_rate * w.grad
        b.data -= learning_rate * b.grad
        
        # 清零梯度
        w.grad = np.zeros_like(w.data)
        b.grad = np.zeros_like(b.data)
        
        # 打印训练信息
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | Loss: {loss.data:.6f} | "
                  f"w: {w.data[0, 0]:.4f} | b: {b.data[0, 0]:.4f}")
    
    print("=" * 60)
    print("训练完成!")
    print(f"最终参数: w = {w.data[0, 0]:.4f}, b = {b.data[0, 0]:.4f}")
    print("=" * 60)
    
    return w, b


def main():
    # 生成数据
    true_w = 2.5
    true_b = 1.5
    n_samples = 100
    
    print("\n生成训练数据...")
    print(f"真实参数: w = {true_w}, b = {true_b}")
    print(f"样本数量: {n_samples}")
    
    x_data, y_data = generate_data(n_samples, true_w, true_b, noise=0.1)
    
    # 训练模型
    w, b = train_linear_regression(x_data, y_data, learning_rate=0.1, epochs=100, print_every=10)
    
    # 评估结果
    print("\n模型评估:")
    print(f"真实参数: w = {true_w:.4f}, b = {true_b:.4f}")
    print(f"学习参数: w = {w.data[0, 0]:.4f}, b = {b.data[0, 0]:.4f}")
    print(f"参数误差: w_error = {abs(w.data[0, 0] - true_w):.4f}, "
          f"b_error = {abs(b.data[0, 0] - true_b):.4f}")


if __name__ == "__main__":
    main()

