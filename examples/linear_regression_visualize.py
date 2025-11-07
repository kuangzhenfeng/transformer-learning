"""
线性回归训练可视化示例
展示训练过程中损失和参数的变化
"""

import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tensor import Tensor


def generate_data(n_samples=100, true_w=2.5, true_b=1.5, noise=0.1):
    """生成线性回归的训练数据"""
    np.random.seed(42)
    x_data = np.random.randn(n_samples, 1).astype(np.float32)
    y_data = true_w * x_data + true_b + noise * np.random.randn(n_samples, 1).astype(np.float32)
    return x_data, y_data


def mse_loss(y_pred, y_true):
    """计算均方误差损失"""
    diff = y_pred + (y_true * Tensor(-1.0))
    squared_diff = diff ** 2
    loss = squared_diff.sum() * Tensor(1.0 / len(y_true.data))
    return loss


def train_with_history(x_data, y_data, learning_rate=0.1, epochs=100):
    """训练并记录历史"""
    # 初始化参数
    w = Tensor(np.random.randn(1, 1).astype(np.float32), requires_grad=True)
    b = Tensor(np.zeros((1, 1), dtype=np.float32), requires_grad=True)
    
    history = {
        'loss': [],
        'w': [],
        'b': []
    }
    
    for epoch in range(epochs):
        # 前向传播
        x = Tensor(x_data, requires_grad=False)
        y_true = Tensor(y_data, requires_grad=False)
        y_pred = x * w + b
        
        # 计算损失
        loss = mse_loss(y_pred, y_true)
        
        # 记录历史
        history['loss'].append(float(loss.data))
        history['w'].append(float(w.data[0, 0]))
        history['b'].append(float(b.data[0, 0]))
        
        # 反向传播
        loss.backward()
        
        # 梯度下降
        w.data -= learning_rate * w.grad
        b.data -= learning_rate * b.grad
        
        # 清零梯度
        w.grad = np.zeros_like(w.data)
        b.grad = np.zeros_like(b.data)
    
    return w, b, history


def print_training_summary(history, true_w=2.5, true_b=1.5):
    """打印训练摘要"""
    print("\n" + "=" * 70)
    print("训练摘要")
    print("=" * 70)
    
    print(f"\n初始状态:")
    print(f"  损失: {history['loss'][0]:.6f}")
    print(f"  参数: w = {history['w'][0]:.4f}, b = {history['b'][0]:.4f}")
    
    print(f"\n最终状态:")
    print(f"  损失: {history['loss'][-1]:.6f}")
    print(f"  参数: w = {history['w'][-1]:.4f}, b = {history['b'][-1]:.4f}")
    
    print(f"\n真实参数:")
    print(f"  w = {true_w:.4f}, b = {true_b:.4f}")
    
    print(f"\n参数误差:")
    print(f"  w_error = {abs(history['w'][-1] - true_w):.4f}")
    print(f"  b_error = {abs(history['b'][-1] - true_b):.4f}")
    
    print(f"\n损失下降:")
    print(f"  初始损失: {history['loss'][0]:.6f}")
    print(f"  最终损失: {history['loss'][-1]:.6f}")
    print(f"  下降比例: {(1 - history['loss'][-1]/history['loss'][0]) * 100:.2f}%")
    
    # 显示关键训练节点
    print(f"\n训练进度:")
    milestones = [0, 9, 19, 49, 99]
    for i in milestones:
        if i < len(history['loss']):
            print(f"  Epoch {i+1:3d}: Loss = {history['loss'][i]:.6f}, "
                  f"w = {history['w'][i]:.4f}, b = {history['b'][i]:.4f}")
    
    print("=" * 70)


def main():
    """主函数"""
    # 生成数据
    true_w = 2.5
    true_b = 1.5
    n_samples = 100
    
    print("线性回归训练可视化")
    print("=" * 70)
    print(f"真实参数: w = {true_w}, b = {true_b}")
    print(f"样本数量: {n_samples}")
    print(f"学习率: 0.1")
    print(f"训练轮数: 100")
    
    x_data, y_data = generate_data(n_samples, true_w, true_b, noise=0.1)
    
    # 训练模型
    w, b, history = train_with_history(x_data, y_data, learning_rate=0.1, epochs=100)
    
    # 打印摘要
    print_training_summary(history, true_w, true_b)


if __name__ == "__main__":
    main()

