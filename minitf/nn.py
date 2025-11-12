"""
神经网络模块
实现多层感知机(MLP)的核心组件
"""

import numpy as np
from .tensor import Tensor


class Module:
    """所有神经网络层的基类"""
    
    def __call__(self, x):
        """前向传播"""
        return self.forward(x)
    
    def forward(self, x):
        """前向传播，子类需要实现"""
        raise NotImplementedError
    
    def parameters(self):
        """返回模块的所有参数"""
        return []
    
    def zero_grad(self):
        """清零所有参数的梯度"""
        for param in self.parameters():
            param.grad = np.zeros_like(param.data)


class Linear(Module):
    """
    线性层（全连接层）
    执行线性变换: y = xW + b
    
    Args:
        in_features: 输入特征数
        out_features: 输出特征数
        bias: 是否使用偏置项
    """
    
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # 使用 Xavier/Glorot 初始化
        # 权重初始化为均匀分布 U(-sqrt(k), sqrt(k)), 其中 k = 1/in_features
        k = 1.0 / in_features
        self.weight = Tensor(
            np.random.uniform(-np.sqrt(k), np.sqrt(k), (in_features, out_features)).astype(np.float32),
            requires_grad=True
        )
        
        if bias:
            self.bias = Tensor(
                np.random.uniform(-np.sqrt(k), np.sqrt(k), (out_features,)).astype(np.float32),
                requires_grad=True
            )
        else:
            self.bias = None
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, in_features) 或 (in_features,)
        
        Returns:
            输出张量，形状为 (batch_size, out_features) 或 (out_features,)
        """
        # 矩阵乘法: x @ W
        out = x.matmul(self.weight)
        
        # 加上偏置
        if self.use_bias:
            out = out + self.bias
        
        return out
    
    def parameters(self):
        """返回层的参数"""
        if self.use_bias:
            return [self.weight, self.bias]
        else:
            return [self.weight]
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})"


class ReLU(Module):
    """
    ReLU 激活函数
    ReLU(x) = max(0, x)
    """
    
    def forward(self, x):
        """前向传播"""
        return x.relu()
    
    def __repr__(self):
        return "ReLU()"


class Sigmoid(Module):
    """
    Sigmoid 激活函数
    Sigmoid(x) = 1 / (1 + exp(-x))
    """
    
    def forward(self, x):
        """前向传播"""
        return x.sigmoid()
    
    def __repr__(self):
        return "Sigmoid()"


class Sequential(Module):
    """
    Sequential 容器
    按顺序执行多个模块
    
    Args:
        *layers: 要按顺序执行的层
    """
    
    def __init__(self, *layers):
        self.layers = list(layers)
    
    def forward(self, x):
        """
        前向传播
        依次通过所有层
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        """返回所有层的参数"""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def __repr__(self):
        layer_str = "\n  ".join([f"({i}): {layer}" for i, layer in enumerate(self.layers)])
        return f"Sequential(\n  {layer_str}\n)"


# 损失函数

def softmax(x: Tensor, axis=-1):
    """
    Softmax 函数
    Softmax(x) = exp(x) / sum(exp(x))
    Softmax将向量 x 的每个元素映射到 [0, 1] 范围内，并使其和为 1。

    Args:
        x: 输入张量
        axis: 应用 softmax 的轴

    Returns:
        softmax 输出
    """
    # 为了数值稳定性，减去最大值
    x_max = x.max(axis=axis, keepdims=True)
    exp_x = (x + (x_max * Tensor(-1.0))).exp()
    sum_exp = exp_x.sum(axis=axis, keepdims=True)
    return exp_x / sum_exp


def cross_entropy_loss(logits, targets):
    """
    交叉熵损失函数

    Args:
        logits: 模型输出的原始分数（未经过 softmax），形状为 (batch_size, num_classes)
        targets: 目标标签，形状为 (batch_size,)，包含类别索引（0 到 num_classes-1）

    Returns:
        平均交叉熵损失（标量）

    公式:
        loss = -mean(log(softmax(logits)[i, targets[i]]))
    """
    batch_size = logits.data.shape[0]
    num_classes = logits.data.shape[1]

    # 计算 softmax
    probs = softmax(logits, axis=1)

    # 提取正确类别的概率
    # 创建 one-hot 编码
    targets_np = targets.data if isinstance(targets, Tensor) else targets
    targets_np = targets_np.astype(np.int32).flatten()

    # 手动提取对应类别的概率
    # 由于我们的 Tensor 不支持高级索引，我们需要用矩阵乘法来实现
    one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
    one_hot[np.arange(batch_size), targets_np] = 1.0
    one_hot_tensor = Tensor(one_hot, requires_grad=False)

    # 提取正确类别的概率: sum(probs * one_hot, axis=1)
    correct_probs = (probs * one_hot_tensor).sum(axis=1)

    # 计算负对数似然
    # 为了数值稳定性，添加一个小的 epsilon
    epsilon = 1e-7
    log_probs = (correct_probs + Tensor(epsilon)).log()

    # 计算平均损失
    loss = (log_probs.sum() * Tensor(-1.0 / batch_size))

    return loss


def mse_loss(y_pred, y_true):
    """
    均方误差损失函数

    Args:
        y_pred: 预测值
        y_true: 真实值

    Returns:
        均方误差损失
    """
    diff = y_pred + (y_true * Tensor(-1.0))
    squared_diff = diff ** 2
    loss = squared_diff.sum() * Tensor(1.0 / y_true.data.size)
    return loss

