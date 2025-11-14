"""
神经网络模块
实现多层感知机(MLP)的核心组件
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from .tensor import Tensor


class Module:
    """所有神经网络层的基类"""

    def __call__(self, *args, **kwargs) -> Tensor:
        """前向传播"""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        """前向传播，子类需要实现"""
        raise NotImplementedError

    def parameters(self) -> List[Tensor]:
        """返回模块的所有参数"""
        return []

    def zero_grad(self) -> None:
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

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
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

    def forward(self, x: Tensor) -> Tensor:
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

    def parameters(self) -> List[Tensor]:
        """返回层的参数"""
        if self.use_bias:
            return [self.weight, self.bias]
        else:
            return [self.weight]

    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})"


class ReLU(Module):
    """
    ReLU 激活函数
    ReLU(x) = max(0, x)
    """

    def forward(self, x: Tensor) -> Tensor:
        """前向传播"""
        return x.relu()

    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid(Module):
    """
    Sigmoid 激活函数
    Sigmoid(x) = 1 / (1 + exp(-x))
    """

    def forward(self, x: Tensor) -> Tensor:
        """前向传播"""
        return x.sigmoid()

    def __repr__(self) -> str:
        return "Sigmoid()"


class Sequential(Module):
    """
    Sequential 容器
    按顺序执行多个模块

    Args:
        *layers: 要按顺序执行的层
    """

    def __init__(self, *layers: Module) -> None:
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        依次通过所有层
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Tensor]:
        """返回所有层的参数"""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __repr__(self) -> str:
        layer_str = "\n  ".join([f"({i}): {layer}" for i, layer in enumerate(self.layers)])
        return f"Sequential(\n  {layer_str}\n)"


# 损失函数

def softmax(x: Tensor, axis: int = -1) -> Tensor:
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


def cross_entropy_loss(logits: Tensor, targets: Union[Tensor, np.ndarray]) -> Tensor:
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


def mse_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
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


# Attention 机制

def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[np.ndarray] = None
) -> Tuple[Tensor, Tensor]:
    """
    Scaled Dot-Product Attention

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    Args:
        Q: Query 张量，形状为 (..., seq_len_q, d_k)
        K: Key 张量，形状为 (..., seq_len_k, d_k)
        V: Value 张量，形状为 (..., seq_len_v, d_v)，其中 seq_len_k == seq_len_v
        mask: 可选的掩码张量，形状为 (..., seq_len_q, seq_len_k)
              掩码中为 True 的位置会被设置为 -inf（在 softmax 之前）

    Returns:
        attention_output: 注意力输出，形状为 (..., seq_len_q, d_v)
        attention_weights: 注意力权重，形状为 (..., seq_len_q, seq_len_k)

    注意：
        - d_k 是 key 的维度
        - seq_len_k 必须等于 seq_len_v
    """
    # 获取 d_k（key 的维度）
    d_k = K.shape[-1]

    # 计算注意力分数: QK^T
    # Q: (..., seq_len_q, d_k)
    # K^T: (..., d_k, seq_len_k)
    # scores: (..., seq_len_q, seq_len_k)
    K_T = K.transpose(tuple(range(len(K.shape) - 2)) + (len(K.shape) - 1, len(K.shape) - 2))
    scores = Q.matmul(K_T)

    # 缩放: scores / sqrt(d_k)
    scale = Tensor(np.sqrt(d_k))
    scaled_scores = scores / scale

    # 应用掩码（如果提供）
    if mask is not None:
        # 将掩码为 True 的位置设置为一个很大的负数
        mask_value = -1e9
        mask_tensor = Tensor(mask.astype(np.float32) * mask_value)
        scaled_scores = scaled_scores + mask_tensor

    # 应用 softmax 得到注意力权重
    attention_weights = softmax(scaled_scores, axis=-1)

    # 计算加权和: attention_weights @ V
    # attention_weights: (..., seq_len_q, seq_len_k)
    # V: (..., seq_len_v, d_v)
    # output: (..., seq_len_q, d_v)
    attention_output = attention_weights.matmul(V)

    return attention_output, attention_weights


class MultiHeadAttention(Module):
    """
    Multi-Head Attention 层

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        d_model: 模型的维度（输入和输出的维度）
        num_heads: 注意力头的数量
        dropout: dropout 比率（暂不实现）
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # Q, K, V 的线性变换
        self.W_q = Linear(d_model, d_model, bias=False)
        self.W_k = Linear(d_model, d_model, bias=False)
        self.W_v = Linear(d_model, d_model, bias=False)

        # 输出的线性变换
        self.W_o = Linear(d_model, d_model, bias=False)

    def split_heads(self, x: Tensor, batch_size: int) -> Tensor:
        """
        将最后一个维度分割成 (num_heads, d_k)
        然后转置以便进行多头注意力计算

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)

        Returns:
            形状为 (batch_size, num_heads, seq_len, d_k) 的张量
        """
        # x: (batch_size, seq_len, d_model)
        seq_len = x.shape[1]

        # 重塑为 (batch_size, seq_len, num_heads, d_k)
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)

        # 转置为 (batch_size, num_heads, seq_len, d_k)
        x = x.transpose((0, 2, 1, 3))

        return x

    def combine_heads(self, x: Tensor, batch_size: int) -> Tensor:
        """
        将多头的输出合并回来

        Args:
            x: 输入张量，形状为 (batch_size, num_heads, seq_len, d_k)

        Returns:
            形状为 (batch_size, seq_len, d_model) 的张量
        """
        # x: (batch_size, num_heads, seq_len, d_k)
        seq_len = x.shape[2]

        # 转置为 (batch_size, seq_len, num_heads, d_k)
        x = x.transpose((0, 2, 1, 3))

        # 重塑为 (batch_size, seq_len, d_model)
        x = x.reshape(batch_size, seq_len, self.d_model)

        return x

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        """
        前向传播

        Args:
            Q: Query 张量，形状为 (batch_size, seq_len_q, d_model)
            K: Key 张量，形状为 (batch_size, seq_len_k, d_model)
            V: Value 张量，形状为 (batch_size, seq_len_v, d_model)
            mask: 可选的掩码，形状为 (batch_size, seq_len_q, seq_len_k) 或可广播的形状

        Returns:
            output: 注意力输出，形状为 (batch_size, seq_len_q, d_model)
        """
        batch_size = Q.shape[0]

        # 线性变换
        # (batch_size, seq_len, d_model)
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # 分割成多个头
        # (batch_size, num_heads, seq_len, d_k)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # 应用 scaled dot-product attention
        # 注意：这里我们需要对每个 batch 和每个 head 分别计算
        # 由于我们的实现支持批量操作，可以直接计算
        attention_output, _ = scaled_dot_product_attention(Q, K, V, mask)

        # 合并多个头
        # (batch_size, seq_len_q, d_model)
        output = self.combine_heads(attention_output, batch_size)

        # 最后的线性变换
        output = self.W_o(output)

        return output

    def parameters(self) -> List[Tensor]:
        """返回所有参数"""
        params = []
        params.extend(self.W_q.parameters())
        params.extend(self.W_k.parameters())
        params.extend(self.W_v.parameters())
        params.extend(self.W_o.parameters())
        return params

    def __repr__(self) -> str:
        return f"MultiHeadAttention(d_model={self.d_model}, num_heads={self.num_heads}, d_k={self.d_k})"

