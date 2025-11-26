"""
PyTorch 神经网络模块

使用 PyTorch 重写之前手工实现的组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLP(nn.Module):
    """多层感知机 (Multi-Layer Perceptron)
    
    使用 PyTorch 实现的 MLP，用于分类任务
    
    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        num_hidden_layers: 隐藏层数量（默认为1）
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=1):
        super().__init__()
        
        layers = []
        # 第一层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # 中间隐藏层
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, input_dim)
        
        Returns:
            输出张量，形状为 (batch_size, output_dim)
        """
        return self.network(x)


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    
    Args:
        dropout: Dropout 概率（默认为0.0）
    """
    
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        """前向传播
        
        Args:
            Q: Query 张量，形状为 (batch_size, seq_len, d_k)
            K: Key 张量，形状为 (batch_size, seq_len, d_k)
            V: Value 张量，形状为 (batch_size, seq_len, d_v)
            mask: 可选的掩码张量，形状为 (seq_len, seq_len) 或 (batch_size, seq_len, seq_len)
                  True 表示该位置需要被掩盖（设为 -inf）
        
        Returns:
            output: 注意力输出，形状为 (batch_size, seq_len, d_v)
            attention_weights: 注意力权重，形状为 (batch_size, seq_len, seq_len)
        """
        d_k = Q.size(-1)
        
        # 计算注意力分数: QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算输出
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention
    
    将输入投影到多个子空间，并行计算注意力，最后拼接
    
    Args:
        d_model: 模型维度
        num_heads: 注意力头数量
        dropout: Dropout 概率（默认为0.0）
    """
    
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Q, K, V 的线性投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)
        
        # Attention
        self.attention = ScaledDotProductAttention(dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        """前向传播
        
        Args:
            Q: Query 张量，形状为 (batch_size, seq_len, d_model)
            K: Key 张量，形状为 (batch_size, seq_len, d_model)
            V: Value 张量，形状为 (batch_size, seq_len, d_model)
            mask: 可选的掩码张量
        
        Returns:
            输出张量，形状为 (batch_size, seq_len, d_model)
        """
        batch_size = Q.size(0)
        
        # 线性投影并分割成多头
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 如果有掩码，需要扩展维度以匹配多头
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)

        # 计算注意力
        # (batch_size, num_heads, seq_len, d_k)
        attn_output, _ = self.attention(Q, K, V, mask)

        # 拼接多头
        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 输出投影
        output = self.W_o(attn_output)
        output = self.dropout(output)

        return output


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network

    FFN(x) = max(0, xW1 + b1)W2 + b2

    Args:
        d_model: 模型维度
        d_ff: 前馈网络隐藏层维度
        dropout: Dropout 概率（默认为0.0）
    """

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """前向传播

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)

        Returns:
            输出张量，形状为 (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer 编码器块

    包含 Multi-Head Attention、Feed-Forward Network、残差连接和 Layer Normalization
    采用 Pre-LN 结构（Layer Normalization 在子层之前）

    Args:
        d_model: 模型维度
        num_heads: 注意力头数量
        d_ff: 前馈网络隐藏层维度
        dropout: Dropout 概率（默认为0.1）
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Multi-Head Attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Position-wise Feed-Forward Network
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """前向传播

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            mask: 可选的掩码张量

        Returns:
            输出张量，形状为 (batch_size, seq_len, d_model)
        """
        # Multi-Head Attention with residual connection
        # Pre-LN: LayerNorm -> Attention -> Residual
        attn_output = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout(attn_output)

        # Feed-Forward Network with residual connection
        # Pre-LN: LayerNorm -> FFN -> Residual
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)

        return x

