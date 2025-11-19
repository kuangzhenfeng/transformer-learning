"""
Attention 机制示例
演示 Scaled Dot-Product Attention 和 Multi-Head Attention 的使用

阶段二任务1: 实现 Scaled Dot-Product Attention 和 Multi-Head Attention
"""

import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minitf.tensor import Tensor
from minitf.nn import scaled_dot_product_attention, MultiHeadAttention


def example_scaled_dot_product_attention():
    """
    示例1: Scaled Dot-Product Attention
    
    这是 Transformer 中最基础的注意力机制
    公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    print("=" * 60)
    print("示例1: Scaled Dot-Product Attention")
    print("=" * 60)
    
    # 设置参数
    batch_size = 1
    seq_len = 4  # 序列长度
    d_k = 8      # key 的维度
    d_v = 8      # value 的维度
    
    # 创建示例数据
    # 假设我们有一个简单的句子，每个词用一个向量表示
    np.random.seed(42)
    Q = Tensor(np.random.randn(batch_size, seq_len, d_k), requires_grad=True)
    K = Tensor(np.random.randn(batch_size, seq_len, d_k), requires_grad=True)
    V = Tensor(np.random.randn(batch_size, seq_len, d_v), requires_grad=True)
    
    print(f"\n输入形状:")
    print(f"  Q (Query):  {Q.shape}")
    print(f"  K (Key):    {K.shape}")
    print(f"  V (Value):  {V.shape}")
    
    # 计算注意力
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"\n输出形状:")
    print(f"  Output:            {output.shape}")
    print(f"  Attention Weights: {attention_weights.shape}")
    
    print(f"\n注意力权重矩阵 (每行和为1):")
    print(attention_weights.data[0])
    print(f"\n每行的和: {np.sum(attention_weights.data[0], axis=1)}")
    
    # 测试反向传播
    loss = output.sum()
    loss.backward()
    
    print(f"\n梯度已计算:")
    print(f"  Q.grad shape: {Q.grad.shape}")
    print(f"  K.grad shape: {K.grad.shape}")
    print(f"  V.grad shape: {V.grad.shape}")


def example_attention_with_mask():
    """
    示例2: 带掩码的注意力（因果掩码）
    
    在语言模型中，我们需要确保每个位置只能关注它之前的位置
    这通过因果掩码（causal mask）实现
    """
    print("\n" + "=" * 60)
    print("示例2: 带因果掩码的 Attention")
    print("=" * 60)
    
    batch_size = 1
    seq_len = 5
    d_k = 8
    
    # 创建数据
    np.random.seed(42)
    Q = Tensor(np.random.randn(batch_size, seq_len, d_k), requires_grad=True)
    K = Tensor(np.random.randn(batch_size, seq_len, d_k), requires_grad=True)
    V = Tensor(np.random.randn(batch_size, seq_len, d_k), requires_grad=True)
    
    # 创建因果掩码（上三角矩阵）
    # mask[i, j] = True 表示位置 i 不能关注位置 j
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    mask = np.broadcast_to(mask, (batch_size, seq_len, seq_len))
    
    print(f"\n因果掩码 (True 表示被掩盖的位置):")
    print(mask[0].astype(int))
    
    # 计算带掩码的注意力
    output, attention_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
    
    print(f"\n注意力权重矩阵 (被掩盖的位置接近0):")
    print(attention_weights.data[0])
    print(f"\n说明: 每个位置只能关注它自己和之前的位置")


def example_multi_head_attention():
    """
    示例3: Multi-Head Attention
    
    多头注意力允许模型同时关注不同表示子空间的信息
    """
    print("\n" + "=" * 60)
    print("示例3: Multi-Head Attention")
    print("=" * 60)
    
    # 设置参数
    batch_size = 2  # 批量大小：并行处理的序列数量，用于优化并行效率
    seq_len = 6     # 序列长度：单个序列的token数量，直接控制模型上下文窗口大小
    d_model = 16    # 模型维度：每个token的向量维度，决定模型的表示能力和参数规模
    num_heads = 4   # 注意力头数：多头注意力的头数量，允许模型同时关注不同方面的语义信息
    
    print(f"\n参数:")
    print(f"  d_model (模型维度): {d_model}")
    print(f"  num_heads (头数):   {num_heads}")
    print(f"  d_k (每个头的维度): {d_model // num_heads}")
    
    # 创建 Multi-Head Attention 层
    mha = MultiHeadAttention(d_model, num_heads)
    
    print(f"\n模型结构:")
    print(f"  {mha}")
    print(f"  参数数量: {sum(p.data.size for p in mha.parameters())}")
    
    # 创建输入（Self-Attention 的情况）
    np.random.seed(42)
    X = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
    
    print(f"\n输入形状: {X.shape}")
    
    # 前向传播
    output = mha(X, X, X)  # Self-Attention: Q=K=V=X
    
    print(f"输出形状: {output.shape}")
    
    # 测试反向传播
    loss = output.sum()
    loss.backward()
    
    print(f"\n梯度已计算:")
    print(f"  输入梯度形状: {X.grad.shape}")
    print(f"  参数梯度已计算: {all(p.grad is not None for p in mha.parameters())}")


if __name__ == "__main__":
    # 运行所有示例
    example_scaled_dot_product_attention()
    example_attention_with_mask()
    example_multi_head_attention()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)

