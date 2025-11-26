"""
PyTorch 实现对比示例

展示如何使用 PyTorch 重写之前手工实现的组件，并对比两者的性能和易用性
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from minitf.pytorch.nn_modules import (
    MLP,
    ScaledDotProductAttention,
    MultiHeadAttention,
    TransformerBlock,
)


def example_mlp_training():
    """示例1：使用 PyTorch MLP 训练 XOR 问题"""
    print("=" * 60)
    print("示例1：使用 PyTorch MLP 训练 XOR 问题")
    print("=" * 60)
    
    # 准备数据
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    
    # 创建模型
    model = MLP(input_dim=2, hidden_dim=8, output_dim=2, num_hidden_layers=1)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    
    # 训练
    epochs = 1000
    start_time = time.time()
    
    for epoch in range(epochs):
        # 前向传播
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y).sum().item() / len(y)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2%}")
    
    training_time = time.time() - start_time
    
    # 最终测试
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y).sum().item() / len(y)
    
    print(f"\n训练完成！耗时: {training_time:.2f}秒")
    print(f"最终准确率: {accuracy:.2%}")
    print(f"预测结果: {predicted.numpy()}")
    print(f"真实标签: {y.numpy()}")


def example_attention_mechanism():
    """示例2：Attention 机制演示"""
    print("\n" + "=" * 60)
    print("示例2：Attention 机制演示")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 5
    d_model = 16
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 1. Scaled Dot-Product Attention
    print("\n1. Scaled Dot-Product Attention")
    attention = ScaledDotProductAttention()
    output, weights = attention(x, x, x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {weights.shape}")
    print(f"注意力权重和: {weights.sum(dim=-1)[0, 0].item():.4f} (应该接近1.0)")
    
    # 2. Multi-Head Attention
    print("\n2. Multi-Head Attention")
    num_heads = 4
    mha = MultiHeadAttention(d_model, num_heads)
    output = mha(x, x, x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力头数: {num_heads}")
    
    # 3. 带因果掩码的 Attention
    print("\n3. 带因果掩码的 Attention (用于自回归)")
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    output, weights = attention(x, x, x, mask)
    print(f"掩码形状: {mask.shape}")
    print(f"输出形状: {output.shape}")
    print("\n因果掩码 (True 表示被掩盖的位置):")
    print(mask.numpy())
    print("\n第一个样本的注意力权重:")
    print(weights[0].detach().numpy())


def example_transformer_block():
    """示例3：Transformer Block 演示"""
    print("\n" + "=" * 60)
    print("示例3：Transformer Block 演示")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 8
    d_ff = 256
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 单个 Transformer Block
    print("\n1. 单个 Transformer Block")
    block = TransformerBlock(d_model, num_heads, d_ff)
    output = block(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 统计参数数量
    num_params = sum(p.numel() for p in block.parameters())
    print(f"参数数量: {num_params:,}")
    
    # 堆叠多个 Transformer Block
    print("\n2. 堆叠多个 Transformer Block")
    num_layers = 6
    blocks = nn.ModuleList([
        TransformerBlock(d_model, num_heads, d_ff)
        for _ in range(num_layers)
    ])
    
    for i, block in enumerate(blocks):
        x = block(x)
        print(f"Layer {i+1} 输出形状: {x.shape}")
    
    # 统计总参数数量
    total_params = sum(p.numel() for p in blocks.parameters())
    print(f"\n总参数数量: {total_params:,}")


if __name__ == '__main__':
    # 设置随机种子以便复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行示例
    example_mlp_training()
    example_attention_mechanism()
    example_transformer_block()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)

