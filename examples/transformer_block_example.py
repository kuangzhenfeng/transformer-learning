"""
Transformer Block 示例
演示完整的 Transformer 编码器块的使用

阶段二任务2: 组装完整的 Transformer 编码器块
- LayerNorm 层和残差连接
- Position-wise Feed-Forward Network
- 完整的 Transformer Block
"""

import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minitf.tensor import Tensor
from minitf.nn import LayerNorm, PositionwiseFeedForward, TransformerBlock


def example_layernorm():
    """示例1: LayerNorm 的基本使用"""
    print("=" * 60)
    print("示例1: LayerNorm 的基本使用")
    print("=" * 60)
    
    # 创建 LayerNorm 层
    d_model = 4
    layer_norm = LayerNorm(d_model)
    
    # 创建输入数据
    # 形状: (batch_size=2, seq_len=3, d_model=4)
    x = Tensor(np.random.randn(2, 3, d_model), requires_grad=True)
    
    print(f"输入形状: {x.shape}")
    print(f"输入数据:\n{x.data}")
    
    # 前向传播
    y = layer_norm(x)
    
    print(f"\n输出形状: {y.shape}")
    print(f"输出数据:\n{y.data}")
    
    # 验证归一化效果：每个特征维度的均值应该接近0，方差应该接近1
    print("\n验证归一化效果（对最后一个维度）:")
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            mean = np.mean(y.data[i, j])
            var = np.var(y.data[i, j])
            print(f"  位置 [{i}, {j}]: mean={mean:.6f}, var={var:.6f}")
    
    # 反向传播
    loss = y.sum()
    loss.backward()
    
    print(f"\n输入梯度形状: {x.grad.shape}")
    print(f"Gamma 梯度: {layer_norm.gamma.grad}")
    print(f"Beta 梯度: {layer_norm.beta.grad}")


def example_feedforward():
    """示例2: Position-wise Feed-Forward Network"""
    print("\n" + "=" * 60)
    print("示例2: Position-wise Feed-Forward Network")
    print("=" * 60)
    
    # 创建 FFN
    d_model = 8
    d_ff = 32  # 通常是 d_model 的 4 倍
    ffn = PositionwiseFeedForward(d_model, d_ff)
    
    # 创建输入数据
    batch_size, seq_len = 2, 5
    x = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
    
    print(f"输入形状: {x.shape}")
    print(f"FFN 结构: {d_model} -> {d_ff} -> {d_model}")
    
    # 前向传播
    y = ffn(x)
    
    print(f"输出形状: {y.shape}")
    print(f"参数数量: {len(ffn.parameters())}")
    
    # 反向传播
    loss = y.sum()
    loss.backward()
    
    print(f"输入梯度形状: {x.grad.shape}")


def example_transformer_block():
    """示例3: 完整的 Transformer 编码器块"""
    print("\n" + "=" * 60)
    print("示例3: 完整的 Transformer 编码器块")
    print("=" * 60)
    
    # 创建 Transformer Block
    d_model = 16
    num_heads = 4
    d_ff = 64
    
    block = TransformerBlock(d_model, num_heads, d_ff)
    
    print(f"Transformer Block 配置:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_ff: {d_ff}")
    print(f"  每个头的维度: {d_model // num_heads}")
    
    # 创建输入数据
    batch_size, seq_len = 2, 10
    x = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
    
    print(f"\n输入形状: {x.shape}")
    
    # 前向传播（不使用掩码）
    y = block(x)
    
    print(f"输出形状: {y.shape}")
    print(f"参数数量: {len(block.parameters())}")
    
    # 反向传播
    loss = y.sum()
    loss.backward()

    print(f"输入梯度形状: {x.grad.shape}")


def example_transformer_block_with_mask():
    """示例4: 带因果掩码的 Transformer 编码器块"""
    print("\n" + "=" * 60)
    print("示例4: 带因果掩码的 Transformer 编码器块")
    print("=" * 60)

    # 创建 Transformer Block
    d_model = 16
    num_heads = 4
    d_ff = 64

    block = TransformerBlock(d_model, num_heads, d_ff)

    # 创建输入数据
    batch_size, seq_len = 2, 5
    x = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)

    print(f"输入形状: {x.shape}")

    # 创建因果掩码（上三角掩码）
    # 这样每个位置只能看到它自己和之前的位置
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)

    print(f"\n因果掩码 (True 表示被掩盖的位置):")
    print(mask.astype(int))

    # 前向传播（使用掩码）
    y = block(x, mask=mask)

    print(f"\n输出形状: {y.shape}")

    # 反向传播
    loss = y.sum()
    loss.backward()

    print(f"输入梯度形状: {x.grad.shape}")


def example_stacked_transformer_blocks():
    """示例5: 堆叠多个 Transformer 编码器块"""
    print("\n" + "=" * 60)
    print("示例5: 堆叠多个 Transformer 编码器块")
    print("=" * 60)

    # 创建多个 Transformer Block
    d_model = 16
    num_heads = 4
    d_ff = 64
    num_layers = 3

    blocks = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]

    print(f"堆叠 {num_layers} 个 Transformer Block")
    print(f"每个 Block 的配置: d_model={d_model}, num_heads={num_heads}, d_ff={d_ff}")

    # 创建输入数据
    batch_size, seq_len = 2, 8
    x = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)

    print(f"\n输入形状: {x.shape}")

    # 依次通过每个 Block
    for i, block in enumerate(blocks):
        x = block(x)
        print(f"经过 Block {i+1} 后的形状: {x.shape}")

    # 反向传播
    loss = x.sum()
    loss.backward()

    print(f"\n最终输出形状: {x.shape}")

    # 统计总参数数量
    total_params = sum(len(block.parameters()) for block in blocks)
    print(f"总参数数量: {total_params}")


def example_residual_connection_effect():
    """示例6: 演示残差连接的效果"""
    print("\n" + "=" * 60)
    print("示例6: 演示残差连接的效果")
    print("=" * 60)

    d_model = 8
    num_heads = 2
    d_ff = 32

    block = TransformerBlock(d_model, num_heads, d_ff)

    # 创建简单的输入
    x = Tensor(np.ones((1, 3, d_model)), requires_grad=False)

    print(f"输入 (全1向量):")
    print(x.data[0])

    # 前向传播
    y = block(x)

    print(f"\n输出:")
    print(y.data[0])

    print("\n说明: 由于残差连接的存在，即使经过复杂的变换，")
    print("输出仍然保留了输入的一些信息，这有助于梯度流动和训练深层网络。")


if __name__ == "__main__":
    # 设置随机种子以便结果可复现
    np.random.seed(42)

    # 运行所有示例
    example_layernorm()
    example_feedforward()
    example_transformer_block()
    example_transformer_block_with_mask()
    example_stacked_transformer_blocks()
    example_residual_connection_effect()

    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)

