"""
PyTorch 神经网络模块的单元测试
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from minitf.pytorch.nn_modules import (
    MLP,
    ScaledDotProductAttention,
    MultiHeadAttention,
    PositionwiseFeedForward,
    TransformerBlock,
)


class TestMLP(unittest.TestCase):
    """测试 MLP 模块"""
    
    def test_mlp_forward(self):
        """测试 MLP 前向传播"""
        batch_size = 4
        input_dim = 10
        hidden_dim = 20
        output_dim = 5
        
        model = MLP(input_dim, hidden_dim, output_dim)
        x = torch.randn(batch_size, input_dim)
        
        output = model(x)
        
        self.assertEqual(output.shape, (batch_size, output_dim))
    
    def test_mlp_backward(self):
        """测试 MLP 反向传播"""
        model = MLP(10, 20, 5)
        x = torch.randn(4, 10)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # 检查梯度是否计算
        for param in model.parameters():
            self.assertIsNotNone(param.grad)
    
    def test_mlp_multi_hidden_layers(self):
        """测试多隐藏层 MLP"""
        model = MLP(10, 20, 5, num_hidden_layers=3)
        x = torch.randn(4, 10)
        
        output = model(x)
        
        self.assertEqual(output.shape, (4, 5))


class TestScaledDotProductAttention(unittest.TestCase):
    """测试 Scaled Dot-Product Attention"""
    
    def test_attention_forward(self):
        """测试注意力前向传播"""
        batch_size = 2
        seq_len = 5
        d_k = 8
        d_v = 8
        
        attention = ScaledDotProductAttention()
        Q = torch.randn(batch_size, seq_len, d_k)
        K = torch.randn(batch_size, seq_len, d_k)
        V = torch.randn(batch_size, seq_len, d_v)
        
        output, weights = attention(Q, K, V)
        
        self.assertEqual(output.shape, (batch_size, seq_len, d_v))
        self.assertEqual(weights.shape, (batch_size, seq_len, seq_len))
    
    def test_attention_with_mask(self):
        """测试带掩码的注意力"""
        batch_size = 2
        seq_len = 5
        d_k = 8
        
        attention = ScaledDotProductAttention()
        Q = torch.randn(batch_size, seq_len, d_k)
        K = torch.randn(batch_size, seq_len, d_k)
        V = torch.randn(batch_size, seq_len, d_k)
        
        # 创建因果掩码
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        output, weights = attention(Q, K, V, mask)
        
        # 检查掩码位置的权重是否为0
        weights_np = weights.detach().numpy()
        mask_np = mask.numpy()
        
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(seq_len):
                    if mask_np[i, j]:
                        self.assertAlmostEqual(weights_np[b, i, j], 0.0, places=5)
    
    def test_attention_weights_sum_to_one(self):
        """测试注意力权重和为1"""
        attention = ScaledDotProductAttention()
        Q = torch.randn(2, 5, 8)
        K = torch.randn(2, 5, 8)
        V = torch.randn(2, 5, 8)
        
        _, weights = attention(Q, K, V)
        
        # 检查每一行的权重和是否为1
        weights_sum = weights.sum(dim=-1)
        np.testing.assert_allclose(weights_sum.detach().numpy(), 1.0, rtol=1e-5)


class TestMultiHeadAttention(unittest.TestCase):
    """测试 Multi-Head Attention"""
    
    def test_mha_forward(self):
        """测试多头注意力前向传播"""
        batch_size = 2
        seq_len = 5
        d_model = 16
        num_heads = 4
        
        mha = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = mha(x, x, x)
        
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))
    
    def test_mha_with_mask(self):
        """测试带掩码的多头注意力"""
        batch_size = 2
        seq_len = 5
        d_model = 16
        num_heads = 4
        
        mha = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 创建因果掩码
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        output = mha(x, x, x, mask)
        
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

    def test_mha_backward(self):
        """测试多头注意力反向传播"""
        mha = MultiHeadAttention(16, 4)
        x = torch.randn(2, 5, 16, requires_grad=True)

        output = mha(x, x, x)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)


class TestPositionwiseFeedForward(unittest.TestCase):
    """测试 Position-wise Feed-Forward Network"""

    def test_ffn_forward(self):
        """测试 FFN 前向传播"""
        batch_size = 2
        seq_len = 5
        d_model = 16
        d_ff = 64

        ffn = PositionwiseFeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output = ffn(x)

        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

    def test_ffn_backward(self):
        """测试 FFN 反向传播"""
        ffn = PositionwiseFeedForward(16, 64)
        x = torch.randn(2, 5, 16, requires_grad=True)

        output = ffn(x)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)


class TestTransformerBlock(unittest.TestCase):
    """测试 Transformer Block"""

    def test_transformer_block_forward(self):
        """测试 Transformer Block 前向传播"""
        batch_size = 2
        seq_len = 5
        d_model = 16
        num_heads = 4
        d_ff = 64

        block = TransformerBlock(d_model, num_heads, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output = block(x)

        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

    def test_transformer_block_with_mask(self):
        """测试带掩码的 Transformer Block"""
        batch_size = 2
        seq_len = 5
        d_model = 16
        num_heads = 4
        d_ff = 64

        block = TransformerBlock(d_model, num_heads, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        # 创建因果掩码
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        output = block(x, mask)

        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

    def test_transformer_block_backward(self):
        """测试 Transformer Block 反向传播"""
        block = TransformerBlock(16, 4, 64)
        x = torch.randn(2, 5, 16, requires_grad=True)

        output = block(x)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)

    def test_stacked_transformer_blocks(self):
        """测试堆叠多个 Transformer Block"""
        batch_size = 2
        seq_len = 5
        d_model = 16
        num_heads = 4
        d_ff = 64
        num_layers = 3

        blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        x = torch.randn(batch_size, seq_len, d_model)

        for block in blocks:
            x = block(x)

        self.assertEqual(x.shape, (batch_size, seq_len, d_model))


if __name__ == '__main__':
    unittest.main()

