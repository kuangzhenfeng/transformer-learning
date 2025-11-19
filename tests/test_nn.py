"""
神经网络模块的单元测试
"""

import unittest
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minitf.tensor import Tensor
from minitf.nn import (
    Linear, ReLU, Sigmoid, Sequential, softmax, cross_entropy_loss, mse_loss,
    scaled_dot_product_attention, MultiHeadAttention,
    LayerNorm, PositionwiseFeedForward, TransformerBlock
)


class TestTensorOperations(unittest.TestCase):
    """测试新增的 Tensor 操作"""
    
    def test_relu(self):
        """测试 ReLU 激活函数"""
        x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        y = x.relu()
        
        # 检查前向传播
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float32)
        self.assertTrue(np.allclose(y.data, expected))
        
        # 检查反向传播
        y.backward()
        expected_grad = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        self.assertTrue(np.allclose(x.grad, expected_grad))
    
    def test_exp(self):
        """测试指数函数"""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        y = x.exp()
        
        # 检查前向传播
        expected = np.exp(np.array([0.0, 1.0, 2.0], dtype=np.float32))
        self.assertTrue(np.allclose(y.data, expected))
        
        # 检查反向传播
        y.backward()
        expected_grad = np.exp(np.array([0.0, 1.0, 2.0], dtype=np.float32))
        self.assertTrue(np.allclose(x.grad, expected_grad))
    
    def test_matmul(self):
        """测试矩阵乘法"""
        # 测试 2D 矩阵乘法
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        c = a.matmul(b)
        
        # 检查前向传播
        expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32)
        self.assertTrue(np.allclose(c.data, expected))
        
        # 检查反向传播
        c.backward()
        # da = dc @ b^T
        expected_grad_a = np.array([[11.0, 15.0], [11.0, 15.0]], dtype=np.float32)
        # db = a^T @ dc
        expected_grad_b = np.array([[4.0, 4.0], [6.0, 6.0]], dtype=np.float32)
        self.assertTrue(np.allclose(a.grad, expected_grad_a))
        self.assertTrue(np.allclose(b.grad, expected_grad_b))


class TestLinear(unittest.TestCase):
    """测试 Linear 层"""
    
    def test_linear_forward(self):
        """测试 Linear 层的前向传播"""
        # 创建一个简单的线性层
        layer = Linear(3, 2, bias=True)
        
        # 手动设置权重和偏置以便测试
        layer.weight = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
        layer.bias = Tensor([0.1, 0.2], requires_grad=True)
        
        # 输入
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=False)
        
        # 前向传播
        y = layer(x)
        
        # 期望输出: [1, 2, 3] @ [[1, 2], [3, 4], [5, 6]] + [0.1, 0.2]
        #         = [1*1+2*3+3*5, 1*2+2*4+3*6] + [0.1, 0.2]
        #         = [22, 28] + [0.1, 0.2] = [22.1, 28.2]
        expected = np.array([[22.1, 28.2]], dtype=np.float32)
        self.assertTrue(np.allclose(y.data, expected))
    
    def test_linear_backward(self):
        """测试 Linear 层的反向传播"""
        layer = Linear(2, 2, bias=True)
        layer.weight = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        layer.bias = Tensor([0.1, 0.2], requires_grad=True)
        
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        y = layer(x)
        loss = y.sum()
        
        loss.backward()
        
        # 检查梯度是否被计算
        self.assertIsNotNone(layer.weight.grad)
        self.assertIsNotNone(layer.bias.grad)
        self.assertTrue(np.any(layer.weight.grad != 0))
        self.assertTrue(np.any(layer.bias.grad != 0))


class TestActivations(unittest.TestCase):
    """测试激活函数"""
    
    def test_relu_layer(self):
        """测试 ReLU 层"""
        relu = ReLU()
        x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
        y = relu(x)
        
        expected = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.assertTrue(np.allclose(y.data, expected))
    
    def test_sigmoid_layer(self):
        """测试 Sigmoid 层"""
        sigmoid = Sigmoid()
        x = Tensor([0.0], requires_grad=True)
        y = sigmoid(x)
        
        # sigmoid(0) = 0.5
        self.assertAlmostEqual(y.data[0], 0.5, places=5)


class TestSequential(unittest.TestCase):
    """测试 Sequential 容器"""
    
    def test_sequential_forward(self):
        """测试 Sequential 的前向传播"""
        model = Sequential(
            Linear(3, 4, bias=True),
            ReLU(),
            Linear(4, 2, bias=True)
        )
        
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=False)
        y = model(x)
        
        # 检查输出形状
        self.assertEqual(y.data.shape, (1, 2))
    
    def test_sequential_parameters(self):
        """测试 Sequential 的参数收集"""
        model = Sequential(
            Linear(3, 4, bias=True),
            ReLU(),
            Linear(4, 2, bias=True)
        )
        
        params = model.parameters()
        # 应该有 4 个参数: 2 个权重矩阵 + 2 个偏置向量
        self.assertEqual(len(params), 4)


class TestLossFunctions(unittest.TestCase):
    """测试损失函数"""

    def test_softmax(self):
        """测试 softmax 函数"""
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        y = softmax(x, axis=1)

        # 检查 softmax 的和为 1
        self.assertAlmostEqual(np.sum(y.data), 1.0, places=5)

        # 检查所有值都在 [0, 1] 之间
        self.assertTrue(np.all(y.data >= 0))
        self.assertTrue(np.all(y.data <= 1))

    def test_cross_entropy_loss(self):
        """测试交叉熵损失"""
        # 创建一个简单的例子
        logits = Tensor([[2.0, 1.0, 0.1], [0.1, 1.0, 2.0]], requires_grad=True)
        targets = np.array([0, 2])  # 第一个样本的目标是类别 0，第二个是类别 2

        loss = cross_entropy_loss(logits, targets)

        # 损失应该是一个标量
        self.assertEqual(loss.data.shape, ())

        # 损失应该是正数
        self.assertGreater(loss.data, 0)

        # 测试反向传播
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(np.any(logits.grad != 0))

    def test_mse_loss(self):
        """测试均方误差损失"""
        y_pred = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y_true = Tensor([[1.5, 2.5], [3.5, 4.5]], requires_grad=False)

        loss = mse_loss(y_pred, y_true)

        # 手动计算期望的 MSE
        # diff = [[0.5, 0.5], [0.5, 0.5]]
        # squared_diff = [[0.25, 0.25], [0.25, 0.25]]
        # mean = 0.25
        expected = 0.25
        self.assertAlmostEqual(loss.data, expected, places=5)

        # 测试反向传播
        loss.backward()
        self.assertIsNotNone(y_pred.grad)
        self.assertTrue(np.any(y_pred.grad != 0))


class TestEndToEnd(unittest.TestCase):
    """端到端测试"""

    def test_mlp_training_step(self):
        """测试 MLP 的一个训练步骤"""
        # 创建一个简单的 MLP
        model = Sequential(
            Linear(2, 4, bias=True),
            ReLU(),
            Linear(4, 3, bias=True)
        )

        # 创建一些假数据
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
        targets = np.array([0, 1])

        # 前向传播
        logits = model(x)

        # 计算损失
        loss = cross_entropy_loss(logits, targets)

        # 反向传播
        loss.backward()

        # 检查所有参数都有梯度
        for param in model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(np.any(param.grad != 0))

        # 模拟一步梯度下降
        learning_rate = 0.01
        for param in model.parameters():
            param.data -= learning_rate * param.grad

        # 清零梯度
        model.zero_grad()

        # 检查梯度已被清零
        for param in model.parameters():
            self.assertTrue(np.all(param.grad == 0))


class TestNewTensorOperations(unittest.TestCase):
    """测试新增的 Tensor 操作（用于 Attention）"""

    def test_transpose(self):
        """测试转置操作"""
        # 测试 2D 转置
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        y = x.transpose()

        # 检查前向传播
        expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float32)
        self.assertTrue(np.allclose(y.data, expected))

        # 检查反向传播
        y.backward()
        expected_grad = np.ones((2, 3), dtype=np.float32)
        self.assertTrue(np.allclose(x.grad, expected_grad))

    def test_transpose_3d(self):
        """测试 3D 转置操作"""
        x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
        y = x.transpose((0, 2, 1))  # 交换最后两个维度

        # 检查形状
        self.assertEqual(y.shape, (2, 4, 3))

        # 检查反向传播
        y.backward()
        self.assertEqual(x.grad.shape, x.shape)

    def test_reshape(self):
        """测试 reshape 操作"""
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        y = x.reshape(3, 2)

        # 检查前向传播
        expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        self.assertTrue(np.allclose(y.data, expected))

        # 检查反向传播
        y.backward()
        expected_grad = np.ones((2, 3), dtype=np.float32)
        self.assertTrue(np.allclose(x.grad, expected_grad))

    def test_sqrt(self):
        """测试平方根操作"""
        x = Tensor([1.0, 4.0, 9.0, 16.0], requires_grad=True)
        y = x.sqrt()

        # 检查前向传播
        expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        self.assertTrue(np.allclose(y.data, expected))

        # 检查反向传播
        y.backward()
        expected_grad = np.array([0.5, 0.25, 1/6, 0.125], dtype=np.float32)
        self.assertTrue(np.allclose(x.grad, expected_grad))

    def test_shape_property(self):
        """测试 shape 属性"""
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(x.shape, (2, 3))


class TestAttention(unittest.TestCase):
    """测试 Attention 机制"""

    def test_scaled_dot_product_attention_basic(self):
        """测试基本的 Scaled Dot-Product Attention"""
        batch_size = 2
        seq_len = 3
        d_k = 4
        d_v = 5

        # 创建随机的 Q, K, V
        np.random.seed(42)
        Q = Tensor(np.random.randn(batch_size, seq_len, d_k), requires_grad=True)
        K = Tensor(np.random.randn(batch_size, seq_len, d_k), requires_grad=True)
        V = Tensor(np.random.randn(batch_size, seq_len, d_v), requires_grad=True)

        # 计算注意力
        output, weights = scaled_dot_product_attention(Q, K, V)

        # 检查输出形状
        self.assertEqual(output.shape, (batch_size, seq_len, d_v))
        self.assertEqual(weights.shape, (batch_size, seq_len, seq_len))

        # 检查注意力权重是否归一化（每行和为1）
        weights_sum = np.sum(weights.data, axis=-1)
        expected_sum = np.ones((batch_size, seq_len), dtype=np.float32)
        self.assertTrue(np.allclose(weights_sum, expected_sum, atol=1e-6))

        # 检查反向传播
        loss = output.sum()
        loss.backward()

        # 确保梯度已计算
        self.assertTrue(Q.grad is not None)
        self.assertTrue(K.grad is not None)
        self.assertTrue(V.grad is not None)
        self.assertEqual(Q.grad.shape, Q.shape)
        self.assertEqual(K.grad.shape, K.shape)
        self.assertEqual(V.grad.shape, V.shape)

    def test_scaled_dot_product_attention_with_mask(self):
        """测试带掩码的 Scaled Dot-Product Attention"""
        batch_size = 1
        seq_len = 4
        d_k = 8

        # 创建 Q, K, V
        np.random.seed(42)
        Q = Tensor(np.random.randn(batch_size, seq_len, d_k), requires_grad=True)
        K = Tensor(np.random.randn(batch_size, seq_len, d_k), requires_grad=True)
        V = Tensor(np.random.randn(batch_size, seq_len, d_k), requires_grad=True)

        # 创建因果掩码（上三角掩码）
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        mask = np.broadcast_to(mask, (batch_size, seq_len, seq_len))

        # 计算注意力
        output, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

        # 检查掩码位置的权重接近0
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                self.assertTrue(weights.data[0, i, j] < 1e-3)

    def test_multi_head_attention_basic(self):
        """测试基本的 Multi-Head Attention"""
        batch_size = 2
        seq_len = 5
        d_model = 16
        num_heads = 4

        # 创建 Multi-Head Attention 层
        mha = MultiHeadAttention(d_model, num_heads)

        # 创建输入
        np.random.seed(42)
        Q = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
        K = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
        V = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)

        # 前向传播
        output = mha(Q, K, V)

        # 检查输出形状
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

        # 检查反向传播
        loss = output.sum()
        loss.backward()

        # 确保梯度已计算
        self.assertTrue(Q.grad is not None)
        self.assertTrue(K.grad is not None)
        self.assertTrue(V.grad is not None)

        # 检查参数梯度
        for param in mha.parameters():
            self.assertTrue(param.grad is not None)
            self.assertEqual(param.grad.shape, param.shape)

    def test_multi_head_attention_self_attention(self):
        """测试 Self-Attention（Q=K=V）"""
        batch_size = 2
        seq_len = 4
        d_model = 12
        num_heads = 3

        # 创建 Multi-Head Attention 层
        mha = MultiHeadAttention(d_model, num_heads)

        # 创建输入（Self-Attention）
        np.random.seed(42)
        X = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)

        # 前向传播
        output = mha(X, X, X)

        # 检查输出形状
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

        # 检查反向传播
        loss = output.sum()
        loss.backward()

        # 确保梯度已计算
        self.assertTrue(X.grad is not None)
        self.assertEqual(X.grad.shape, X.shape)


class TestTensorMeanVar(unittest.TestCase):
    """测试 Tensor 的 mean 和 var 操作"""

    def test_mean_all(self):
        """测试对所有元素求均值"""
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        y = x.mean()

        # 检查前向传播
        expected = 3.5  # (1+2+3+4+5+6) / 6
        self.assertTrue(np.allclose(y.data, expected))

        # 检查反向传播
        y.backward()
        expected_grad = np.ones((2, 3), dtype=np.float32) / 6
        self.assertTrue(np.allclose(x.grad, expected_grad))

    def test_mean_axis(self):
        """测试沿指定轴求均值"""
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        print(x.shape)
        y = x.mean(axis=1, keepdims=True)

        # 检查前向传播
        expected = np.array([[2.0], [5.0]], dtype=np.float32)
        self.assertTrue(np.allclose(y.data, expected))

        # 检查反向传播
        y.backward()
        expected_grad = np.ones((2, 3), dtype=np.float32) / 3
        self.assertTrue(np.allclose(x.grad, expected_grad))

    def test_var(self):
        """测试方差计算"""
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        y = x.var(axis=1, keepdims=True)

        # 检查前向传播
        # 第一行: var([1, 2, 3]) = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
        # 第二行: var([4, 5, 6]) = ((4-5)^2 + (5-5)^2 + (6-5)^2) / 3 = 2/3
        expected = np.array([[2.0/3.0], [2.0/3.0]], dtype=np.float32)
        self.assertTrue(np.allclose(y.data, expected, atol=1e-5))


class TestLayerNorm(unittest.TestCase):
    """测试 LayerNorm 层"""

    def test_layernorm_forward(self):
        """测试 LayerNorm 的前向传播"""
        d_model = 4
        layer_norm = LayerNorm(d_model)

        # 手动设置 gamma 和 beta
        layer_norm.gamma = Tensor(np.ones(d_model, dtype=np.float32), requires_grad=True)
        layer_norm.beta = Tensor(np.zeros(d_model, dtype=np.float32), requires_grad=True)

        # 输入
        x = Tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], requires_grad=True)

        # 前向传播
        y = layer_norm(x)

        # 检查输出形状
        self.assertEqual(y.shape, x.shape)

        # 检查归一化后的均值接近0，方差接近1
        for i in range(x.shape[0]):
            mean = np.mean(y.data[i])
            var = np.var(y.data[i])
            self.assertTrue(np.abs(mean) < 1e-5, f"Mean should be close to 0, got {mean}")
            self.assertTrue(np.abs(var - 1.0) < 1e-5, f"Var should be close to 1, got {var}")

    def test_layernorm_backward(self):
        """测试 LayerNorm 的反向传播"""
        d_model = 4
        layer_norm = LayerNorm(d_model)

        x = Tensor(np.random.randn(2, 3, d_model), requires_grad=True)

        # 前向传播
        y = layer_norm(x)

        # 反向传播
        loss = y.sum()
        loss.backward()

        # 确保梯度已计算
        self.assertTrue(x.grad is not None)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertTrue(layer_norm.gamma.grad is not None)
        self.assertTrue(layer_norm.beta.grad is not None)

    def test_layernorm_parameters(self):
        """测试 LayerNorm 的参数"""
        d_model = 4
        layer_norm = LayerNorm(d_model)

        params = layer_norm.parameters()
        self.assertEqual(len(params), 2)  # gamma 和 beta
        self.assertEqual(params[0].shape, (d_model,))
        self.assertEqual(params[1].shape, (d_model,))


class TestPositionwiseFeedForward(unittest.TestCase):
    """测试 PositionwiseFeedForward 层"""

    def test_ffn_forward(self):
        """测试 FFN 的前向传播"""
        d_model = 8
        d_ff = 32
        ffn = PositionwiseFeedForward(d_model, d_ff)

        # 输入
        batch_size, seq_len = 2, 3
        x = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)

        # 前向传播
        y = ffn(x)

        # 检查输出形状
        self.assertEqual(y.shape, (batch_size, seq_len, d_model))

    def test_ffn_backward(self):
        """测试 FFN 的反向传播"""
        d_model = 8
        d_ff = 32
        ffn = PositionwiseFeedForward(d_model, d_ff)

        x = Tensor(np.random.randn(2, 3, d_model), requires_grad=True)

        # 前向传播
        y = ffn(x)

        # 反向传播
        loss = y.sum()
        loss.backward()

        # 确保梯度已计算
        self.assertTrue(x.grad is not None)
        self.assertEqual(x.grad.shape, x.shape)

    def test_ffn_parameters(self):
        """测试 FFN 的参数"""
        d_model = 8
        d_ff = 32
        ffn = PositionwiseFeedForward(d_model, d_ff)

        params = ffn.parameters()
        # 两个 Linear 层，每个有 weight 和 bias
        self.assertEqual(len(params), 4)


class TestTransformerBlock(unittest.TestCase):
    """测试 TransformerBlock"""

    def test_transformer_block_forward(self):
        """测试 Transformer Block 的前向传播"""
        d_model = 16
        num_heads = 4
        d_ff = 64

        block = TransformerBlock(d_model, num_heads, d_ff)

        # 输入
        batch_size, seq_len = 2, 5
        x = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)

        # 前向传播
        y = block(x)

        # 检查输出形状
        self.assertEqual(y.shape, (batch_size, seq_len, d_model))

    def test_transformer_block_backward(self):
        """测试 Transformer Block 的反向传播"""
        d_model = 16
        num_heads = 4
        d_ff = 64

        block = TransformerBlock(d_model, num_heads, d_ff)

        x = Tensor(np.random.randn(2, 5, d_model), requires_grad=True)

        # 前向传播
        y = block(x)

        # 反向传播
        loss = y.sum()
        loss.backward()

        # 确保梯度已计算
        self.assertTrue(x.grad is not None)
        self.assertEqual(x.grad.shape, x.shape)

    def test_transformer_block_with_mask(self):
        """测试带掩码的 Transformer Block"""
        d_model = 16
        num_heads = 4
        d_ff = 64

        block = TransformerBlock(d_model, num_heads, d_ff)

        # 输入
        batch_size, seq_len = 2, 5
        x = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)

        # 创建因果掩码（上三角掩码）
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)

        # 前向传播
        y = block(x, mask=mask)

        # 检查输出形状
        self.assertEqual(y.shape, (batch_size, seq_len, d_model))

    def test_transformer_block_parameters(self):
        """测试 Transformer Block 的参数"""
        d_model = 16
        num_heads = 4
        d_ff = 64

        block = TransformerBlock(d_model, num_heads, d_ff)

        params = block.parameters()
        # MultiHeadAttention: 4个Linear层 (W_q, W_k, W_v, W_o)，每个没有bias，所以4个参数
        # PositionwiseFeedForward: 2个Linear层，每个有weight和bias，所以4个参数
        # 2个LayerNorm: 每个有gamma和beta，所以4个参数
        # 总共: 4 + 4 + 4 = 12 个参数
        self.assertEqual(len(params), 12)


if __name__ == '__main__':
    unittest.main()
