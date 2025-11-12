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
from minitf.nn import Linear, ReLU, Sigmoid, Sequential, softmax, cross_entropy_loss, mse_loss


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
        print(y.data)

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


if __name__ == '__main__':
    unittest.main()


