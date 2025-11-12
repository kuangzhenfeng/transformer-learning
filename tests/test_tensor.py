import unittest
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minitf.tensor import Tensor

class TestTensor(unittest.TestCase):
    
    def test_initialization(self):
        """测试 Tensor 初始化"""
        # 测试不带梯度的基本初始化
        t1 = Tensor([1, 2, 3])
        self.assertTrue(np.array_equal(t1.data, np.array([1, 2, 3], dtype=np.float32)))
        self.assertTrue(np.array_equal(t1.grad, np.zeros(3)))
        self.assertFalse(t1.requires_grad)
        
        # 测试带梯度的初始化
        t2 = Tensor([1, 2, 3], requires_grad=True)
        self.assertTrue(t2.requires_grad)
        
    def test_addition(self):
        """测试加法运算"""
        a = Tensor(2.0, requires_grad=True)
        b = Tensor(3.0, requires_grad=True)
        c = a + b
        
        self.assertEqual(c.data, 5.0)
        self.assertEqual(c._op, '+')
        
        # 测试反向传播
        c.backward()
        self.assertEqual(a.grad, 1.0)
        self.assertEqual(b.grad, 1.0)
        
    def test_multiplication(self):
        """测试乘法运算"""
        a = Tensor(2.0, requires_grad=True)
        b = Tensor(3.0, requires_grad=True)
        c = a * b
        
        self.assertEqual(c.data, 6.0)
        self.assertEqual(c._op, '*')
        
        # 测试反向传播
        c.backward()
        self.assertEqual(a.grad, 3.0)
        self.assertEqual(b.grad, 2.0)
        
    def test_power(self):
        """测试幂运算"""
        a = Tensor(2.0, requires_grad=True)
        b = a ** 3
        
        self.assertEqual(b.data, 8.0)
        self.assertEqual(b._op, '**3')
        
        # 测试反向传播
        b.backward()
        self.assertEqual(a.grad, 12.0)  # 3 * 2^2 = 12
        
    def test_division(self):
        """测试除法运算"""
        a = Tensor(6.0, requires_grad=True)
        b = Tensor(2.0, requires_grad=True)
        c = a / b
        
        self.assertEqual(c.data, 3.0)
        self.assertEqual(c._op, '/')
        
        # 测试反向传播
        c.backward()
        self.assertEqual(a.grad, 0.5)     # 1/b = 1/2 = 0.5
        self.assertEqual(b.grad, -1.5)    # -a/(b^2) = -6/4 = -1.5
        
    def test_sigmoid(self):
        """测试 sigmoid 函数"""
        a = Tensor(0.0, requires_grad=True)
        b = a.sigmoid()
        
        expected = 1 / (1 + np.exp(0))  # 应该是 0.5
        self.assertAlmostEqual(b.data, expected)
        self.assertEqual(b._op, 'sigmoid')
        
        # 测试反向传播
        b.backward()
        # sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        self.assertAlmostEqual(a.grad, 0.25)
        
    def test_sin(self):
        """测试 sin 函数"""
        a = Tensor(0.0, requires_grad=True)
        b = a.sin()
        
        self.assertEqual(b.data, 0.0)  # sin(0) = 0
        self.assertEqual(b._op, 'sin')
        
        # 测试反向传播
        b.backward()
        self.assertEqual(a.grad, 1.0)  # cos(0) = 1
        
    def test_log(self):
        """测试 log 函数"""
        a = Tensor(np.e, requires_grad=True)
        b = a.log()
        
        self.assertAlmostEqual(b.data, 1.0)  # ln(e) = 1
        self.assertEqual(b._op, 'log')
        
        # 测试反向传播
        b.backward()
        self.assertAlmostEqual(a.grad, 1/np.e)  # 1/e
        
    def test_sum(self):
        """测试 sum 函数"""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = a.sum()
        
        self.assertEqual(b.data, 6.0)
        self.assertEqual(b._op, 'sum')
        
        # 测试反向传播
        b.backward()
        self.assertTrue(np.array_equal(a.grad, np.array([1.0, 1.0, 1.0])))
        
    def test_complex_expression(self):
        """测试复杂表达式"""
        # 测试 f(a, b) = (a * b + sin(a)) / log(b)
        a = Tensor(2.0, requires_grad=True)
        b = Tensor(3.0, requires_grad=True)
        
        c = a * b          # 2 * 3 = 6
        d = c + a.sin()    # 6 + sin(2) ≈ 6 + 0.909 = 6.909
        e = d / b.log()    # 6.909 / log(3) ≈ 6.909 / 1.099 = 6.287
        
        e.backward()
        
        # 验证数值是否正确
        self.assertAlmostEqual(c.data, 6.0)
        self.assertAlmostEqual(d.data, 6.0 + np.sin(2.0))
        self.assertAlmostEqual(e.data, (6.0 + np.sin(2.0)) / np.log(3.0))
        
    def test_broadcast_operations(self):
        """测试广播操作"""
        a = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        b = Tensor([[4.0], [5.0]], requires_grad=True)
        c = a + b
        
        expected_shape = (2, 3)
        self.assertEqual(c.data.shape, expected_shape)
        self.assertTrue(np.array_equal(c.data, np.array([[5.0, 6.0, 7.0], [6.0, 7.0, 8.0]])))
        
    def test_gradient_accumulation(self):
        """测试梯度累积"""
        a = Tensor(2.0, requires_grad=True)
        b = a * a  # 4
        c = a * 2  # 4
        d = b + c  # 8
        
        d.backward()
        # da = da/da * a + da/da * 2 = 2*a + 2 = 6
        self.assertEqual(a.grad, 6.0)

if __name__ == '__main__':
    unittest.main()