from typing import Any, Optional, Tuple, Union
import numpy as np


class Tensor:
    """支持自动微分的张量类"""

    def __init__(
        self,
        data: Union[float, int, list, np.ndarray],
        requires_grad: bool = False,
        _children: Tuple['Tensor', ...] = (),
        _op: str = ''
    ) -> None:
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.requires_grad = requires_grad or any(child.requires_grad for child in _children)

    @staticmethod
    def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """
        处理广播后的梯度，将其还原到原始形状

        Args:
            grad: 广播后的梯度
            shape: 原始张量的形状

        Returns:
            还原后的梯度
        """
        # 如果形状已经匹配，直接返回
        if grad.shape == shape:
            return grad

        # 计算需要求和的维度
        ndims_added = len(grad.shape) - len(shape)
        for i in range(ndims_added):
            grad = grad.sum(axis=0)

        # 对于被广播的维度（大小为1的维度），需要沿该维度求和
        for i, (grad_dim, shape_dim) in enumerate(zip(grad.shape, shape)):
            if shape_dim == 1 and grad_dim > 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad

    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op='+')

        def _backward():
            if self.requires_grad:
                self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += Tensor._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other), _op='*')

        def _backward():
            if self.requires_grad:
                self.grad += Tensor._unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += Tensor._unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __pow__(self, exponent: Union['Tensor', float, int]) -> 'Tensor':
        if isinstance(exponent, Tensor):
            exponent = exponent.data
        out = Tensor(self.data ** exponent, _children=(self,), _op=f'**{exponent}')

        def _backward():
            if self.requires_grad:
                self.grad += (exponent * self.data**(exponent-1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, _children=(self, other), _op='/')

        def _backward():
            if self.requires_grad:
                self.grad += Tensor._unbroadcast((1 / other.data) * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += Tensor._unbroadcast((-self.data / (other.data ** 2)) * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def sigmoid(self) -> 'Tensor':
        x = self.data
        s = 1 / (1 + np.exp(-x))
        out = Tensor(s, _children=(self,), _op='sigmoid')

        def _backward():
            if self.requires_grad:
                self.grad += (s * (1 - s)) * out.grad
        out._backward = _backward
        return out

    def sin(self) -> 'Tensor':
        out = Tensor(np.sin(self.data), _children=(self,), _op='sin')

        def _backward():
            if self.requires_grad:
                self.grad += np.cos(self.data) * out.grad
        out._backward = _backward
        return out

    def log(self) -> 'Tensor':
        out = Tensor(np.log(self.data), _children=(self,), _op='log')

        def _backward():
            if self.requires_grad:
                self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out
    
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """
        对张量进行求和

        Args:
            axis: 求和的轴，None表示对所有元素求和
            keepdims: 是否保持维度
        """
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), _children=(self,), _op='sum')

        def _backward():
            if self.requires_grad:
                if axis is None:
                    # 对所有元素求和，梯度广播到所有位置
                    self.grad += np.ones_like(self.data) * out.grad
                else:
                    # 对特定轴求和，需要扩展梯度维度
                    grad_shape = list(self.data.shape)
                    if not keepdims:
                        # 如果没有保持维度，需要在求和的轴上添加维度
                        if isinstance(axis, int):
                            axes = [axis]
                        else:
                            axes = list(axis)
                        for ax in sorted(axes):
                            grad_shape[ax] = 1
                        expanded_grad = out.grad.reshape(grad_shape)
                    else:
                        expanded_grad = out.grad
                    self.grad += np.broadcast_to(expanded_grad, self.data.shape)
        out._backward = _backward
        return out

    def relu(self) -> 'Tensor':
        """ReLU 激活函数: max(0, x)"""
        out = Tensor(np.maximum(0, self.data), _children=(self,), _op='relu')

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0).astype(np.float32) * out.grad
        out._backward = _backward
        return out

    def exp(self) -> 'Tensor':
        """
        指数函数
        x.exp() = e^x
        """
        out = Tensor(np.exp(self.data), _children=(self,), _op='exp')

        def _backward():
            if self.requires_grad:
                self.grad += np.exp(self.data) * out.grad
        out._backward = _backward
        return out

    def max(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """
        求最大值

        Args:
            axis: 求最大值的轴，None表示对所有元素求最大值
            keepdims: 是否保持维度
        """
        out = Tensor(np.max(self.data, axis=axis, keepdims=keepdims), _children=(self,), _op='max')

        def _backward():
            if self.requires_grad:
                if axis is None:
                    # 对所有元素求最大值
                    mask = (self.data == out.data).astype(np.float32)
                    self.grad += mask * out.grad / np.sum(mask)
                else:
                    # 对特定轴求最大值
                    if not keepdims:
                        # 扩展维度以匹配原始形状
                        if isinstance(axis, int):
                            expanded_out = np.expand_dims(out.data, axis=axis)
                            expanded_grad = np.expand_dims(out.grad, axis=axis)
                        else:
                            expanded_out = out.data
                            expanded_grad = out.grad
                            for ax in sorted(axis):
                                expanded_out = np.expand_dims(expanded_out, axis=ax)
                                expanded_grad = np.expand_dims(expanded_grad, axis=ax)
                    else:
                        expanded_out = out.data
                        expanded_grad = out.grad

                    mask = (self.data == expanded_out).astype(np.float32)
                    # 处理多个最大值的情况
                    mask_sum = np.sum(mask, axis=axis, keepdims=True)
                    self.grad += mask * expanded_grad / mask_sum
        out._backward = _backward
        return out

    def matmul(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        """
        矩阵乘法

        Args:
            other: 另一个张量
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.matmul(self.data, other.data), _children=(self, other), _op='@')

        def _backward():
            if self.requires_grad:
                # dL/dA = dL/dC @ B^T
                grad = np.matmul(out.grad, other.data.swapaxes(-2, -1))
                self.grad += Tensor._unbroadcast(grad, self.data.shape)
            if other.requires_grad:
                # dL/dB = A^T @ dL/dC
                grad = np.matmul(self.data.swapaxes(-2, -1), out.grad)
                other.grad += Tensor._unbroadcast(grad, other.data.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        """支持 @ 运算符"""
        return self.matmul(other)

    def transpose(self, axes=None):
        """
        转置张量

        Args:
            axes: 轴的排列顺序，None表示反转所有轴

        Returns:
            转置后的张量
        """
        if axes is None:
            axes = tuple(range(len(self.data.shape) - 1, -1, -1))

        out = Tensor(np.transpose(self.data, axes), _children=(self,), _op='transpose')

        def _backward():
            if self.requires_grad:
                # 转置的梯度是反向转置
                inverse_axes = np.argsort(axes)
                self.grad += np.transpose(out.grad, inverse_axes)
        out._backward = _backward
        return out

    def reshape(self, *shape):
        """
        改变张量形状

        Args:
            *shape: 新的形状

        Returns:
            改变形状后的张量
        """
        # 如果传入的是一个元组，解包它
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]

        out = Tensor(self.data.reshape(shape), _children=(self,), _op='reshape')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def sqrt(self):
        """
        平方根函数
        sqrt(x) = x^0.5
        """
        out = Tensor(np.sqrt(self.data), _children=(self,), _op='sqrt')

        def _backward():
            if self.requires_grad:
                # d(sqrt(x))/dx = 1/(2*sqrt(x))
                self.grad += (0.5 / np.sqrt(self.data)) * out.grad
        out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """
        计算张量的均值

        Args:
            axis: 求均值的轴，None表示对所有元素求均值
            keepdims: 是否保持维度

        Returns:
            均值张量
        """
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), _children=(self,), _op='mean')

        def _backward():
            if self.requires_grad:
                if axis is None:
                    # 对所有元素求均值，梯度均匀分配到所有位置
                    self.grad += np.ones_like(self.data) * out.grad / self.data.size
                else:
                    # 对特定轴求均值
                    grad_shape = list(self.data.shape)
                    if not keepdims:
                        # 如果没有保持维度，需要在求均值的轴上添加维度
                        if isinstance(axis, int):
                            axes = [axis]
                        else:
                            axes = list(axis)
                        for ax in sorted(axes):
                            grad_shape[ax] = 1
                        expanded_grad = out.grad.reshape(grad_shape)
                    else:
                        expanded_grad = out.grad

                    # 计算归一化因子
                    if isinstance(axis, int):
                        n = self.data.shape[axis]
                    else:
                        n = np.prod([self.data.shape[ax] for ax in axis])

                    self.grad += np.broadcast_to(expanded_grad, self.data.shape) / n
        out._backward = _backward
        return out

    def var(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False, unbiased: bool = False) -> 'Tensor':
        """
        计算张量的方差

        Args:
            axis: 求方差的轴，None表示对所有元素求方差
            keepdims: 是否保持维度
            unbiased: 是否使用无偏估计（除以 n-1 而不是 n）

        Returns:
            方差张量
        """
        # 计算均值
        mean_val = self.mean(axis=axis, keepdims=True)

        # 计算 (x - mean)^2
        diff = self + (mean_val * Tensor(-1.0))
        squared_diff = diff ** 2

        # 计算方差
        if unbiased:
            # 无偏估计：除以 n-1
            if axis is None:
                n = self.data.size
            elif isinstance(axis, int):
                n = self.data.shape[axis]
            else:
                n = np.prod([self.data.shape[ax] for ax in axis])
            var_val = squared_diff.sum(axis=axis, keepdims=keepdims) * Tensor(1.0 / (n - 1))
        else:
            # 有偏估计：除以 n
            var_val = squared_diff.mean(axis=axis, keepdims=keepdims)

        return var_val

    @property
    def shape(self):
        """返回张量的形状"""
        return self.data.shape

    def backward(self):
        # 拓扑排序
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # 反向传播
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()
    
    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, requires_grad={self.requires_grad})"

# 测试自动微分
def test_autograd():
    # 测试简单函数: f(a, b) = (a * b + sin(a)) / log(b)
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    
    print("初始张量:")
    print(f"a: {a}")
    print(f"b: {b}")
    
    c = a * b
    d = c + a.sin()
    e = d / b.log()
    
    print("\n计算图节点:")
    print(f"c = a * b: {c}")
    print(f"d = c + sin(a): {d}") 
    print(f"e = d / log(b): {e}")
    
    e.backward()
    
    print(f"\n计算结果:")
    print(f"f(a,b) = {e.data}")
    print(f"df/da = {a.grad}")
    print(f"df/db = {b.grad}")

if __name__ == "__main__":
    test_autograd()