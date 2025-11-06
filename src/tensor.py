import numpy as np

class Tensor:
    """支持自动微分的张量类"""
    
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.requires_grad = requires_grad or any(child.requires_grad for child in _children)
        
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op='+')
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other), _op='*')
        
        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, exponent):
        if isinstance(exponent, Tensor):
            exponent = exponent.data
        out = Tensor(self.data ** exponent, _children=(self,), _op=f'**{exponent}')
        
        def _backward():
            if self.requires_grad:
                self.grad += (exponent * self.data**(exponent-1)) * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, _children=(self, other), _op='/')
        
        def _backward():
            if self.requires_grad:
                self.grad += (1 / other.data) * out.grad
            if other.requires_grad:
                other.grad += (-self.data / (other.data ** 2)) * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        x = self.data
        s = 1 / (1 + np.exp(-x))
        out = Tensor(s, _children=(self,), _op='sigmoid')
        
        def _backward():
            if self.requires_grad:
                self.grad += (s * (1 - s)) * out.grad
        out._backward = _backward
        return out
    
    def sin(self):
        out = Tensor(np.sin(self.data), _children=(self,), _op='sin')
        
        def _backward():
            if self.requires_grad:
                self.grad += np.cos(self.data) * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        out = Tensor(np.log(self.data), _children=(self,), _op='log')
        
        def _backward():
            if self.requires_grad:
                self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out
    
    def sum(self):
        out = Tensor(np.sum(self.data), _children=(self,), _op='sum')
        
        def _backward():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out
    
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