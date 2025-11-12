"""
使用示例文件，展示如何使用 Tensor 类进行自动微分计算
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minitf.tensor import Tensor

def main():
    print("AutoGrad Tensor Library 使用示例")
    print("=" * 40)
    
    # 创建带梯度的张量
    print("1. 创建张量:")
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    print(f"   a = {a}")
    print(f"   b = {b}")
    
    # 基本运算
    print("\n2. 基本运算:")
    c = a * b
    print(f"   c = a * b = {c}")
    
    d = c + a.sin()
    print(f"   d = c + sin(a) = {d}")
    
    e = d / b.log()
    print(f"   e = d / log(b) = {e}")
    
    # 反向传播
    print("\n3. 执行反向传播:")
    e.backward()
    print(f"   e.data = {e.data}")
    print(f"   a.grad = {a.grad}")
    print(f"   b.grad = {b.grad}")
    
    # 更复杂的示例
    print("\n4. 复杂示例 - 多变量函数:")
    x = Tensor(1.0, requires_grad=True)
    y = Tensor(2.0, requires_grad=True)
    z = Tensor(3.0, requires_grad=True)
    
    # 计算 f(x,y,z) = x*y + y*z + z*x
    f = x*y + y*z + z*x
    print(f"   f(x,y,z) = x*y + y*z + z*x = {f}")
    
    # 反向传播
    f.backward()
    print(f"   f.data = {f.data}")
    print(f"   x.grad = {x.grad}")
    print(f"   y.grad = {y.grad}")
    print(f"   z.grad = {z.grad}")
    
    print("\n示例完成!")

if __name__ == "__main__":
    main()