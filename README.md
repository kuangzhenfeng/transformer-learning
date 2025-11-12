# Transformer 学习项目

本项目旨在从零开始实现深度学习 Transformer 架构的关键组件，帮助理解现代深度学习框架的底层原理。

## 项目目标

总目标：不依赖高级框架的"黑箱"，从零开始构建关键组件，最终整合一个微型 GPT，从而获得对深度学习底层原理的深刻直觉。

### 第一阶段：基础夯实与"手感"建立

目标：亲手打造核心工具（自动微分库），并在简单模型上验证，建立对梯度计算和模型训练的直接感觉。

#### 任务 1：实现微型自动微分库 (micrograd)
- 定义 Tensor 类，包含 data（数据）和 grad（梯度）属性
- 实现基础运算（+, *, sin, log, sigmoid等）的重载，并能构建计算图
- 实现反向传播算法，能通过拓扑排序计算任意节点的梯度

#### 任务 2：使用 micrograd 训练线性回归模型
- 使用自制的自动微分库
- 定义线性模型 y_pred = w * x + b，其中 w, b 为需要优化的 Tensor
- 实现均方误差损失函数和梯度下降算法

#### 任务 3：从零实现多层感知机 (MLP)
- 实现 Linear 层（线性变换）
- 实现 Sigmoid 和 ReLU 激活函数
- 实现 Sequential 容器，用于组合多个层
- 实现交叉熵损失函数

#### 任务 4：可视化工具实现
- 在minitf/utils 中实现一套可视化工具，帮助直观理解神经网络训练过程和模型结构，为后续Transformer实现提供调试和分析支持。

### 第二阶段：深入 Transformer 架构核心

目标：将矩阵运算能力用于实现 Transformer 的核心组件，理解其内部数据流动。

#### 任务 1：实现 Scaled Dot-Product Attention 和 Multi-Head Attention
- 实现 softmax 函数（注意数值稳定性）
- 实现单头的 Scaled Dot-Product Attention
- 实现 Multi-Head 机制

#### 任务 2：组装完整的 Transformer 编码器块
- 实现 LayerNorm 层和残差连接
- 实现 Position-wise Feed-Forward Network
- 将以上组件组装成 Transformer Block

### 第三阶段：整合与框架入门

目标：将手写组件与工业级框架（PyTorch）对接，体验高效开发，并最终完成微型 GPT 的构建与训练。

#### 任务 1：学习 PyTorch 基础并重写之前组件
- 学习 PyTorch 的 torch.Tensor, autograd, nn.Module, DataLoader 等核心概念
- 使用 torch.nn 重写之前实现的组件

#### 任务 2：实现与训练一个微型 GPT
- 实现因果自注意力掩码
- 将 Transformer 解码器块堆叠成 GPT 模型
- 实现 tokenizer
- 编写训练循环，在 CPU 或 GPU 上训练模型

## 当前项目结构

```
.
├── minitf/              # 源代码目录
│   ├── __init__.py      # 包初始化文件
│   ├── tensor.py        # 核心张量实现 (阶段1任务1)
│   └── nn.py            # 神经网络模块 (阶段1任务3)
├── tests/               # 测试代码目录
│   ├── test_tensor.py   # 张量单元测试 (阶段1任务1)
│   └── test_nn.py       # 神经网络模块单元测试 (阶段1任务3)
├── docs/                # 文档目录
├── examples/            # 示例代码目录
│   ├── tensor_example.py           # 张量使用示例 (阶段1任务1)
│   ├── linear_regression.py        # 线性回归训练示例 (阶段1任务2)
│   └── mlp_classification.py       # MLP 分类训练示例 (阶段1任务3)
├── requirements.txt     # 依赖包列表
└── README.md            # 项目说明文件
```

## 阶段1完成情况

### 任务1：微型自动微分库 (micrograd) ✅
已完成自动微分张量库的实现：
- 基本数学运算：加法、乘法、幂运算、除法
- 常用数学函数：sigmoid、sin、log
- 自动微分系统
- 计算图构建与反向传播
- 支持广播操作的梯度计算

### 任务2：线性回归模型训练 ✅
已完成使用 micrograd 训练线性回归模型：
- 实现线性模型 y = w * x + b
- 实现均方误差 (MSE) 损失函数
- 实现梯度下降优化算法
- 成功训练并验证模型收敛性

### 任务3：多层感知机 (MLP) ✅
已完成从零实现多层感知机：
- 实现 Linear 层（全连接层）：支持线性变换 y = xW + b
- 实现激活函数：ReLU 和 Sigmoid
- 实现 Sequential 容器：用于组合多个层
- 实现交叉熵损失函数：用于分类任务
- 扩展 Tensor 类：添加 relu、exp、max、matmul 等操作
- 成功训练 MLP 解决 XOR 问题（准确率 100%）
- 成功训练 MLP 解决螺旋形多类分类问题（准确率 98%）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行测试

```bash
python -m unittest tests/test_tensor.py
```

或者

```bash
python -m unittest discover tests/
```

## 使用示例

### 基础自动微分示例

```python
from minitf.tensor import Tensor

# 创建带梯度的张量
a = Tensor(2.0, requires_grad=True)
b = Tensor(3.0, requires_grad=True)

# 构建计算图
c = a * b
d = c + a.sin()
e = d / b.log()

# 执行反向传播
e.backward()

# 获取梯度
print(f"Gradient of a: {a.grad}")
print(f"Gradient of b: {b.grad}")
```

### 线性回归训练示例

```bash
# 运行线性回归训练示例
python examples/linear_regression.py
```

该示例展示了如何：
- 生成训练数据
- 定义线性模型 y = w * x + b
- 使用均方误差损失函数
- 通过梯度下降优化参数
- 评估模型性能

### MLP 分类训练示例

```bash
# 运行 MLP 分类训练示例
python examples/mlp_classification.py
```

该示例展示了如何：
- 构建多层感知机（MLP）模型
- 使用 Sequential 容器组合多个层
- 训练 MLP 解决 XOR 问题（非线性分类）
- 训练 MLP 解决螺旋形多类分类问题
- 使用交叉熵损失函数
- 通过梯度下降优化神经网络参数

示例代码：

```python
from minitf.tensor import Tensor
from minitf.nn import Linear, ReLU, Sequential, cross_entropy_loss

# 创建 MLP 模型
model = Sequential(
    Linear(2, 4, bias=True),   # 输入层到隐藏层
    ReLU(),                     # ReLU 激活
    Linear(4, 2, bias=True)     # 隐藏层到输出层
)

# 训练循环
for epoch in range(epochs):
    # 前向传播
    logits = model(x_tensor)

    # 计算损失
    loss = cross_entropy_loss(logits, targets)

    # 反向传播
    loss.backward()

    # 梯度下降
    for param in model.parameters():
        param.data -= learning_rate * param.grad

    # 清零梯度
    model.zero_grad()
```