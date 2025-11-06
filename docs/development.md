# 开发指南

## 项目结构

```
.
├── src/                    # 源代码目录
│   ├── __init__.py         # 包初始化文件
│   └── tensor.py           # 核心张量实现
├── tests/                  # 测试代码目录
│   └── test_tensor.py      # 张量单元测试
├── docs/                   # 文档目录
│   └── development.md      # 开发文档
├── examples/               # 示例代码目录
│   └── tensor_example.py   # 张量使用示例
├── requirements.txt        # 依赖包列表
├── README.md               # 项目说明文件
└── .venv/                  # 虚拟环境目录
```

## 开发环境设置

1. 创建虚拟环境:
   ```bash
   python -m venv .venv
   ```

2. 激活虚拟环境:
   ```bash
   source .venv/bin/activate
   ```

3. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```

## 代码规范

### 自动微分实现规范

1. 实现自动微分的张量类应重载基本数学运算符，包括除法（`__truediv__`），确保前向传播和梯度反向传播的正确性。

2. 自定义张量类需要实现常见的数学函数（如sin、log等）的方法，支持正向计算和反向传播。每个函数需包含前向计算和对应的导数计算逻辑，并正确记录操作类型。应通过对象方法调用（如`tensor.sin()`）而非`np.sin(tensor)`的方式使用。

3. 所有数学运算都必须通过支持反向传播的自定义方法执行，禁止依赖外部库（如NumPy）的函数，以保证计算图的完整性。

## 运行测试

```bash
# 运行所有测试
python -m unittest discover tests/

# 运行特定测试文件
python -m unittest tests.test_tensor

# 运行时显示详细信息
python -m unittest tests.test_tensor -v
```

## 添加新功能

1. 在 `src/` 目录下创建新的模块文件
2. 在 `tests/` 目录下添加相应的测试用例
3. 运行测试确保新功能正常工作且不破坏现有功能
4. 在 [examples](file:///home/steven/ai/transformer-learning/examples) 目录下添加示例代码展示新功能
5. 更新文档

## 实现新数学函数的步骤

1. 在 `Tensor` 类中添加新方法
2. 实现前向计算逻辑
3. 实现反向传播逻辑（`_backward` 函数）
4. 记录操作类型（`_op`）
5. 添加相应的测试用例
6. 在示例文件中展示使用方法

## 后续阶段开发指南

随着项目进展，将按阶段添加更多组件：

- 阶段2：添加注意力机制相关模块到 `src/` 目录
- 阶段3：集成 PyTorch 实现，创建对应的 PyTorch 版本模块