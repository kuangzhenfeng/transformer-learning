"""
交互式可视化演示 - 阶段2任务3
融合阶段1和阶段2的所有内容，提供全面的可视化演示

本演示包括：
1. 自动微分计算图可视化
2. 线性回归训练过程可视化
3. MLP 决策边界可视化
4. 注意力机制热力图可视化
5. Transformer Block 完整训练演示
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minitf.tensor import Tensor
from minitf.nn import (
    Linear, ReLU, Sequential,
    scaled_dot_product_attention, MultiHeadAttention,
    LayerNorm, PositionwiseFeedForward, TransformerBlock,
    cross_entropy_loss, softmax
)


def visualize_computation_graph():
    """
    演示1: 可视化自动微分的计算图和梯度反向传播
    """
    print("\n" + "=" * 80)
    print("演示1: 自动微分计算图可视化")
    print("=" * 80)

    # 创建一个简单的计算图
    # z = (x * y + x.sin()) / y.log()
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)

    # 构建计算图
    a = x * y           # a = 2.0 * 3.0 = 6.0
    b = x.sin()         # b = sin(2.0) ≈ 0.909
    c = a + b           # c = 6.0 + 0.909 ≈ 6.909
    d = y.log()         # d = log(3.0) ≈ 1.099
    z = c / d           # z = 6.909 / 1.099 ≈ 6.288

    print("\n计算图结构:")
    print("  x = 2.0")
    print("  y = 3.0")
    print("  a = x * y = {:.4f}".format(a.data))
    print("  b = sin(x) = {:.4f}".format(b.data))
    print("  c = a + b = {:.4f}".format(c.data))
    print("  d = log(y) = {:.4f}".format(d.data))
    print("  z = c / d = {:.4f}".format(z.data))

    # 反向传播
    z.backward()

    print("\n反向传播后的梯度:")
    print("  dz/dx = {:.4f}".format(x.grad))
    print("  dz/dy = {:.4f}".format(y.grad))

    # 数值梯度验证
    epsilon = 1e-5
    x_plus = Tensor(x.data + epsilon, requires_grad=False)
    y_val = Tensor(y.data, requires_grad=False)
    z_plus = ((x_plus * y_val + x_plus.sin()) / y_val.log()).data
    numerical_grad_x = (z_plus - z.data) / epsilon

    print("\n数值梯度验证:")
    print("  数值梯度 dz/dx = {:.4f}".format(numerical_grad_x))
    print("  自动微分 dz/dx = {:.4f}".format(x.grad))
    print("  误差 = {:.6f}".format(abs(numerical_grad_x - x.grad)))

    return x, y, z


def visualize_linear_regression():
    """
    演示2: 线性回归训练过程可视化
    """
    print("\n" + "=" * 80)
    print("演示2: 线性回归训练过程可视化")
    print("=" * 80)

    # 生成数据
    np.random.seed(42)
    n_samples = 50
    X = np.random.randn(n_samples, 1).astype(np.float32) * 2
    true_w, true_b = 3.0, -1.0
    y = X * true_w + true_b + np.random.randn(n_samples, 1).astype(np.float32) * 0.5

    print(f"\n数据集信息:")
    print(f"  样本数量: {n_samples}")
    print(f"  真实参数: w = {true_w:.2f}, b = {true_b:.2f}")

    # 初始化参数
    w = Tensor(np.random.randn(1, 1).astype(np.float32), requires_grad=True)
    b = Tensor(np.zeros((1, 1), dtype=np.float32), requires_grad=True)

    # 训练参数
    learning_rate = 0.01
    epochs = 100

    # 记录训练历史
    history = {
        'w': [w.data[0, 0]],
        'b': [b.data[0, 0]],
        'loss': []
    }

    print(f"\n训练参数:")
    print(f"  学习率: {learning_rate}")
    print(f"  训练轮数: {epochs}")

    print(f"\n开始训练...")
    print("-" * 80)

    for epoch in range(epochs):
        # 前向传播
        X_tensor = Tensor(X, requires_grad=False)
        y_pred = X_tensor.matmul(w) + b

        # 计算损失
        y_tensor = Tensor(y, requires_grad=False)
        diff = y_pred + (y_tensor * Tensor(-1.0))
        loss = (diff ** 2).sum() * Tensor(1.0 / n_samples)

        # 反向传播
        loss.backward()

        # 梯度下降
        w.data -= learning_rate * w.grad
        b.data -= learning_rate * b.grad

        # 清零梯度
        w.grad = np.zeros_like(w.data)
        b.grad = np.zeros_like(b.data)

        # 记录历史
        history['w'].append(w.data[0, 0])
        history['b'].append(b.data[0, 0])
        history['loss'].append(loss.data)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | Loss: {loss.data:.6f} | w: {w.data[0,0]:.4f} | b: {b.data[0,0]:.4f}")

    print("-" * 80)
    print("训练完成!")
    print(f"\n最终参数:")
    print(f"  学习到的参数: w = {w.data[0,0]:.4f}, b = {b.data[0,0]:.4f}")
    print(f"  真实参数:     w = {true_w:.4f}, b = {true_b:.4f}")
    print(f"  最终损失: {history['loss'][-1]:.6f}")

    # 可视化训练过程
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('线性回归训练过程可视化', fontsize=16, fontweight='bold')

    # 1. 数据和拟合线
    ax = axes[0, 0]
    ax.scatter(X, y, alpha=0.6, label='数据点', s=30)
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = x_line * w.data[0, 0] + b.data[0, 0]
    y_true_line = x_line * true_w + true_b
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'学习到的线 (w={w.data[0,0]:.2f}, b={b.data[0,0]:.2f})')
    ax.plot(x_line, y_true_line, 'g--', linewidth=2, label=f'真实的线 (w={true_w:.2f}, b={true_b:.2f})')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('数据和拟合结果', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 损失曲线
    ax = axes[0, 1]
    ax.plot(history['loss'], linewidth=2, color='blue')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('训练损失曲线', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 3. 参数w的变化
    ax = axes[1, 0]
    ax.plot(history['w'], linewidth=2, label='学习到的 w', color='red')
    ax.axhline(y=true_w, color='green', linestyle='--', linewidth=2, label='真实的 w')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('w', fontsize=12)
    ax.set_title('权重 w 的学习过程', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 参数b的变化
    ax = axes[1, 1]
    ax.plot(history['b'], linewidth=2, label='学习到的 b', color='red')
    ax.axhline(y=true_b, color='green', linestyle='--', linewidth=2, label='真实的 b')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('b', fontsize=12)
    ax.set_title('偏置 b 的学习过程', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/linear_regression_visualization.png', dpi=150, bbox_inches='tight')
    print("\n可视化图表已保存到: output/linear_regression_visualization.png")
    plt.show()

    return w, b, history


def visualize_mlp_decision_boundary():
    """
    演示3: MLP 决策边界可视化
    """
    print("\n" + "=" * 80)
    print("演示3: MLP 决策边界可视化 (螺旋形数据)")
    print("=" * 80)

    # 生成螺旋形数据
    def make_spiral_data(n_samples=300, n_classes=3, noise=0.1):
        X = np.zeros((n_samples * n_classes, 2), dtype=np.float32)
        y = np.zeros(n_samples * n_classes, dtype=np.int32)

        for class_idx in range(n_classes):
            idx = range(n_samples * class_idx, n_samples * (class_idx + 1))
            r = np.linspace(0.0, 1.0, n_samples)
            t = np.linspace(class_idx * 4, (class_idx + 1) * 4, n_samples) + np.random.randn(n_samples) * noise
            X[idx] = np.c_[r * np.sin(t), r * np.cos(t)]
            y[idx] = class_idx

        return X, y

    np.random.seed(42)
    X, y = make_spiral_data(n_samples=100, n_classes=3, noise=0.15)

    print(f"\n数据集信息:")
    print(f"  样本数量: {len(X)}")
    print(f"  类别数量: 3")
    print(f"  特征维度: 2")

    # 创建 MLP 模型
    model = Sequential(
        Linear(2, 32, bias=True),
        ReLU(),
        Linear(32, 16, bias=True),
        ReLU(),
        Linear(16, 3, bias=True)
    )

    print(f"\n模型结构:")
    print(model)

    # 训练参数
    learning_rate = 0.5
    epochs = 500

    print(f"\n训练参数:")
    print(f"  学习率: {learning_rate}")
    print(f"  训练轮数: {epochs}")

    # 训练历史
    history = {'loss': [], 'accuracy': []}

    print(f"\n开始训练...")
    print("-" * 80)

    for epoch in range(epochs):
        # 前向传播
        x_tensor = Tensor(X, requires_grad=False)
        logits = model(x_tensor)

        # 计算损失
        loss = cross_entropy_loss(logits, y)

        # 反向传播
        loss.backward()

        # 梯度下降
        for param in model.parameters():
            param.data -= learning_rate * param.grad

        # 清零梯度
        model.zero_grad()

        # 计算准确率
        predictions = np.argmax(logits.data, axis=1)
        accuracy = np.mean(predictions == y)

        # 记录历史
        history['loss'].append(loss.data)
        history['accuracy'].append(accuracy)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | Loss: {loss.data:.6f} | Accuracy: {accuracy:.4f}")

    print("-" * 80)
    print("训练完成!")
    print(f"\n最终准确率: {history['accuracy'][-1]:.4f}")

    # 可视化决策边界
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('MLP 决策边界可视化 (螺旋形数据)', fontsize=16, fontweight='bold')

    # 创建网格
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    # 预测网格点
    grid_tensor = Tensor(grid_points, requires_grad=False)
    grid_logits = model(grid_tensor)
    grid_predictions = np.argmax(grid_logits.data, axis=1)
    grid_predictions = grid_predictions.reshape(xx.shape)

    # 1. 决策边界
    ax = axes[0]
    contour = ax.contourf(xx, yy, grid_predictions, alpha=0.4, cmap='viridis', levels=2)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='viridis', edgecolors='black', linewidth=1)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title('决策边界和数据分布', fontsize=13, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Class')

    # 2. 训练损失
    ax = axes[1]
    ax.plot(history['loss'], linewidth=2, color='blue')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('训练损失曲线', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. 训练准确率
    ax = axes[2]
    ax.plot(history['accuracy'], linewidth=2, color='green')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('训练准确率曲线', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig('output/mlp_decision_boundary.png', dpi=150, bbox_inches='tight')
    print("\n可视化图表已保存到: output/mlp_decision_boundary.png")
    plt.show()

    return model, history


def visualize_attention_weights():
    """
    演示4: 注意力权重可视化
    """
    print("\n" + "=" * 80)
    print("演示4: 注意力权重热力图可视化")
    print("=" * 80)

    # 设置参数
    seq_len = 8
    d_model = 16
    num_heads = 4

    print(f"\n参数设置:")
    print(f"  序列长度: {seq_len}")
    print(f"  模型维度: {d_model}")
    print(f"  注意力头数: {num_heads}")

    # 创建示例输入（模拟一个句子的词向量）
    np.random.seed(42)
    X = Tensor(np.random.randn(1, seq_len, d_model).astype(np.float32), requires_grad=True)

    print(f"\n输入形状: {X.shape}")

    # 1. 无掩码的自注意力
    print("\n计算自注意力（无掩码）...")
    Q = K = V = X
    output_no_mask, weights_no_mask = scaled_dot_product_attention(Q, K, V)

    # 2. 因果掩码的自注意力
    print("计算自注意力（因果掩码）...")
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    mask = np.broadcast_to(mask, (1, seq_len, seq_len))
    output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask=mask)

    # 3. Multi-Head Attention
    print("计算多头注意力...")
    mha = MultiHeadAttention(d_model, num_heads)
    output_mha = mha(X, X, X)

    # 为了可视化，我们需要重新计算每个头的注意力权重
    batch_size = X.shape[0]
    Q_proj = mha.W_q(X)
    K_proj = mha.W_k(X)
    V_proj = mha.W_v(X)

    Q_heads = mha.split_heads(Q_proj, batch_size)
    K_heads = mha.split_heads(K_proj, batch_size)
    V_heads = mha.split_heads(V_proj, batch_size)

    # 计算每个头的注意力权重
    head_weights = []
    for head in range(num_heads):
        Q_h = Tensor(Q_heads.data[:, head:head+1, :, :], requires_grad=False)
        K_h = Tensor(K_heads.data[:, head:head+1, :, :], requires_grad=False)
        V_h = Tensor(V_heads.data[:, head:head+1, :, :], requires_grad=False)
        _, weights_h = scaled_dot_product_attention(Q_h, K_h, V_h)
        head_weights.append(weights_h.data[0, 0])

    print("\n注意力权重计算完成!")

    # 可视化
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
    fig.suptitle('注意力机制可视化', fontsize=16, fontweight='bold')

    # 1. 无掩码的注意力权重
    ax = fig.add_subplot(gs[0, 0:2])
    im = ax.imshow(weights_no_mask.data[0], cmap='viridis', aspect='auto', interpolation='nearest')
    ax.set_xlabel('Key Position', fontsize=11)
    ax.set_ylabel('Query Position', fontsize=11)
    ax.set_title('自注意力权重 (无掩码)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(seq_len):
        for j in range(seq_len):
            text = ax.text(j, i, f'{weights_no_mask.data[0, i, j]:.2f}',
                          ha="center", va="center", color="white", fontsize=8)

    # 2. 因果掩码的注意力权重
    ax = fig.add_subplot(gs[0, 2:4])
    im = ax.imshow(weights_masked.data[0], cmap='viridis', aspect='auto', interpolation='nearest')
    ax.set_xlabel('Key Position', fontsize=11)
    ax.set_ylabel('Query Position', fontsize=11)
    ax.set_title('自注意力权重 (因果掩码)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(seq_len):
        for j in range(seq_len):
            text = ax.text(j, i, f'{weights_masked.data[0, i, j]:.2f}',
                          ha="center", va="center", color="white", fontsize=8)

    # 3. 每个注意力头的权重
    for head_idx in range(num_heads):
        row = 1 + head_idx // 4
        col = head_idx % 4
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(head_weights[head_idx], cmap='viridis', aspect='auto', interpolation='nearest')
        ax.set_xlabel('Key', fontsize=9)
        ax.set_ylabel('Query', fontsize=9)
        ax.set_title(f'Head {head_idx + 1}', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 4. 多头注意力平均权重
    avg_weights = np.mean(head_weights, axis=0)
    ax = fig.add_subplot(gs[2, 1:3])
    im = ax.imshow(avg_weights, cmap='viridis', aspect='auto', interpolation='nearest')
    ax.set_xlabel('Key Position', fontsize=11)
    ax.set_ylabel('Query Position', fontsize=11)
    ax.set_title('多头注意力平均权重', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(seq_len):
        for j in range(seq_len):
            text = ax.text(j, i, f'{avg_weights[i, j]:.2f}',
                          ha="center", va="center", color="white", fontsize=8)

    plt.savefig('output/attention_visualization.png', dpi=150, bbox_inches='tight')
    print("\n可视化图表已保存到: output/attention_visualization.png")
    plt.show()

    return weights_no_mask, weights_masked, head_weights


def visualize_transformer_training():
    """
    演示5: Transformer Block 完整训练可视化
    训练一个简单的序列分类任务
    """
    print("\n" + "=" * 80)
    print("演示5: Transformer Block 序列分类训练")
    print("=" * 80)

    # 任务：判断序列的平均值是正数还是负数
    np.random.seed(42)
    n_samples = 200
    seq_len = 6
    d_model = 16
    num_heads = 4
    d_ff = 64

    print(f"\n任务描述:")
    print(f"  根据输入序列的数值特征进行二分类")
    print(f"  类别0: 序列平均值为负")
    print(f"  类别1: 序列平均值为正")

    # 生成数据
    X_data = []
    y_data = []

    for _ in range(n_samples):
        # 随机生成序列
        seq = np.random.randn(seq_len, d_model).astype(np.float32)
        mean_val = np.mean(seq)

        X_data.append(seq)
        y_data.append(1 if mean_val > 0 else 0)

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    print(f"\n数据集信息:")
    print(f"  样本数量: {n_samples}")
    print(f"  序列长度: {seq_len}")
    print(f"  特征维度: {d_model}")
    print(f"  类别分布: 类别0={np.sum(y_data==0)}, 类别1={np.sum(y_data==1)}")

    # 构建模型
    # Transformer Block + 全局平均池化 + 线性分类器
    transformer_block = TransformerBlock(d_model, num_heads, d_ff)
    classifier = Linear(d_model, 2, bias=True)

    print(f"\n模型结构:")
    print(f"  Transformer Block: d_model={d_model}, num_heads={num_heads}, d_ff={d_ff}")
    print(f"  全局平均池化: (batch, seq_len, d_model) -> (batch, d_model)")
    print(f"  分类器: Linear({d_model}, 2)")

    total_params = sum(p.data.size for p in transformer_block.parameters()) + sum(p.data.size for p in classifier.parameters())
    print(f"  总参数量: {total_params}")

    # 训练参数
    learning_rate = 0.01
    epochs = 150
    batch_size = 20

    print(f"\n训练参数:")
    print(f"  学习率: {learning_rate}")
    print(f"  训练轮数: {epochs}")
    print(f"  批次大小: {batch_size}")

    # 训练历史
    history = {'loss': [], 'accuracy': []}

    print(f"\n开始训练...")
    print("-" * 80)

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0

        # 随机打乱数据
        indices = np.random.permutation(n_samples)

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_data[batch_indices]
            y_batch = y_data[batch_indices]

            # 前向传播
            x_tensor = Tensor(X_batch, requires_grad=False)

            # Transformer Block
            transformer_output = transformer_block(x_tensor)

            # 全局平均池化
            pooled = transformer_output.mean(axis=1)

            # 分类器
            logits = classifier(pooled)

            # 计算损失
            loss = cross_entropy_loss(logits, y_batch)

            # 反向传播
            loss.backward()

            # 梯度下降
            all_params = transformer_block.parameters() + classifier.parameters()
            for param in all_params:
                param.data -= learning_rate * param.grad

            # 清零梯度
            transformer_block.zero_grad()
            classifier.zero_grad()

            # 统计
            epoch_loss += loss.data * len(y_batch)
            predictions = np.argmax(logits.data, axis=1)
            epoch_correct += np.sum(predictions == y_batch)

        # 记录历史
        avg_loss = epoch_loss / n_samples
        accuracy = epoch_correct / n_samples
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)

        if (epoch + 1) % 30 == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | Loss: {avg_loss:.6f} | Accuracy: {accuracy:.4f}")

    print("-" * 80)
    print("训练完成!")
    print(f"\n最终准确率: {history['accuracy'][-1]:.4f}")

    # 可视化训练过程
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Transformer Block 训练过程', fontsize=16, fontweight='bold')

    # 1. 训练损失
    ax = axes[0]
    ax.plot(history['loss'], linewidth=2, color='blue')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('训练损失曲线', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 2. 训练准确率
    ax = axes[1]
    ax.plot(history['accuracy'], linewidth=2, color='green')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='随机猜测')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('训练准确率曲线', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig('output/transformer_training.png', dpi=150, bbox_inches='tight')
    print("\n可视化图表已保存到: output/transformer_training.png")
    plt.show()

    return transformer_block, classifier, history


def comprehensive_demo(interactive=True):
    """
    综合演示：展示从基础到高级的完整流程

    Args:
        interactive: 是否使用交互模式（需要用户按 Enter 继续）
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "Transformer 学习项目 - 综合可视化演示")
    print(" " * 30 + "阶段2任务3")
    print("=" * 80)
    print("\n本演示将展示以下内容:")
    print("  1. 自动微分计算图和梯度反向传播")
    print("  2. 线性回归训练过程和参数优化")
    print("  3. MLP 决策边界和分类能力")
    print("  4. 注意力机制和权重分布")
    print("  5. Transformer Block 完整训练流程")

    if interactive:
        print("\n按 Enter 键开始演示...")
        input()
    else:
        print("\n自动运行模式...")

    # 演示1: 自动微分
    x, y, z = visualize_computation_graph()
    if interactive:
        print("\n按 Enter 键继续下一个演示...")
        input()

    # 演示2: 线性回归
    w, b, lr_history = visualize_linear_regression()
    if interactive:
        print("\n按 Enter 键继续下一个演示...")
        input()

    # 演示3: MLP 决策边界
    mlp_model, mlp_history = visualize_mlp_decision_boundary()
    if interactive:
        print("\n按 Enter 键继续下一个演示...")
        input()

    # 演示4: 注意力权重
    weights_no_mask, weights_masked, head_weights = visualize_attention_weights()
    if interactive:
        print("\n按 Enter 键继续下一个演示...")
        input()

    # 演示5: Transformer 训练
    transformer_block, classifier, tf_history = visualize_transformer_training()

    # 总结
    print("\n" + "=" * 80)
    print("综合演示完成！")
    print("=" * 80)
    print("\n所有可视化图表已保存到 output/ 目录:")
    print("  1. linear_regression_visualization.png - 线性回归训练过程")
    print("  2. mlp_decision_boundary.png - MLP 决策边界")
    print("  3. attention_visualization.png - 注意力权重热力图")
    print("  4. transformer_training.png - Transformer 训练过程")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import sys

    # 设置中文字体支持（可通过fc-list :lang=zh查看系统支持的中文字体）
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 检查是否为交互模式
    interactive = '--no-interactive' not in sys.argv

    # 运行综合演示
    comprehensive_demo(interactive=interactive)
