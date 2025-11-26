"""
微型 GPT 训练示例（字符级）

- 使用 CharTokenizer 构建词表
- 在小文本上训练自回归语言模型
- 训练后进行采样生成
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim

# 添加项目根目录到路径，便于直接运行脚本
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minitf.pytorch import GPT, CharTokenizer


def build_dataset(text, block_size=64, split_ratio=0.9):
    """构建字符级数据集，返回 (train_data, val_data, tokenizer)"""
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(len(data) * split_ratio)
    return (data[:n], data[n:]), tok


class CharDataset(torch.utils.data.Dataset):
    def __init__(self, data, block_size):
        self.data = data
        # 确保 block_size 不超过数据长度-1
        self.block_size = max(1, min(block_size, len(self.data) - 1))

    def __len__(self):
        # 至少返回 0
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        # 保护性处理，防止越界（当 __len__ == 0 时不会被调用）
        idx = int(idx)
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1: idx + 1 + self.block_size]
        return x, y


def get_batch(dataset, batch_size, device):
    n = len(dataset)
    if n <= 0:
        raise ValueError("Dataset too small for the chosen block_size. Reduce block_size or provide more text.")
    ix = torch.randint(n, (batch_size,))
    X = torch.stack([dataset[i][0] for i in ix]).to(device)
    Y = torch.stack([dataset[i][1] for i in ix]).to(device)
    return X, Y


def train(train_text=None, block_size=128, batch_size=64, steps=1000,
          lr=3e-4, eval_interval=200, device=None):
    # 默认使用一小段文本（莎士比亚节选或占位符）
    if train_text is None:
        train_text = (
            "To be, or not to be, that is the question:\n"
            "Whether 'tis nobler in the mind to suffer\n"
            "The slings and arrows of outrageous fortune,\n"
            "Or to take arms against a sea of troubles\n"
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集
    (train_data, val_data), tok = build_dataset(train_text, block_size)
    train_ds = CharDataset(train_data, block_size)
    val_ds = CharDataset(val_data, block_size)

    # 模型
    model = GPT(
        vocab_size=tok.vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        max_len=block_size,
        dropout=0.1,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    def estimate_loss(iters=50):
        model.eval()
        losses = {}
        with torch.no_grad():
            for split, ds in [("train", train_ds), ("val", val_ds)]:
                total = 0.0
                for _ in range(iters):
                    X, Y = get_batch(ds, batch_size, device)
                    _, loss = model(X, Y)
                    total += loss.item()
                losses[split] = total / iters
        model.train()
        return losses

    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        X, Y = get_batch(train_ds, batch_size, device)
        logits, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == 1:
            losses = estimate_loss(20)
            dt = time.time() - t0
            print(f"step {step:4d} | train {losses['train']:.3f} | val {losses['val']:.3f} | time {dt:.1f}s")

    # 采样
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=50)[0]
    print("\n=== Sample ===")
    print(tok.decode(out.cpu().tolist()))

    return model, tok


if __name__ == "__main__":
    train(steps=200, eval_interval=50, block_size=128, batch_size=64)

