"""
PyTorch 实现模块

本模块使用 PyTorch 重写之前手工实现的组件，包括：
- 基础神经网络层（Linear, ReLU, Sigmoid）
- Attention 机制（Scaled Dot-Product Attention, Multi-Head Attention）
- Transformer 组件（LayerNorm, FFN, Transformer Block）
- 完整的 GPT 模型
"""

from .nn_modules import (
    MLP,
    ScaledDotProductAttention,
    MultiHeadAttention,
    PositionwiseFeedForward,
    TransformerBlock,
)
from .gpt import GPT
from .tokenizer import CharTokenizer, SimpleWordTokenizer

__all__ = [
    'MLP',
    'ScaledDotProductAttention',
    'MultiHeadAttention',
    'PositionwiseFeedForward',
    'TransformerBlock',
    'GPT',
    'CharTokenizer',
    'SimpleWordTokenizer',
]

