"""
微型 GPT 模型实现

基于 Transformer 解码器架构的 GPT 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .nn_modules import TransformerBlock


class PositionalEncoding(nn.Module):
    """位置编码
    
    为序列中的每个位置添加位置信息
    
    Args:
        d_model: 模型维度
        max_len: 最大序列长度
        dropout: Dropout 概率
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
        
        Returns:
            添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class GPT(nn.Module):
    """微型 GPT 模型
    
    基于 Transformer 解码器的自回归语言模型
    
    Args:
        vocab_size: 词汇表大小
        d_model: 模型维度
        num_heads: 注意力头数量
        num_layers: Transformer 层数
        d_ff: 前馈网络隐藏层维度
        max_len: 最大序列长度
        dropout: Dropout 概率
    """
    
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=6, 
                 d_ff=1024, max_len=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Token 嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer 解码器层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终的 Layer Normalization
        self.ln_f = nn.LayerNorm(d_model)
        
        # 输出层（语言模型头）
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, idx, targets=None):
        """前向传播
        
        Args:
            idx: 输入 token 索引，形状为 (batch_size, seq_len)
            targets: 目标 token 索引（用于训练），形状为 (batch_size, seq_len)
        
        Returns:
            如果 targets 为 None，返回 logits (batch_size, seq_len, vocab_size)
            否则返回 (logits, loss)
        """
        batch_size, seq_len = idx.shape
        
        # Token 嵌入
        token_emb = self.token_embedding(idx)  # (batch_size, seq_len, d_model)
        
        # 添加位置编码
        x = self.pos_encoding(token_emb)
        
        # 创建因果掩码（防止看到未来的 token）
        mask = torch.triu(torch.ones(seq_len, seq_len, device=idx.device), diagonal=1).bool()
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 通过 Transformer 层
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # 最终的 Layer Normalization
        x = self.ln_f(x)
        
        # 计算 logits
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        # 如果提供了 targets，计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        
        return logits if loss is None else (logits, loss)

    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """生成新的 token
        
        Args:
            idx: 初始 token 索引，形状为 (batch_size, seq_len)
            max_new_tokens: 要生成的最大 token 数量
            temperature: 采样温度（越高越随机）
            top_k: 如果设置，只从概率最高的 k 个 token 中采样
        
        Returns:
            生成的 token 索引，形状为 (batch_size, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # 如果序列太长，截断到 max_len
            idx_cond = idx if idx.size(1) <= self.max_len else idx[:, -self.max_len:]
            
            # 获取预测
            logits = self(idx_cond)
            
            # 只关注最后一个时间步
            logits = logits[:, -1, :] / temperature
            
            # 可选的 top-k 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # 应用 softmax 得到概率
            probs = F.softmax(logits, dim=-1)
            
            # 从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 拼接到序列
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
