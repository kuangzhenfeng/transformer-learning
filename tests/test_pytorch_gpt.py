"""
GPT 与 Tokenizer 的基础单元测试
"""

import os
import sys
import unittest
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from minitf.pytorch import GPT, CharTokenizer


class TestCharTokenizer(unittest.TestCase):
    def test_encode_decode(self):
        text = "hello"
        tok = CharTokenizer(text)
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        self.assertEqual(decoded, text)
        self.assertGreater(tok.vocab_size, 0)

    def test_partial_chars(self):
        tok = CharTokenizer("abc")
        ids = tok.encode("ad")  # 'd' 不在词表
        # 仅包含已知字符
        self.assertEqual(ids, [tok.char_to_idx['a']])


class TestGPT(unittest.TestCase):
    def setUp(self):
        self.text = "To be or not to be"
        self.tok = CharTokenizer(self.text)
        self.vocab = self.tok.vocab_size
        self.block_size = 16
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_forward_shapes(self):
        model = GPT(
            vocab_size=self.vocab,
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=128,
            max_len=self.block_size,
        ).to(self.device)
        x = torch.randint(self.vocab, (2, self.block_size), device=self.device)
        logits = model(x)
        self.assertEqual(logits.shape, (2, self.block_size, self.vocab))

    def test_forward_with_targets(self):
        model = GPT(
            vocab_size=self.vocab,
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=128,
            max_len=self.block_size,
        ).to(self.device)
        x = torch.randint(self.vocab, (2, self.block_size), device=self.device)
        logits, loss = model(x, x)
        self.assertEqual(logits.shape, (2, self.block_size, self.vocab))
        self.assertTrue(torch.isfinite(loss))

    def test_generate(self):
        model = GPT(
            vocab_size=self.vocab,
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=128,
            max_len=self.block_size,
        ).to(self.device)
        context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        out = model.generate(context, max_new_tokens=10)
        self.assertEqual(out.shape[1], 11)


if __name__ == '__main__':
    unittest.main()

