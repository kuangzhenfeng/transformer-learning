"""
简单的字符级 Tokenizer

用于将文本转换为 token 索引序列
"""


class CharTokenizer:
    """字符级 Tokenizer
    
    将文本中的每个字符映射到一个唯一的整数索引
    
    Args:
        text: 用于构建词汇表的文本
    """
    
    def __init__(self, text=None):
        if text is not None:
            # 从文本中提取所有唯一字符
            chars = sorted(list(set(text)))
            self.vocab_size = len(chars)
            
            # 创建字符到索引的映射
            self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
            self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        else:
            self.vocab_size = 0
            self.char_to_idx = {}
            self.idx_to_char = {}
    
    def encode(self, text):
        """将文本编码为索引序列
        
        Args:
            text: 输入文本字符串
        
        Returns:
            索引列表
        """
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]
    
    def decode(self, indices):
        """将索引序列解码为文本
        
        Args:
            indices: 索引列表或张量
        
        Returns:
            解码后的文本字符串
        """
        # 如果是张量，转换为列表
        if hasattr(indices, 'tolist'):
            indices = indices.tolist()
        
        return ''.join([self.idx_to_char[i] for i in indices if i in self.idx_to_char])
    
    def save(self, path):
        """保存 tokenizer 到文件
        
        Args:
            path: 保存路径
        """
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab_size': self.vocab_size,
                'char_to_idx': self.char_to_idx,
                'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()}
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path):
        """从文件加载 tokenizer
        
        Args:
            path: 文件路径
        
        Returns:
            CharTokenizer 实例
        """
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls()
        tokenizer.vocab_size = data['vocab_size']
        tokenizer.char_to_idx = data['char_to_idx']
        tokenizer.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
        
        return tokenizer


class SimpleWordTokenizer:
    """简单的词级 Tokenizer
    
    基于空格分词，将文本中的每个词映射到一个唯一的整数索引
    
    Args:
        text: 用于构建词汇表的文本
        max_vocab_size: 最大词汇表大小（默认为 None，不限制）
    """
    
    def __init__(self, text=None, max_vocab_size=None):
        self.max_vocab_size = max_vocab_size
        
        if text is not None:
            # 分词并统计词频
            words = text.split()
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # 按词频排序
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            
            # 限制词汇表大小
            if max_vocab_size is not None:
                sorted_words = sorted_words[:max_vocab_size]
            
            # 添加特殊 token
            vocab = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
            vocab.extend([word for word, _ in sorted_words])
            
            self.vocab_size = len(vocab)
            self.word_to_idx = {word: i for i, word in enumerate(vocab)}
            self.idx_to_word = {i: word for i, word in enumerate(vocab)}
            
            # 特殊 token 的索引
            self.pad_idx = 0
            self.unk_idx = 1
            self.bos_idx = 2
            self.eos_idx = 3
        else:
            self.vocab_size = 0
            self.word_to_idx = {}
            self.idx_to_word = {}
            self.pad_idx = 0
            self.unk_idx = 1
            self.bos_idx = 2
            self.eos_idx = 3
    
    def encode(self, text, add_special_tokens=False):
        """将文本编码为索引序列
        
        Args:
            text: 输入文本字符串
            add_special_tokens: 是否添加 BOS 和 EOS token
        
        Returns:
            索引列表
        """
        words = text.split()
        indices = [self.word_to_idx.get(word, self.unk_idx) for word in words]
        
        if add_special_tokens:
            indices = [self.bos_idx] + indices + [self.eos_idx]
        
        return indices

    
    def decode(self, indices):
        """将索引序列解码为文本
        
        Args:
            indices: 索引列表或张量
        
        Returns:
            解码后的文本字符串
        """
        # 如果是张量，转换为列表
        if hasattr(indices, 'tolist'):
            indices = indices.tolist()
        
        words = [self.idx_to_word.get(i, '<UNK>') for i in indices]
        # 过滤掉特殊 token
        words = [w for w in words if w not in ['<PAD>', '<BOS>', '<EOS>']]
        
        return ' '.join(words)
