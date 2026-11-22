import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        return self.token(x) + self.position(pos)


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, -1).transpose(1, 2)
        k = k.view(B, T, self.num_heads, -1).transpose(1, 2)
        v = v.view(B, T, self.num_heads, -1).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.attention = SelfAttention(d_model, num_heads)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=4, num_layers=8):
        super().__init__()

        self.embedding = Embedding(vocab_size, d_model, 512)
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, num_heads) for _ in range(num_layers)]
        )
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.lm_head(x)