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

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, -1).transpose(1, 2)
        k = k.view(B, T, self.num_heads, -1).transpose(1, 2)
        v = v.view(B, T, self.num_heads, -1).transpose(1, 2)

        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].bool()
        else:
            mask = None

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
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

    def forward(self, x, attention_mask=None):
        x = x + self.attention(self.norm1(x), attention_mask=attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=4, num_layers=8, max_seq_len=512):
        super().__init__()

        self.embedding = Embedding(vocab_size, d_model, max_seq_len)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads) for _ in range(num_layers)]
        )
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x, attention_mask=None):
        x = self.embedding(x)
        for block in self.blocks: x = block(x, attention_mask=attention_mask)
        x = self.norm(x)
        return x

class SentimentHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.linear = nn.Linear(d_model, 2)

    def forward(self, x, attention_mask=None):
        if attention_mask is None:
            x = x.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(x.dtype)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.linear(x)

class MLMHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()

        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(x)

class QAHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, 2)

    def forward(self, x):
        logits = self.linear(x)
        start_logits = logits[:, :, 0]
        end_logits = logits[:, :, 1]

        return start_logits, end_logits

class EncoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        num_heads=4,
        num_layers=8,
        max_seq_len=512
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len
        )

        self.sentiment_head = SentimentHead(d_model)
        self.mlm_head = MLMHead(d_model, vocab_size)
        self.qa_head = QAHead(d_model)

    def forward(self, x, mode="mlm", attention_mask=None):
        h = self.encoder(x, attention_mask=attention_mask)

        if mode == "mlm":
            return self.mlm_head(h)

        elif mode == "sentiment":
            return self.sentiment_head(
                h, attention_mask=attention_mask
            )

        elif mode == "qa":
            start_logits, end_logits = self.qa_head(h)
            if attention_mask is not None:
                invalid = ~attention_mask.bool()
                start_logits = start_logits.masked_fill(invalid, float("-inf"))
                end_logits = end_logits.masked_fill(invalid, float("-inf"))
            return start_logits, end_logits

        else:
            raise ValueError(f"Unknown mode: {mode}")