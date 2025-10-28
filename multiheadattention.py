import torch
import torch.nn as nn
from selfattention import CausalAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads = 2, qkv_bias = False):
        super().__init__()
        assert (d_out % num_heads) == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.heads_dim = d_out // num_heads # reduce the projection dim to match desired output dim

        self.W_q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.output_proj = torch.nn.Linear(d_out, d_out) # linear layer to project the concatenated heads back to the original dimension (combine head outputs)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )


    def forward(self, x):
        b, num_tokens, d_in = x.shape # new batch dimension b

        keys = self.W_k(x) # shape: (b, num_tokens, d_out)
        queries = self.W_q(x)
        values = self.W_v(x)

        # implicitly split the matrix by adding a num_heads dimension
        # unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, heads_dim
        keys = keys.view(b, num_tokens, self.num_heads, self.heads_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.heads_dim)
        values = values.view(b, num_tokens, self.num_heads, self.heads_dim)

        # transpose: (b, num_tokens, num_heads, heads_dim) -> (b, num_heads, num_tokens, heads_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # compute scaled dot-product attention (self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3) # shape: (b, num_heads, num_tokens, num_tokens)

        # original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # use the mask to fill attention scores
        attn_scores.masked_fill_(~mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # shape: (b, num_tokens, num_heads, heads_dim)
        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.output_proj(context_vec)
        return context_vec
       




class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads = 2, qkv_bias = False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, dropout, context_length, qkv_bias) for _ in range(num_heads)
        ])

    # concatenate the outputs of the heads
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
