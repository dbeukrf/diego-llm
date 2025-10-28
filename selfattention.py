import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, context_length, qkv_bias=False):
        super().__init__()
        self.W_q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape # new batch dimension b
        # x = batch, 2 x 6 x 3
        
        # For inputs where `num_tokens` exceeds `context_length`, this will result in errors
        # in the mask creation further below.
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs  
        # do not exceed `context_length` before reaching this forward method. 
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        attn_scores = queries @ keys.transpose(1, 2) # changed transpose to transpose(1, 2) to get the correct shape
        attn_scores.masked_fill_( # _ ops are in place operations (doesn't create a copy as an intermediate value... for efficiency)
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec




class SelfAttention_v2(nn.Module):

    # modern llms don't use qvk biases anymore...
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # better training dynamics using linear layers instead of torch random
        self.W_q = torch.nn.Linear(d_in, d_out, bias=qkv_bias) # query
        self.W_k = torch.nn.Linear(d_in, d_out, bias=qkv_bias) # key
        self.W_v = torch.nn.Linear(d_in, d_out, bias=qkv_bias) # value

    def forward(self, x):
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[1]**0.5, dim=-1)
        # attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
        context_vec = attn_weights @ values

        return context_vec


'''
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_q = torch.nn.Parameter(torch.randn(d_in, d_out)) # query
        self.W_k = torch.nn.Parameter(torch.randn(d_in, d_out)) # key
        self.W_v = torch.nn.Parameter(torch.randn(d_in, d_out)) # value


    def forward(self, x):
        queries = x @ self.W_q
        keys = x @ self.W_k
        values = x @ self.W_v

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[1]**0.5, dim=-1)
        context_vec = attn_weights @ values

        return context_vec
'''