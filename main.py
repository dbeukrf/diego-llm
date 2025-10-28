from torch._tensor import Tensor
from torch._tensor import Tensor


from typing import Any
from gptdataset import GPTDatasetV1
from tokenizer import SimpleTokenizerV2
from selfattention import CausalAttention, SelfAttention_v2
from multiheadattention import MultiHeadAttentionWrapper, MultiHeadAttention
from gptmodel import GPTModel
from layernorm import LayerNorm
from feedforward import FeedForward
from deepneuralnetwork import ExampleDeepNeuralNetwork, print_gradients

import torch.nn as nn
import os
import requests
import re
import torch
import urllib.request
import tiktoken


def main():

    ## ********** Chapter 2: Working with Text Data (tokenization & embedding) ********** ##

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    max_length = 4
    dataloader = GPTDatasetV1.create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=4, shuffle=False)

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    # print("Token IDs:\n", inputs)
    # print("\nInputs shape:\n", inputs.shape)

    vocab_size = 50257
    output_dim = 256

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    token_embeddings = token_embedding_layer(inputs)
    # print(token_embeddings.shape)

    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    # print(pos_embedding_layer.weight)

    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    # print(pos_embeddings.shape)
    # print(pos_embeddings)


    input_embeddings = token_embeddings + pos_embeddings
    # print(input_embeddings.shape)
    # print(input_embeddings)

    ## ********** Chapter 3: Attention Mechanisms ********** ##

    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your     (x^1)
        [0.55, 0.87, 0.66], # journey  (x^2)
        [0.57, 0.85, 0.64], # starts   (x^3)
        [0.22, 0.58, 0.33], # with     (x^4)
        [0.77, 0.25, 0.10], # one      (x^5)
        [0.05, 0.80, 0.55]] # step     (x^6)
    )
    # only 2nd input token is the query
    query = inputs[1] # 2nd input token is the query

    attn_scores2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate[Tensor](inputs):
        attn_scores2[i] = torch.dot(x_i, query) # dot product (transpose not necessary here since they are 1-dim vectors)


    # dot product attention: multiplying two vectors elements-wise and summing the resulting products:
    res = 0.
    for idx, element in enumerate[Tensor](inputs[0]):
        res += inputs[0][idx] * query[idx]


    # print(res)
    # print(torch.dot(inputs[0], query))
    
    # Normalizing attention scores to help with the optimization process:
    attn_weights_2 = torch.softmax(attn_scores2, dim=0)
    # print(attn_weights_2)
    # print(attn_weights_2.sum())

    # naive approach:
    # attn_weights_2 = torch.exp(attn_scores2) / torch.exp(attn_scores2).sum(dim=0)
    # print(attn_weights_2)

    context_vec_2 = torch.zeros(query.shape)
    for i, x_i in enumerate[Tensor](inputs):
        # print(i, x_i)
        # print(f"{attn_weights_2[i]} ----> {inputs[i]}")
        context_vec_2 += attn_weights_2[i] * x_i

    # print(context_vec_2)    


    # all input tokens are queries
    # simple self-attention mechanism without trainable weights
    attn_scores = torch.empty(6, 6)

    for i, x_i in enumerate[Tensor](inputs):
        for j, x_j in enumerate[Tensor](inputs):
            attn_scores[i, j] = torch.dot(x_i, x_j)

    # for loops are slow because they can't be optimised compared to matrix multiplications
    # matrix multiplication:
    attn_scores = inputs @ inputs.T # (6, 3) @ (3, 6) = (6, 6)
    # print(attn_scores)

    attn_weight = torch.softmax(attn_scores, dim=1)
    # print(attn_weight)

    all_context_ves = attn_weight @ inputs # (6, 6) @ (6, 3) = (6, 3)
    # print(all_context_ves)


    # implementing self-attention with trainable weights:
    x_2 = inputs[1]
    d_in = inputs.shape[1]
    d_out = 2

    torch.manual_seed(123)

    W_q = torch.nn.Parameter(torch.randn(d_in, d_out)) # query
    # print(W_q)
    W_k = torch.nn.Parameter(torch.randn(d_in, d_out)) # key
    # print(W_k)
    W_v = torch.nn.Parameter(torch.randn(d_in, d_out)) # value
    # print(W_v)

    # projecting 3 dimensional input token into 2 dimensional query, key, and value vectors
    query_2 = x_2 @ W_q # (3, 2)
    key_2 = x_2 @ W_k # (3, 2)
    value_2 = x_2 @ W_v # (3, 2)

    # print(query_2)
    # print(key_2)
    # print(value_2)

    keys = inputs @ W_k # (6, 3) @ (3, 2) = (6, 2)
    values = inputs @ W_v # (6, 3) @ (3, 2) = (6, 2)
    queries = inputs @ W_q # (6, 3) @ (3, 2) = (6, 2)

    # print(keys)
    # print(values)
    # print(queries)

    keys_2 = keys[1]
    attn_scores_22 = torch.dot(query_2, keys_2)
    # print(attn_scores_22)
    attn_scores_2 = query_2 @ keys_2.T # (2, 2)
    # print(attn_scores_2)

    d_k = keys.shape[1]
    torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
    # print(attn_weights_2)

    context_vec_2 = attn_weights_2 @ values # (2, 2) @ (2, 3) = (2, 3)
    # print(context_vec_2)

    self_attn = SelfAttention_v2(d_in, d_out)
    self_attn_output = self_attn(inputs)
    # print(self_attn_output)

    queries = self_attn.W_q(inputs)
    keys = self_attn.W_k(inputs)
    values = self_attn.W_v(inputs)



    # Step 1 attention scores (unnormalized)
    attn_scores = queries @ keys.T
    # print(attn_scores)

    # Step 2attention weights (normalized)
    attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
    # print(attn_weights)

    context_length = attn_scores.shape[0]
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    # print(mask_simple)

    # Step 3 masked attention scores (unnormalized)
    masked_simple = attn_weights * mask_simple
    # print(masked_simple)

    # Step 4 masked attention weights (normalized)
    row_sums = masked_simple.sum(dim=-1, keepdim=True)
    masked_simple_norm = masked_simple / row_sums
    # print(masked_simple_norm)


    # do it in 3 steps instead:
    
    # attention scores (unnormalized):
    attn_scores = queries @ keys.T

    # masked attention scores (unnormalized):
    masked = attn_scores * mask_simple
    
    # masked attention weights (normalized):
    attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)

    layer = torch.nn.Dropout(0.5)
    example = torch.ones(6, 6)
    # print(layer(example))
    # print(layer(attn_weights))


    batch = torch.stack((inputs, inputs), dim=0)
    # print(batch.shape)


    context_length = batch.shape[1]
    ca = CausalAttention(d_in, d_out, dropout=0.0, context_length=context_length)
    
    context_vecs = ca(batch)
    # print(context_vecs)
    # print(context_vecs.shape)

    d_in, d_out = batch.shape[-1], 2
    mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout=0.0)
    mha_output = mha(batch)
    # print(mha_output)

    batch_size, context_length, d_in = batch.shape
    d_out = 4
    mha = MultiHeadAttention(d_in, d_out, context_length, dropout=0.0, num_heads=2)

    context_vecs = mha(batch)
    # print(context_vecs)
    # print(context_vecs.shape)



    ## ********** Chapter 4: LLM Architecture (GPT-2) ********** ##

    GPT_CONFIG_124M = {
        "vocab_size": 50257, # vocabulary size of the model
        "context_length": 1024, # context length of the model, note: the larger, the more computationally taxing it will be for your hardware (RAM, VRAM, etc.)
        "emb_dim": 768, # embedding dimension of the model
        "n_heads": 12, # number of attention heads in the model
        "n_layers": 12, # number of layers in the model (# of transformer blocks)
        "drop_rate": 0.1, # dropout rate of the model
        "qkv_bias": False # Query, Key, Value bias
    }

    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []

    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    # print(batch)

    
    model = GPTModel(cfg = GPT_CONFIG_124M)

    logits = model(batch) # logits is jargon for the last layer outputs 
    # print(logits)
    # print(logits.shape)
    # print(batch.shape)

    
    batch_example = torch.randn(2, 5) # batch of 2 examples, each with 5 tokens

    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU()) # nn.Linear is a linear layer (fully connected layer) neural network layer
    # ReLu is a activation function that is used to introduce non-linearity into the model, you should pair  a linear layer with a non-linear activation function.
    # Thus the network can learn more complex patterns.
    
    out = layer(batch_example)
    # print(out)
    # print(out.shape)

    out.shape # (2, 5)
    mean = out.mean(dim=-1, keepdim=True) # (2,) assuming the last dimension is the feature dimension
    var = out.var(dim=-1, keepdim=True) # (2,) assuming the last dimension is the feature dimension

    normed = ((out - mean) / torch.sqrt(var)) # the standrad deviation is the square root of the variance
    normed.var(dim=-1, keepdim=True) # (2,)


    ln = LayerNorm(6)
    outputs_normed = ln(out)
    # print(outputs_normed)
    # print(outputs_normed.mean(dim=-1, keepdim=True))
    # print(outputs_normed.var(dim=-1, keepdim=True)) # (2,)

    ffn = FeedForward(GPT_CONFIG_124M)


    x = torch.randn(2, 3, 768)
    # print(ffn(x).shape)

    layer_sizes = [3, 3, 3, 3, 3, 1]
    model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
    )
    sample_input = torch.randn(2, 3)
    # print_gradients(model_without_shortcut, sample_input)


    model = GPTModel(cfg = GPT_CONFIG_124M)
    sample_input = torch.randint(0, 50257, (2, 1024))
    # print(model(sample_input).shape)


    model = GPTModel(cfg = GPT_CONFIG_124M)

    out = model(batch)
    # print("Input batch:\n", batch)
    # print("\nOutput shape:", out.shape)
    # print(out)


    total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters: {total_params:,}")

    # print("Token embedding layer shape:", model.tok_emb.weight.shape)
    # print("Output layer shape:", model.out_head.weight.shape)      

    total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())
    # print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")  

    # Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
    total_size_bytes = total_params * 4

    # Convert to megabytes
    total_size_mb = total_size_bytes / (1024 * 1024)

    # print(f"Total size of the model: {total_size_mb:.2f} MB")

    start_context = "Hello, I am"

    encoded = tokenizer.encode(start_context)
    print("encoded: ", encoded)

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    model.eval() # disable dropout

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor, 
        max_new_tokens=6, 
        context_size=GPT_CONFIG_124M["context_length"]
    )

    print("Output:", out)
    print("Output length:", len(out[0]))

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print("Decoded text:", decoded_text)



def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx



if __name__ == "__main__":
    main()