from torch._tensor import Tensor
from gptdownload import download_and_load_gpt2
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
# Alternatively:
# from llms_from_scratch.ch05 import download_and_load_gpt2


from typing import Any
from pathlib import Path
from gptdataset import GPTDatasetV1
from tokenizer import SimpleTokenizerV2
from selfattention import CausalAttention, SelfAttention_v2
from multiheadattention import MultiHeadAttentionWrapper, MultiHeadAttention
from gptmodel import GPTModel
from layernorm import LayerNorm
from feedforward import FeedForward
from deepneuralnetwork import ExampleDeepNeuralNetwork, print_gradients
from spamdataset import SpamDataset
from instructiondataset import InstructionDataset

import torch.nn as nn
import os
import requests
import json
import zipfile
import psutil
import re
import torch
import urllib.request
import tiktoken
import numpy as np
import pandas as pd
import time


def main():
    '''
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

    '''
    '''
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


    '''
    '''
    ## ********** Chapter 4: LLM Architecture (GPT-2) ********** ##

    GPT_CONFIG_124M = {
        "vocab_size": 50257, # vocabulary size of the model
        "context_length": 256, # context length of the model, note: the larger, the more computationally taxing it will be for your hardware (RAM, VRAM, etc.)
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
    # print("encoded: ", encoded)

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    # print("encoded_tensor.shape:", encoded_tensor.shape)

    model.eval() # disable dropout

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor, 
        max_new_tokens=6, 
        context_size=GPT_CONFIG_124M["context_length"]
    )

    # print("Output:", out)
    # print("Output length:", len(out[0]))

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    # print("Decoded text:", decoded_text)


    '''
    ## ********** Chapter 5: Pretraining on Unlabeled Data ********** ##

    '''

    file_path = "the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    GPT_CONFIG_124M = {
        "vocab_size": 50257, # vocabulary size of the model
        "context_length": 256, # context length of the model, note: the larger, the more computationally taxing it will be for your hardware (RAM, VRAM, etc.)
        "emb_dim": 768, # embedding dimension of the model
        "n_heads": 12, # number of attention heads in the model
        "n_layers": 12, # number of layers in the model (# of transformer blocks)
        "drop_rate": 0.1, # dropout rate of the model
        "qkv_bias": False # Query, Key, Value bias
    }


    model = GPTModel(cfg = GPT_CONFIG_124M)
    model.eval() # disable dropout

    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = text_to_token_ids(start_context, tokenizer)
    # token_ids = SimpleTokenizerV2.token_ids_to_text(start_context, tokenizer)
    # print(token_ids)
    # print(SimpleTokenizerV2.token_ids_to_text(token_ids, tokenizer))
    

    token_ids = generate_text_simple(model, text_to_token_ids(start_context, tokenizer), 10, GPT_CONFIG_124M["context_length"])
    # print(token_ids)
    # print(token_ids.squeeze(0).shape)

    decoded_text = SimpleTokenizerV2.token_ids_to_text(token_ids, tokenizer)
    # print(decoded_text)


    inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]

    targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [1107,  588, 11311]]) #  " really like chocolate"]

    with torch.no_grad():
        logits = model(inputs)

    probas = torch.softmax(logits, dim=-1) # Probability of each token in vocabulary
    # print(probas.shape) # Shape: (batch_size, num_tokens, vocab_size)

    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    # print("Token IDs:\n", token_ids)

    # print(f"Targets batch 1: {SimpleTokenizerV2.token_ids_to_text(targets[0], tokenizer)}")
    # print(f"Outputs batch 1: {SimpleTokenizerV2.token_ids_to_text(token_ids[0].flatten(), tokenizer)}")


    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    # print("Text 1:", target_probas_1)

    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    # print("Text 2:", target_probas_2)



    # Compute logarithm of all token probabilities
    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    # print(log_probas)

    # Calculate the average probability for each token
    avg_log_probas = torch.mean(log_probas)
    # print(avg_log_probas)

    neg_avg_log_probas = avg_log_probas * -1
    # print(neg_avg_log_probas)

    # Logits have shape (batch_size, num_tokens, vocab_size)
    # print("Logits shape:", logits.shape)

    # Targets have shape (batch_size, num_tokens)
    # print("Targets shape:", targets.shape)

    # for the cross-entropy loss, we need to flatten the logits and targets
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()

    # print("Flattened logits:", logits_flat.shape)
    # print("Flattened targets:", targets_flat.shape)

    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    # print(loss)

    perplexity = torch.exp(loss)
    # print(perplexity)

    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))

    # print("Characters:", total_characters)
    # print("Tokens:", total_tokens)

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]



    train_loader = GPTDatasetV1.create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = GPTDatasetV1.create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )


    # Sanity check
    # if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
    #     print("Not enough tokens for the training loader. "
    #         "Try to lower the `GPT_CONFIG_124M['context_length']` or "
    #         "increase the `training_ratio`")

    # if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
    #     print("Not enough tokens for the validation loader. "
    #         "Try to lower the `GPT_CONFIG_124M['context_length']` or "
    #         "decrease the `training_ratio`")


    # print("Train loader:")
    # for x, y in train_loader:
    #     print(x.shape, y.shape)

    # print("\nValidation loader:")
    # for x, y in val_loader:
    #     print(x.shape, y.shape)

    
    train_tokens = 0
    for input_batch, target_batch in train_loader:
        train_tokens += input_batch.numel()

    val_tokens = 0
    for input_batch, target_batch in val_loader:
        val_tokens += input_batch.numel()

    # print("Training tokens:", train_tokens)
    # print("Validation tokens:", val_tokens)
    # print("All tokens:", train_tokens + val_tokens)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes
    model.eval()  # Set model to evaluation mode to disable dropout

    # torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader

    with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    # print("Training loss:", train_loss)
    # print("Validation loss:", val_loss)

    '''

    ## TRAINING THE MODEL ##
    '''

    # Uncomment the following code to calculate the execution time
    # import time
    # start_time = time.time()

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    # Uncomment the following code to show the execution time
    # end_time = time.time()
    # execution_time_minutes = (end_time - start_time) / 60
    # print(f"Training completed in {execution_time_minutes:.2f} minutes.")


    # NEW: use CPU here as inference is cheap with 
    # this model and to ensure readers get same results in the
    # remaining sections of this book
    inference_device = torch.device("cpu")

    model.to(inference_device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(inference_device),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    print("Output text:\n", SimpleTokenizerV2.token_ids_to_text(token_ids, tokenizer))
    
    '''
    ######################################
    '''

    vocab = { 
    "closer": 0,
    "every": 1, 
    "effort": 2, 
    "forward": 3,
    "inches": 4,
    "moves": 5, 
    "pizza": 6,
    "toward": 7,
    "you": 8
    } 
    inference_device = torch.device("cpu")
    inverse_vocab = {v: k for k, v in vocab.items()}

    # Suppose input is "every effort moves you", and the LLM
    # returns the following logits for the next token:
    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )

    probas = torch.softmax(next_token_logits, dim=0)
    next_token_id = torch.argmax(probas).item()

    # The next generated token is then as follows:
    print(inverse_vocab[next_token_id])



    next_token_id = torch.multinomial(probas, num_samples=1).item()
    print(inverse_vocab[next_token_id])

    print_sampled_tokens(probas)


    # Temperature values
    temperatures = [1, 0.1, 5]  # Original, higher confidence, and lower confidence
    # lower temperature -> more confident, higher probability of the most likely token
    # higher temperature -> less confident, lower probability of the most likely token
    

    # Calculate scaled probabilities
    scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
    print_sampled_tokens(scaled_probas[1])
    print_sampled_tokens(scaled_probas[2])



    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)

    print("Top logits:", top_logits)
    print("Top positions:", top_pos)


    # new_logits = torch.where(
    #     condition=next_token_logits < top_logits[-1],
    #     input=torch.tensor(float("-inf")), 
    #     other=next_token_logits
    # )

    # alternative way, more efficient:
    new_logits = torch.full_like( # create tensor containing -inf values
    next_token_logits, -torch.inf
    )   
    new_logits[top_pos] = next_token_logits[top_pos] # copy top k values into the -inf tensor
    print(new_logits)

    topk_probas = torch.softmax(new_logits, dim=0)
    print(topk_probas)



    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(inference_device),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4
    )

    print("Output text:\n", SimpleTokenizerV2.token_ids_to_text(token_ids, tokenizer))

    torch.save(model.state_dict(), "model.pth")

    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
    model.eval();

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        }, 
        "model_and_optimizer.pth"
        )

    checkpoint = torch.load("model_and_optimizer.pth", weights_only=True)

    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train();

    
    torch.manual_seed(123)

    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=50,
        temperature=1.5
    )

    print("Output text:\n", SimpleTokenizerV2.token_ids_to_text(token_ids, tokenizer))


    '''


    

    ## Loading pretrained weights from OpenAI ##
    '''

    import tensorflow as tf
    from importlib.metadata import version
    try:
        print("TensorFlow version:", version("tensorflow"))
        print("tqdm version:", version("tqdm"))
    except Exception:
        print("Note: Could not retrieve package versions")


    file_name = "gpt2-small-124M.pth"
    # file_name = "gpt2-medium-355M.pth"
    # file_name = "gpt2-large-774M.pth"
    # file_name = "gpt2-xl-1558M.pth"

    url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}"

    if not os.path.exists(file_name):
        urllib.request.urlretrieve(url, file_name)
        print(f"Downloaded to {file_name}")


    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
    # print("Settings:", settings)
    # print("Params:", params)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": True
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Copy the base configuration and update with specific model settings
    model_name = "gpt2-small (124M)"  # Example model name
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

    gpt = GPTModel(NEW_CONFIG)
    gpt.eval();

    load_weights_into_gpt(gpt, params)
    gpt.to(device)


    torch.manual_seed(123)
    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=25,
        context_size=NEW_CONFIG["context_length"],
        top_k=50,
        temperature=1.5
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))



    '''
    ## ********** Chapter 6: Finetuning for Text Classification ********** ##

    '''
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"


    
    try:
        download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    except (requests.exceptions.RequestException, TimeoutError) as e:
        print(f"Primary URL failed: {e}. Trying backup URL...")
        url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
        download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

    
    
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    # print(df["Label"].value_counts())

    balanced_df = create_balanced_dataset(df)
    # print(balanced_df["Label"].value_counts())
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})   
    # print(balanced_df)


    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1) # 70% for training, 10% for validation (70 - 80 is a good range for training)
    # Test size is implied to be 0.2 as the remainder

    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)

    # print(train_df)
    # print(validation_df)
    # print(test_df)


    tokenizer = tiktoken.get_encoding("gpt2")
    # print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})) # this is the end of text token


    train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer
    )

    # print(train_dataset.max_length)


    val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
    )
    test_dataset = SpamDataset(
        csv_file="test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )


    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    # print("Train loader:")
    # for input_batch, target_batch in train_loader:
    #     pass

    # print("Input batch dimensions:", input_batch.shape)
    # print("Label batch dimensions", target_batch.shape)

    # print(f"{len(train_loader)} training batches")
    # print(f"{len(val_loader)} validation batches")
    # print(f"{len(test_loader)} test batches")



    CHOOSE_MODEL = "gpt2-small (124M)"

    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
        f"Dataset length {train_dataset.max_length} exceeds model's context "
        f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
        f"`max_length={BASE_CONFIG['context_length']}`"
    )


    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval();


    text_1 = "Every effort moves you"

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_1, tokenizer),
        max_new_tokens=15,
        context_size=BASE_CONFIG["context_length"]
    )

    # print(token_ids_to_text(token_ids, tokenizer))


    text_2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_2, tokenizer),
        max_new_tokens=23,
        context_size=BASE_CONFIG["context_length"]
    )

    # print(token_ids_to_text(token_ids, tokenizer))



    for param in model.parameters(): # weight tensor or bias tensor won't be trained when training the model (freezing the model)
        param.requires_grad = False


    torch.manual_seed(123)

    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)


    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0)
    # print("Inputs:", inputs)
    # print("Inputs dimensions:", inputs.shape) # shape: (batch_size, num_tokens)

    with torch.no_grad():
        outputs = model(inputs)

    # print("Outputs:\n", outputs)
    # print("Outputs dimensions:", outputs.shape) # shape: (batch_size, num_tokens, num_classes)

    # print("Last output token:", outputs[:, -1, :])

    probas = torch.softmax(outputs[:, -1, :], dim=-1)
    label = torch.argmax(probas)
    # print("Class label:", label.item())

    logits = outputs[:, -1, :]
    label = torch.argmax(logits)
    # print("Class label:", label.item())



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes

    torch.manual_seed(123) # For reproducibility due to the shuffling in the training data loader

    train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")


    with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
        test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

    print(f"Training loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")
    print(f"Test loss: {test_loss:.3f}")


    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)


    start_time = time.time()

    num_epochs = 5
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")




    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")



    text_1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )

    print(classify_review(
        text_1, model, tokenizer, device, max_length=train_dataset.max_length
    ))


    text_2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )

    print(classify_review(
        text_2, model, tokenizer, device, max_length=train_dataset.max_length
    ))

    # save the model
    torch.save(model.state_dict(), "review_classifier.pth")

    # load the model
    # model_state_dict = torch.load("review_classifier.pth", map_location=device, weights_only=True)
    # model.load_state_dict(model_state_dict)

    '''
    ## ********** Chapter 7: Instruction Finetuning ********** ## 
    '''

    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )

    data = download_and_load_file(file_path, url)
    # print("Number of entries:", len(data))

    # print("Example entry:\n", data[50])
    # print("Another example entry:\n", data[999])

    model_input = format_input(data[50])
    desired_response = f"\n\n### Response:\n{data[50]['output']}"

    # print(model_input + desired_response)

    model_input = format_input(data[999])
    desired_response = f"\n\n### Response:\n{data[999]['output']}"

    # print(model_input + desired_response)


    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)    # 10% for testing
    val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    # print("Training set length:", len(train_data))
    # print("Validation set length:", len(val_data))
    # print("Test set length:", len(test_data))




    tokenizer = tiktoken.get_encoding("gpt2")
    # print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})) # this is the end of text token

    # inputs_1 = [0, 1, 2, 3, 4]
    # inputs_2 = [5, 6]
    # inputs_3 = [7, 8, 9]

    # batch = (
    #     inputs_1,
    #     inputs_2,
    #     inputs_3
    # )
    # print(custom_collate_draft_1(batch))


    # inputs, targets = custom_collate_draft_2(batch)
    # print(inputs)
    # print(targets)


    # inputs, targets = custom_collate_fn(batch)
    # print(inputs)
    # print(targets)



    # logits_1 = torch.tensor(
    # [[-1.0, 1.0],  # 1st training example
    #  [-0.5, 1.5]]  # 2nd training example
    # )
    # targets_1 = torch.tensor([0, 1])

    # loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
    # print(loss_1)



    # logits_2 = torch.tensor(
    # [[-1.0, 1.0],
    #  [-0.5, 1.5],
    #  [-0.5, 1.5]]  # New 3rd training example
    # )
    # targets_2 = torch.tensor([0, 1, 1])
    # loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
    # print(loss_2)



    # targets_3 = torch.tensor([0, 1, -100])
    # loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
    # print(loss_3)
    # print("loss_1 == loss_3:", loss_1 == loss_3)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=1024
    )


    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )


    # print("Train loader:")
    # for inputs, targets in train_loader:
    #     print(inputs.shape, targets.shape)

    # print(inputs[0])

    # print(targets[0])

    '''

    ## Loading a pretrained LLM
    '''

    BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir="gpt2"
    )

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval();
    

    torch.manual_seed(123)

    input_text = format_input(val_data[0])
    # print(input_text)



    token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
    )
    # generated_text = token_ids_to_text(token_ids, tokenizer)


    # change it to how you like, this is done in the background of the model, chatgpt style just gives you answer
    # response_text = (
    # generated_text[len(input_text):]
    # .replace("### Response:", "")
    # .strip()
    # )
    # print(response_text)



    # model.to(device)

    # torch.manual_seed(123)

    # with torch.no_grad():
    #     train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    #     val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    # print("Training loss:", train_loss)
    # print("Validation loss:", val_loss)
    
    '''

    '''
    # run chapter 5 versions calc_loss_batch & calc_loss_loader (we are interested in all tokens, not just the last one)
    start_time = time.time()

    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    num_epochs = 2

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    '''


    '''
    torch.manual_seed(123)


    for entry in test_data[:3]:

        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
    )

        print(input_text)
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nModel response:\n>> {response_text.strip()}")
        print("-------------------------------------")




    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        test_data[i]["model_response"] = response_text


    with open("instruction-data-with-response.json", "w") as file:
        json.dump(test_data, file, indent=4)  # "indent" for pretty-printing

    print(test_data[0])
    
    
    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth" # sft = supervised finetuning
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")
    '''


    # Load test data from json file
    with open("instruction-data-with-response.json", "r") as file:
        test_data = json.load(file)

    # Load model via pth file
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "emb_dim": 1024,
        "n_layers": 24,
        "n_heads": 16
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPTModel(BASE_CONFIG)
    model.load_state_dict(torch.load("gpt2-medium355M-sft.pth", map_location=device))
    model.eval()
    model.to(device)



    # Evaluating the finetuned LLM with Ollama

    # start ollama in seperate terminal:
    # ollama run llama3.2
    # ollama serve
    ollama_running = check_if_running("ollama")

    if not ollama_running:
        raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
    print("Ollama running:", check_if_running("ollama"))

    

    model = "llama3.2"
    # result = query_model("What do Llamas eat?", model)
    # print(result)



    for entry in test_data[:3]:
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry['model_response']}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
        )
        print("\nDataset response:")
        print(">>", entry['output'])
        print("\nModel response:")
        print(">>", entry["model_response"])
        print("\nScore:")
        print(">>", query_model(prompt))
        print("\n-------------------------")


    scores = generate_model_scores(test_data, "model_response")
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    # print(f"Average score: {sum(scores)/len(scores):.2f}\n")
    # ^^division by zero error...












def generate_model_scores(json_data, json_key, model="llama3.2"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = f"""
            You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance.
            You will be given an instruction, a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing the evaluation criteria.
            Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
            Please do not generate any other opening, closing, and explanations.

            Here is the rubric you should use to build your answer:
            1: The response fails to address the instructions, providing irrelevant, incorrect, or excessively verbose information that detracts from the user's request.
            2: The response partially addresses the instructions but includes significant inaccuracies, irrelevant details, or excessive elaboration that detracts from the main task.
            3: The response follows the instructions with some minor inaccuracies or omissions. It is generally relevant and clear, but may include some unnecessary details or could be more concise.
            4: The response adheres to the instructions, offering clear, accurate, and relevant information in a concise manner, with only occasional, minor instances of excessive detail or slight lack of clarity.
            5: The response fully adheres to the instructions, providing a clear, accurate, and relevant answer in a concise and efficient manner. It addresses all aspects of the request without unnecessary details or elaboration.

            Provide your feedback as follows:

            Feedback:::
            Evaluation: (your rationale for the rating, as a text)
            Total rating: (your rating, as a number between 1 and 5)

            You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

            Now here is the instruction, the reference answer, and the response.

            Instruction: {format_input(entry)}
            Reference Answer: {entry['output']}
            Answer: {entry[json_key]}

            Provide your feedback. Respond with the integer number only.
            """
            # instruction = input 
            # reference answer = correct output
            # aswer = model response

        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores


def query_model(
    prompt,
    model="llama3.2",
    # If you used OLLAMA_HOST=127.0.0.1:11435 ollama serve
    # update the address from 11434 to 11435
    url="http://localhost:11434/api/chat"
):
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {     # Settings below are required for deterministic responses
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    

    # Send the POST request
    with requests.post(url, json=data, stream=True, timeout=30) as r:
        r.raise_for_status()
        response_data = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            response_json = json.loads(line)
            if "message" in response_json:
                response_data += response_json["message"]["content"]

    return response_data




def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running



def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100, # default value for the ignore index (in the cross-entropy loss)
    allowed_max_length=None,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def custom_collate_draft_2(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs to tensor and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def custom_collate_draft_1(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    # Find the longest sequence in the batch
    # and increase the max length by +1, which will add one extra
    # padding token below
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs
    inputs_lst = []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to batch_max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        # Via padded[:-1], we remove the extra padded token
        # that has been added via the +1 setting in batch_max_length
        # (the extra padding token will be relevant in later codes)
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)

    # Convert list of inputs to tensor and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    # Note: In the book, this was originally written as pos_emb.weight.shape[1] by mistake
    # It didn't break the code but would have caused unnecessary truncation (to 768 instead of 1024)

    # Truncate sequences if they too long
    input_ids = input_ids[:min(max_length, supported_context_length)]
    assert max_length is not None, (
        "max_length must be specified. If you want to use the full model context, "
        "pass max_length=model.pos_emb.weight.shape[0]."
    )
    assert max_length <= supported_context_length, (
        f"max_length ({max_length}) exceeds model's supported context length ({supported_context_length})."
    )    
    # Alternatively, a more robust version is the following one, which handles the max_length=None case better
    # max_len = min(max_length,supported_context_length) if max_length else supported_context_length
    # input_ids = input_ids[:max_len]
    
    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam"

# Same as chapter 5
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

# Overall the same as `train_model_simple` in chapter 5
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    # Initialize lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            examples_seen += input_batch.shape[0] # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen



# Chapter 6 version
# def calc_loss_loader(data_loader, model, device, num_batches=None):
#     total_loss = 0.
#     if len(data_loader) == 0:
#         return float("nan")
#     elif num_batches is None:
#         num_batches = len(data_loader)
#     else:
#         # Reduce the number of batches to match the total number of batches in the data loader
#         # if num_batches exceeds the number of batches in the data loader
#         num_batches = min(num_batches, len(data_loader))
#     for i, (input_batch, target_batch) in enumerate(data_loader):
#         if i < num_batches:
#             loss = calc_loss_batch(input_batch, target_batch, model, device)
#             total_loss += loss.item()
#         else:
#             break
#     return total_loss / num_batches


# Chapter 6 version
# def calc_loss_batch(input_batch, target_batch, model, device):
#     input_batch, target_batch = input_batch.to(device), target_batch.to(device)
#     logits = model(input_batch)[:, -1, :]  # Logits of last output token
#     loss = torch.nn.functional.cross_entropy(logits, target_batch)
#     return loss

# Chapter 6 version
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # Logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def random_split(df, train_frac, validation_frac):
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end] # for training the model
    validation_df = df[train_end:validation_end] # for tunning
    test_df = df[validation_end:] # for testing the model

    return train_df, validation_df, test_df


def create_balanced_dataset(df):
    # make sure that the non spam and spam messages are balanced (same number of samples)

    # Count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]
    
    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    
    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df



def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # Downloading the file
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(zip_path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)

    # Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")



def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
    


def assign(left, right):
    """
    Safely convert numpy array `right` into a torch.nn.Parameter matching
    dtype and device of `left`. Raises if shapes don't match.
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    # preserve dtype/device of `left`
    tensor = torch.tensor(right, dtype=left.dtype, device=left.device)
    return torch.nn.Parameter(tensor)



def generate(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature: float = 0.0,
    top_k: int = None,
    eos_id: int = None,
):
    """
    Robust batched generation supporting top-k and temperature.
    idx: (batch, n_tokens) LongTensor on any device
    Returns: idx with generated tokens appended
    """
    device = idx.device

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:].to(device)
        with torch.no_grad():
            logits = model(idx_cond)  # (batch, seq_len, vocab)
        logits = logits[:, -1, :]  # (batch, vocab)
        logits = logits.to(device)

        # Top-k filtering (per-batch)
        if top_k is not None and top_k > 0:
            # compute the topk threshold per batch-row
            topk_vals, _ = torch.topk(logits, k=top_k, dim=-1)
            min_topk = topk_vals[:, -1].unsqueeze(-1)  # (batch, 1)
            logits = torch.where(logits < min_topk, torch.tensor(float("-inf")).to(device), logits)

        if temperature == 0.0:
            # deterministic
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch,1)
        else:
            # scale and sample
            logits = logits - logits.max(dim=-1, keepdim=True).values
            scaled = logits / temperature
            probs = torch.softmax(scaled, dim=-1)
            # sample per batch row
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch,1)

        # eos handling: if eos_id provided, stop generation for those sequences that hit eos
        if eos_id is not None:
            # create mask of which batch members have already hit eos previously
            # we will append normally but then optionally stop further appends for finished sequences
            finished = (idx_next.squeeze(1) == eos_id)
            if finished.all():
                # all finished -> break early
                break
            # If some are finished, we still append eos for them; typical behavior is to still include eos then stop.
            # (Optionally you could mask to not extend those sequences further in subsequent steps.)

        idx = torch.cat((idx.to(device), idx_next.to(device)), dim=1)

    return idx



def softmax_with_temperature(logits, temperature):
    """
    logits: 1D or 2D tensor (if 2D, softmax along last dim).
    temperature: positive float (should not be 0).
    Returns probability tensor same shape as logits.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0 for softmax_with_temperature")
    # subtract max for numerical stability
    logits = logits - logits.max(dim=-1, keepdim=True).values
    scaled = logits / temperature
    return torch.softmax(scaled, dim=-1)

    

def print_sampled_tokens(probas):
    """
    probas: 1D tensor of probabilities summing to 1.
    Prints counts for a repeated sampling to show distribution.
    """
    if probas.dim() != 1:
        probas = probas.flatten()
    torch.manual_seed(123)
    vocab = {
        "closer": 0,
        "every": 1,
        "effort": 2,
        "forward": 3,
        "inches": 4,
        "moves": 5,
        "pizza": 6,
        "toward": 7,
        "you": 8,
    }
    inverse_vocab = {v: k for k, v in vocab.items()}
    # sample a bunch of times
    samples = torch.multinomial(probas, num_samples=1000, replacement=True)
    counts = torch.bincount(samples, minlength=len(probas)).tolist()
    for i, c in enumerate(counts):
        print(f"{c} x {inverse_vocab.get(i, str(i))}")


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


# Chapter 5 version
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = SimpleTokenizerV2.token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()



# Chapter 5 version
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

# Chapter 5 version
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())



def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor



def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Greedy generation (always argmax). idx: (batch, n_tokens)
    """
    device = idx.device
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:].to(device)
        with torch.no_grad():
            logits = model(idx_cond)  # (batch, seq_len, vocab)
        logits = logits[:, -1, :]  # (batch, vocab)
        # numeric stability
        logits = logits - logits.max(dim=-1, keepdim=True).values
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # greedy
        idx = torch.cat((idx, idx_next.to(device)), dim=1)
    return idx


if __name__ == "__main__":
    main()