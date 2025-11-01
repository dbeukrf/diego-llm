from torch._tensor import Tensor
from gptdownload import download_and_load_gpt2
# Alternatively:
# from llms_from_scratch.ch05 import download_and_load_gpt2


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
import numpy as np


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




def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


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