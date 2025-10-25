from typing import Any

import os
import requests
import re
from gptdataset import GPTDatasetV1
from tokenizer import SimpleTokenizerV2
import torch


def main():
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    max_length = 4
    dataloader = GPTDatasetV1.create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=4, shuffle=False)

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)

    vocab_size = 50257
    output_dim = 256

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)

    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    # print(pos_embedding_layer.weight)

    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    # print(pos_embeddings.shape)
    # print(pos_embeddings)


    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)
    print(input_embeddings)


if __name__ == "__main__":
    main()