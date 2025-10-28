import torch
import torch.nn as nn
from gelu import GELU

# help the neural network extract more complex patterns from the data
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), # linear layer with 4 times the embedding dimension (768 * 4 = 3072)
            GELU(), # gelu activation function
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]) # linear layer with the embedding dimension (3072 -> 768)
        )
    
    def forward(self, x):
        return self.layers(x)
