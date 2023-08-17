import torch

# Importing attention module
from attentions import ScaledDotProductAttention

input = torch.rand((1, 10, 64))
mask = torch.ones((10))
attention = ScaledDotProductAttention(512)

output = attention(input, input, input, mask) # Q, K, V, Mask

print(output.shape)