import torch
import torch.nn as nn
import numpy as np

# For convinence
from torch import Tensor


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.scale = 1.0 / np.sqrt(self.d_model)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: Tensor):
        assert mask.shape[0] == Q.shape[-2], f"Mask size ({mask.shape[0]}) must be equal to sequence length ({Q.shape[-2]})"
        
        output = self.softmax(torch.bmm(Q.float(), K.float().transpose(1, 2)).mul_(self.scale))
        output = output.mul_(mask).div_(output.sum(dim=-1, keepdim=True))
        output = torch.bmm(output, V.float())
        return output
        