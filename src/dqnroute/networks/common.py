import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from typing import List
from functools import partial
from ..constants import TORCH_MODELS_DIR


def get_activation(name):
    if type(name) != str:
        return name
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise Exception('Unknown activation function: ' + name)


def get_optimizer(name, params={}):
    if isinstance(name, optim.Optimizer):
        return name
    if name == 'rmsprop':
        return partial(optim.RMSprop, **dict(params, lr=0.001))
    elif name == 'adam':
        return partial(optim.Adam,  **dict(params, lr=0.0005))
    elif name == 'adadelta':
        return partial(optim.Adadelta, **params)
    elif name == 'adagrad':
        return partial(optim.Adagrad, **dict(params, lr=0.001))
    else:
        raise Exception('Invalid optimizer: ' + str(name))


def get_distance_function(name):
    if type(name) != str:
        return name
    if name == 'euclid':
        return euclidean_distance
    elif name == 'linear':
        pass
    elif name == 'cosine':
        pass
    else:
        raise Exception(f'Invalid distance function: {str(name)}')


def atleast_dim(x: torch.Tensor, dim: int, axis=0) -> torch.Tensor:
    while x.dim() < dim:
        x = x.unsqueeze(axis)
    return x


def one_hot(indices, dim) -> torch.Tensor:
    """
    Creates a one-hot tensor from a tensor of integers.
    indices.size() should be [seq,batch] or [batch,]
    result size() would be [seq,batch,dim] or [batch,dim]
    """
    out = torch.zeros(indices.size() + torch.Size([dim]))
    d = len(indices.size())
    return out.scatter_(d, indices.unsqueeze(d).to(dtype=torch.int64), 1)


def xavier_init(m: nn.Module):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


class FFNetwork(nn.Sequential):
    """
    Simple feed-forward network with fully connected layers
    """

    def __init__(self, input_dim: int, output_dim: int, layers: List[int], activation='relu'):
        super().__init__()
        act_module = get_activation(activation)

        prev_dim = input_dim
        for i, layer in enumerate(layers):
            if type(layer) == int:
                lsize = layer
                self.add_module(f'fc{i + 1}', nn.Linear(prev_dim, lsize))
                self.add_module(f'activation{i + 1}', act_module)
                prev_dim = lsize
            elif layer == 'dropout':
                self.add_module(f'dropout_{i + 1}', nn.Dropout(0.001))

        self.add_module('output', nn.Linear(prev_dim, output_dim))


class SaveableModel(nn.Module):
    """
    Mixin which provides `save` and `restore`
    methods for (de)serializing the model.
    """

    def _savedir(self):
        dir = TORCH_MODELS_DIR
        if self._scope is not None:
            dir += '/' + self._scope
        return dir

    def _savepath(self):
        return self._savedir() + '/' + self._label

    def save(self):
        print('SAVE, path:', self._savepath())
        os.makedirs(self._savedir(), exist_ok=True)
        return torch.save(self.state_dict(), self._savepath())

    def restore(self):
        print('RESTORE, path:', self._savepath())
        return self.load_state_dict(torch.load(self._savepath()))


# distance functions
def euclidean_distance(allowed_neighbours: torch.Tensor, existing_state: torch.Tensor):
    sum = torch.sum((allowed_neighbours - existing_state) ** 2, dim=1)
    return torch.sqrt(sum)


def linear_distance(existing_state: torch.Tensor, predicted_state: torch.Tensor):
    pass  # TODO Implement


def cosine_distance(existing_state, predicted_state):
    pass  # TODO Implement


class Norm(nn.Module):

    def __init__(self, input_dim, eps=1e-6):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        norm = self.alpha * (x - mean) / (std + self.eps)
        output = norm + self.bias
        return output


class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, heads, dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.d_k = input_dim // heads
        self.heads = heads

        self.q_linear = nn.Linear(input_dim, input_dim)
        self.v_linear = nn.Linear(input_dim, input_dim)
        self.k_linear = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(input_dim, input_dim)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.heads, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.heads, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.heads, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.input_dim)

        output = self.out(concat)
        return output

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)
        return output
