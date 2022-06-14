import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import *
from ..constants import INFTY
from .common import *


def _transform_add_inputs(n, add_inputs):
    def _get_dim(inp):
        if inp['tag'] == 'amatrix':
            return n * n
        else:
            return inp.get('dim', n)

    return [(inp['tag'], _get_dim(inp)) for inp in add_inputs]


class GlobalQNetwork(SaveableModel):
    def __init__(self, n, layers, activation, additional_inputs=[],
                 embedding_dim=None, embedding_shift=False,
                 one_out=True, scope='', with_attn=True, **kwargs):

        if embedding_dim is not None and not one_out:
            raise Exception('Embedding-using networks are one-out only!')

        super().__init__()
        self.graph_size = n
        self.add_inputs = _transform_add_inputs(n, additional_inputs)
        self.uses_embedding = embedding_dim is not None
        self.embedding_shift = embedding_shift
        self.embedding_dim = embedding_dim
        self.one_out = one_out

        input_dim = sum([d for (_, d) in self.add_inputs])
        if not self.uses_embedding:
            input_dim += 3 * n
            self.original_input_dim = 3 * n
        else:
            mult = 2 if self.embedding_shift else 3
            input_dim += mult * embedding_dim
            self.original_input_dim = mult * embedding_dim

        output_dim = 1 if one_out else n

        self._scope = scope if len(scope) > 0 else None
        self._label = 'global_qnetwork{}{}_{}_{}_{}_{}_{}'.format(
            '-oneinp' if one_out else '',
            '-emb-{}'.format(embedding_dim) if self.uses_embedding else '',
            input_dim,
            '-'.join(map(str, layers)),
            output_dim,
            activation,
            '_'.join(map(lambda p: p[0]+'-'+str(p[1]), self.add_inputs)))

        attn_module_dim = embedding_dim if embedding_dim is not None else n
        self.emb_pre_norm = Norm(attn_module_dim)
        self.emb_attn = MultiHeadAttention(attn_module_dim, 4)
        self.emb_post_norm = Norm(attn_module_dim)

        self.amatrix_pre_norm = Norm(self.graph_size)
        self.amatrix_attn = MultiHeadAttention(self.graph_size, 5)
        self.amatrix_post_norm = Norm(self.graph_size)

        self.amatrix_ff_net = FFNetwork(self.graph_size * self.graph_size, embedding_dim, layers=[64, 64], activation=activation)

        self.main_pre_norm = Norm(attn_module_dim)
        self.main_attn = MultiHeadAttention(embedding_dim, 4)
        self.main_post_norm = Norm(attn_module_dim)

        self.with_attn = with_attn

        self.ff_net = FFNetwork(input_dim, output_dim, layers=layers, activation=activation)
        self.layers = layers

    def init_xavier(self):
        self.emb_attn.apply(xavier_init)
        self.amatrix_attn.apply(xavier_init)
        self.ff_net.apply(xavier_init)

    def forward(self, addr, dst, neighbour, *others):
        if self.uses_embedding:
            addr_ = atleast_dim(addr, 2)
            dst_ = atleast_dim(dst, 2)
            neighbour_ = atleast_dim(neighbour, 2)

            if self.embedding_shift:
                # re-center embeddings linearly against origin
                input_tensors = [dst_ - addr_, neighbour_ - addr_]
            else:
                input_tensors = [addr_, dst_, neighbour_]
        else:
            addr_ = one_hot(atleast_dim(addr, 1), self.graph_size)
            dst_ = one_hot(atleast_dim(dst, 1), self.graph_size)

            if self.one_out:
                neighbour_ = one_hot(atleast_dim(neighbour, 1), self.graph_size)
            else:
                neighbour_ = atleast_dim(neighbour, 2)
                # neighbour_ = one_hot(neighbour, self.graph_size)
                # neighbour_ = torch.sum(neighbour_, dim=1)

            input_tensors = [addr_, dst_, neighbour_]

        # print(input_tensors[0].size())
        # print(input_tensors[1].size())
        # print(input_tensors[2].size())
        ff_input = (self.emb_attention(input_tensors), ) if self.with_attn else input_tensors

        batch_dim = input_tensors[0].size()[0]

        need_cat = True

        for ((tag, dim), inp) in zip(self.add_inputs, others):
            inp = atleast_dim(inp, 2)
            if inp.size()[0] != batch_dim:
                inp = inp.transpose(0, 1)

            if inp.size()[1] != dim:
                raise Exception(f'Wrong `{tag}` input dimension: expected {dim}, actual {inp.size()[1]}')

            if tag == 'amatrix':
                if self.with_attn:
                    amatrix_conv = self.amatrix_ff_net(torch.flatten(inp, start_dim=1))
                    # print(ff_input[0].size())
                    # print(inp.size())
                    # print(amatrix_conv.size())
                    input_with_amatrix = torch.cat(ff_input + (amatrix_conv,), dim=1)
                    input_attn_values = torch.split(input_with_amatrix, self.embedding_dim, dim=1)
                    # print(input_with_amatrix.size())
                    # print(len(input_attn_values))
                    # print(input_attn_values[0].size())
                    main_attn = self.main_attention(input_attn_values)
                    # print(main_attn.size())
                    ff_input = main_attn
                    need_cat = False
                    continue
                    amatrix_attn = self.amatrix_attention(inp)
                    ff_input = ff_input + (amatrix_attn,)
                else:
                    ff_input.append(torch.flatten(inp, start_dim=1))
            # else:
            #     input_tensors.append(inp)

        if need_cat:
            ff_input = torch.cat(ff_input, dim=1)
        output = self.ff_net(ff_input)

        if not self.one_out:
            inf_mask = torch.mul(torch.add(neighbour_, -1), INFTY)
            output = torch.add(output, inf_mask)

        return output

    def emb_attention(self, embs):
        input_tensors = torch.stack(embs)
        input_tensors = torch.transpose(input_tensors, 0, 1)

        pre_norm = self.emb_pre_norm(input_tensors)
        # pre_norm = input_tensors
        attn = self.emb_attn(pre_norm, pre_norm, pre_norm)
        post_norm = self.emb_post_norm(attn)
        # post_norm = attn
        output = post_norm.view(-1, self.original_input_dim)
        # print('attn:', post_norm.size(), output.size(), self.original_input_dim, self.embedding_dim)
        return output

    def amatrix_attention(self, amatrixes):
        # return amatrixes
        amatrixes_2d = amatrixes.view(-1, self.graph_size, self.graph_size)

        pre_norm = self.amatrix_pre_norm(amatrixes_2d)
        # pre_norm = amatrixes_2d
        attn = self.amatrix_attn(pre_norm, pre_norm, pre_norm)
        post_norm = self.amatrix_post_norm(attn)
        # post_norm = attn

        output = torch.flatten(post_norm, start_dim=1)
        return output

    def main_attention(self, embs):
        input_tensors = torch.stack(embs)
        # print('main:', input_tensors.size())
        input_tensors = torch.transpose(input_tensors, 0, 1)
        # print('main:', input_tensors.size())

        pre_norm = self.main_pre_norm(input_tensors)
        # pre_norm = input_tensors
        attn = self.main_attn(pre_norm, pre_norm, pre_norm)
        post_norm = self.main_post_norm(attn)
        # post_norm = attn

        output = post_norm.view(-1, self.original_input_dim)
        # print('attn:', post_norm.size(), output.size(), self.original_input_dim)
        return output

    def change_label(self, new_label_value):
        self._label = new_label_value

    def get_title(self):
        with_self_attn = 'Self-attention,' if self.with_attn else ''
        return f"{with_self_attn}\nff layers: {self.layers}"
