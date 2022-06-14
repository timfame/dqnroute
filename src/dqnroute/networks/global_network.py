import torch

from .common import *


class GlobalNetwork(SaveableModel):
    def __init__(
            self,
            layers: List[int],
            activation: str = 'relu',
            embedding_dim: int = None,
            embedding_shift: bool = True,
            scope='',
            optimizer='adam',
            dropout=0.1,
    ):

        super(GlobalNetwork, self).__init__()

        self.embedding_shift = False

        # Input = (addr_emb, dst_emb, nb1_emb, nb2_emb, nb3_emb) or shifted by addr_emb
        # Output = (predicted_next_emb)
        input_cnt = 4 if self.embedding_shift else 5
        input_dim = input_cnt * embedding_dim
        output_dim = embedding_dim
        self.input_cnt = input_cnt
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim

        self.norm_1 = Norm(embedding_dim)
        self.norm_2 = Norm(embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, 5)
        self.attn_ff = AttentionFeedForward(embedding_dim)

        self.ff_net = FFNetwork(input_dim, output_dim, layers, activation)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self._scope = scope if len(scope) > 0 else None
        self._label = None

        self.optimizer_was_init = False
        self.optimizer_type = optimizer
        self.optimizer = None

    def forward(self, addr, dst, nb1, nb2, nb3):
        addr_ = atleast_dim(addr, 2)
        dst_ = atleast_dim(dst, 2)
        nb1_ = atleast_dim(nb1, 2)
        nb2_ = atleast_dim(nb2, 2)
        nb3_ = atleast_dim(nb3, 2)

        if self.embedding_shift:
            dst_ = dst_ - addr_
            nb1_ = nb1_ - addr_
            nb2_ = nb2_ - addr_
            nb3_ = nb3_ - addr_
            input_tensors = [dst_, nb1_, nb2_, nb3_]
        else:
            input_tensors = [addr_, dst_, nb1_, nb2_, nb3_]
        # input_tensors = torch.cat(input_tensors, dim=1)
        input_tensors = torch.stack(input_tensors)
        input_tensors = torch.transpose(input_tensors, 0, 1)

        norm = self.norm_1(input_tensors)
        attention = self.attention(norm, norm, norm)
        # input_attn = self.dropout_1(attention)
        input_attn = attention

        attn_norm = self.norm_2(input_attn)
        # attn_ff = attn_norm
        # attn_ff = attn_norm
        attn_ff = self.attn_ff(attn_norm)
        attn = self.dropout_2(attn_ff)
        # attn = attn_ff

        attn = attn.view(-1, self.input_dim)

        outputs = self.ff_net(attn)
        # outputs = torch.nan_to_num(outputs)

        if self.embedding_shift:
            outputs = outputs + addr_

        # if self._label is not None:
        #     print('addr_:', addr_ )
        #     print('input:', input_tensors)
        #     print('norm:', norm)
        #     print('attention:', attention)
        #     print('input_attn:', input_attn)
        #     print('ff:', ff)
        #     print('outputs:', outputs)
        return outputs

    def init_xavier(self):
        print('INIT XAVIER')
        self.ff_net.apply(xavier_init)
        self.attention.apply(xavier_init)
        self.attn_ff.apply(xavier_init)

    def change_label(self, new_label):
        self._label = new_label

    def init_optimizer(self, params):
        if not self.optimizer_was_init:
            self.optimizer_was_init = True
            self.optimizer = get_optimizer(self.optimizer_type)(params)


class AttentionFeedForward(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x