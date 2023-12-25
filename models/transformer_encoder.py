import math
import torch
import numpy as np
import torch.nn as nn

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


d_model = 64
d_k = d_v = 64
d_ff = 512
n_layers = 3
n_heads = 3


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)
        output = self.fc(context)
        return nn.LayerNorm(d_model).cuda()(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, type_aware, code_num, type_num, code_att, visualization):
        super(Encoder, self).__init__()
        self.code_att = code_att
        self.type_aware = type_aware
        self.visualization = visualization
        self.src_emb = nn.Embedding(code_num + 1, d_model)
        self.typ0_emb = nn.Embedding(type_num + 1, d_model)
        self.typ1_emb = nn.Embedding(type_num + 1, d_model)
        self.typ2_emb = nn.Embedding(type_num + 1, d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, enc_inputs_x, enc_inputs_type0, enc_inputs_type1, enc_inputs_type2):
        if self.type_aware is True:
            enc_outputs = self.src_emb(enc_inputs_x) + self.typ0_emb(enc_inputs_type0)\
                        + self.typ1_emb(enc_inputs_type1) + self.typ2_emb(enc_inputs_type2)
        else:
            enc_outputs = self.src_emb(enc_inputs_x)

        enc_self_attn_mask = get_attn_pad_mask(enc_inputs_x, enc_inputs_x)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        if self.code_att is True:
            enc_outputs = enc_outputs.squeeze(0)
            alph = self.softmax(self.linear(enc_outputs)).T
            print(alph)
            enc_outputs = torch.matmul(alph, enc_outputs).squeeze(0)

        if self.visualization is True:
            diag_emb = enc_outputs.squeeze().cpu().detach().numpy()
            tsne = TSNE(n_components=2, init='pca', random_state=0)
            diag_tsne = tsne.fit_transform(diag_emb)
            plt.scatter(diag_tsne[:, 0], diag_tsne[:, 1])
            plt.show()

        return enc_outputs, enc_self_attns
