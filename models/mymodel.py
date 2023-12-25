import torch
from torch import nn
import torch.nn.functional as F

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from models.transformer_encoder import Encoder


class VisitEmbedding(nn.Module):

    def __init__(self, code_size, visit_size, attention_size, context_aware, visualization):
        super(VisitEmbedding, self).__init__()

        self.context_aware = context_aware
        self.visualization = visualization
        self.linear = nn.Linear(code_size, visit_size)
        self.softmax = nn.Softmax(dim=1)
        self.Q_linear = nn.Linear(code_size, attention_size, bias=False)
        self.K_linear = nn.Linear(code_size, attention_size, bias=False)
        self.V_linear = nn.Linear(code_size, visit_size, bias=False)

    def forward(self, v_emb_ini, lens):

        if self.context_aware is True:
            size = v_emb_ini.size()
            Q = self.Q_linear(v_emb_ini)
            K = self.K_linear(v_emb_ini).permute(0, 2, 1)  # 先进行一次转置
            V = self.V_linear(v_emb_ini)

            max_len = size[1]
            sentence_lengths = torch.Tensor(lens)  # 代表每个句子的长度
            mask = torch.arange(max_len).to(v_emb_ini.device)[None, :] < sentence_lengths[:, None]
            mask = mask.unsqueeze(dim=1)  # [batch_size, 1, max_len]
            mask = mask.expand(size[0], max_len, max_len)  # [batch_size, max_len, max_len]

            padding_num = torch.ones_like(mask)
            padding_num = -2 ** 31 * padding_num.float()

            alpha = torch.matmul(Q, K)
            alpha = torch.where(mask, alpha, padding_num)
            alpha = F.softmax(alpha, dim=2)
            v_emb = torch.matmul(alpha, V) + V

        else:
            v_emb = self.linear(v_emb_ini)

        if self.visualization is True:
            tsne = TSNE(n_components=2, init='pca', random_state=0)
            visit_tsne = tsne.fit_transform(v_emb.numpy())
            plt.scatter(visit_tsne[:, 0], visit_tsne[:, 1])
            plt.show()

        return v_emb


class MedRNN(nn.Module):
    def __init__(self, visit_size, time_size, hidden_size, visit_att, rnn_select, time_aware):
        super(MedRNN, self).__init__()
        self.hidden_size = hidden_size
        self.time_size = time_size
        self.rnn_select = rnn_select
        self.time_aware = time_aware
        if self.time_aware is False:
            time_size = 0
        self.time_embedding = nn.Linear(1, time_size)
        self.grucell = nn.GRUCell(input_size=visit_size+time_size, hidden_size=self.hidden_size)
        self.lstmcell = nn.LSTMCell(input_size=visit_size+time_size, hidden_size=self.hidden_size)
        self.rnncell = nn.RNNCell(input_size=visit_size+time_size, hidden_size=self.hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.weight_layer = nn.Softmax(dim=0)

    def forward(self, visit_emb, lens, intervals):
        hidden = torch.zeros(self.hidden_size).to(visit_emb.device)
        if self.rnn_select == 'LSTM':
            cell = torch.zeros(self.hidden_size).to(visit_emb.device)
        out = []
        for visit_emb_i, len, interval in zip(visit_emb, lens, intervals):
            for visit_emb_it, t in zip(visit_emb_i, range(len)):
                if self.time_aware is True:
                    t_emb = self.time_embedding(interval[t].reshape(1, 1))
                    input = torch.cat((visit_emb_it, t_emb.reshape(self.time_size)))
                else:
                    input = visit_emb_it

                if self.rnn_select == 'GRU':
                    hidden = self.grucell(input, hidden)
                elif self.rnn_select == 'LSTM':
                    hidden, cell = self.lstmcell(input, (hidden, cell))
                elif self.rnn_select == 'RNN':
                    hidden = self.rnncell(input, hidden)
            out.append(hidden)
        out = torch.vstack(out)
        return out


class Classifier(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_rate=0.):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(hidden_size, output_size)
        self.activate = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, hidden):
        output = self.dropout(hidden)
        output = self.linear(output)
        output = self.activate(output)
        return output


class MyModel(nn.Module):
    def __init__(self, code_num, type_num, code_size, attention_size, visit_size, time_size, hidden_size, batch_size,
                 output_size, dropout_rate, code_att, visit_att, rnn_select, type_aware, time_aware, context_aware,
                 visualization):
        super().__init__()
        self.code_att = code_att
        self.code_num = code_num
        self.code_size = code_size
        self.batch_size = batch_size
        self.encoder = Encoder(type_aware, code_num, type_num, code_att, visualization)
        self.weight_code_layer = nn.Linear(self.code_size, 1)
        self.softmax = nn.Softmax(dim=0)
        self.visit_embedding = VisitEmbedding(self.code_size, visit_size, attention_size, context_aware, visualization)
        self.gru = MedRNN(visit_size, time_size, hidden_size, visit_att, rnn_select, time_aware)
        self.classifier = Classifier(hidden_size, output_size=output_size, dropout_rate=dropout_rate)

    def forward(self, code_x, code_type_class, lens, intervals):
        code_type0, code_type1, code_type2 = code_type_class[0], code_type_class[1], code_type_class[2]
        shape = list(code_x.size())
        shape[-1] = self.code_size
        v_emb_ini = torch.zeros(shape).to(code_x.device)
        for pid, (code_x_i, len_i) in enumerate(zip(code_x, lens)):
            for cx_it, len_it in zip(code_x_i, range(len_i)):
                codes = torch.where(cx_it > 0)[0]+1
                cx_it = codes.long().unsqueeze(0)
                ct0_it = code_type0[codes].long().unsqueeze(0)
                ct1_it = code_type1[codes].long().unsqueeze(0)
                ct2_it = code_type2[codes].long().unsqueeze(0)
                c_emb = self.encoder(cx_it, ct0_it, ct1_it, ct2_it)[0].squeeze(0)
                if self.code_att is True:
                    v_emb_ini[pid][len_it] = c_emb
                else:
                    v_emb_ini[pid][len_it] = torch.mean(c_emb, dim=0)
        v_emb = self.visit_embedding(v_emb_ini, lens)
        h_n = self.gru(v_emb, lens, intervals)
        output = self.classifier(h_n)
        return output
