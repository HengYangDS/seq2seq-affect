import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, decoder_output_size,
                 encoder_output_size,
                 attention_type,
                 attention_size):
        super(Attention, self).__init__()
        assert attention_type in ['dot', 'general', 'concat', 'perceptron']  # 点乘、权重、拼接权重、感知机

        self.linear_q = nn.Sequential(nn.Linear(decoder_output_size, attention_size), nn.ReLU())
        self.linear_k = nn.Sequential(nn.Linear(encoder_output_size, attention_size), nn.ReLU())
        self.linear_v = nn.Sequential(nn.Linear(encoder_output_size, attention_size), nn.ReLU())

        if attention_type == 'general':
            self.general_w = nn.Linear(attention_size, attention_size, bias=False)
        elif attention_type == 'concat':
            self.concat_w = nn.Linear(attention_size*2, 1, bias=False)
        elif attention_type == 'perceptron':
            self.perceptron_w = nn.Linear(attention_size, attention_size, bias=False)
            self.perceptron_u = nn.Linear(attention_size, attention_size, bias=False)
            self.perceptron_v = nn.Linear(attention_size, 1, bias=False)

        self.attention_type = attention_type
        self.attention_size = attention_size

    def forward(self, decoder_outputs,  # [batch_size, decoder_len, output_size]
                encoder_outputs):  # [batch_size, encoder_len, output_size]

        querys = self.linear_q(decoder_outputs)  # [batch_size, decoder_len, attention_size]
        keys = self.linear_k(encoder_outputs)  # [batch_size, encoder_len, attention_size]
        values = self.linear_v(encoder_outputs)  # [batch_size, encoder_len, attention_size]

        if self.attention_type == 'dot':  # Q^TK

            weights = querys.bmm(keys.transpose(1, 2))  # [batch_size, decoder_len, encoder_len]
            weights = weights / self.attention_size ** 0.5
            weights = weights.softmax(-1)
            attention = weights.bmm(values)  # [batch_size, decoder_len, attention_size]

        elif self.attention_type == 'general':  # Q^TWK

            keys = self.general_w(keys)  # [batch_size, encoder_len, attention_size]
            weights = querys.bmm(keys.transpose(1, 2))  # [batch_size, decoder_len, encoder_len]
            weights = weights / self.attention_size ** 0.5
            weights = weights.softmax(-1)
            attention = weights.bmm(values)  # [batch_size, decoder_len, attention_size]

        elif self.attention_type == 'concat':  # W[Q^T;K]

            decoder_len = querys.size()[1]
            encoder_len = keys.size()[1]

            # [batch_size, decoder_len, encoder_len, attention_size]
            querys = querys.unsqueeze(2).repeat(1, 1, encoder_len, 1)
            keys = keys.unsqueeze(1).repeat(1, decoder_len, 1, 1)
            weights = self.concat_w(torch.cat([querys, keys], 3)).squeeze(-1)  # [batch_size, decoder_len, encoder_len]
            weights = weights / self.attention_size ** 0.5
            weights = weights.softmax(-1)
            attention = weights.bmm(values)  # [batch_size, decoder_len, attention_size]

        else:  # V^Ttanh(WQ+UK)

            decoder_len = querys.size()[1]
            encoder_len = keys.size()[1]
            # [batch_size, decoder_len, encoder_len, attention_size]
            querys = querys.unsqueeze(2).repeat(1, 1, encoder_len, 1)
            keys = keys.unsqueeze(1).repeat(1, decoder_len, 1, 1)

            querys = self.perceptron_w(querys)
            keys = self.perceptron_u(keys)
            weights = self.perceptron_v(querys+keys).squeeze(-1)
            weights = weights / self.attention_size ** 0.5
            weights = weights.softmax(-1)
            attention = weights.bmm(values)  # [batch_size, decoder_len, attention_size]

        # attention: [batch_size, decoder_len, attention_size]
        # weights: [batch_size, decoder_len, encoder_len]
        return attention, weights
