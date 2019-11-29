import torch.nn as nn
import torch


class PrepareState(nn.Module):
    def __init__(self, cell_type,  # 编解码器rnn类型
                 input_size,  # 输入状态的维度
                 output_size):  # 输出状态的维度
        super(PrepareState, self).__init__()
        assert cell_type in ['GRU', 'LSTM']

        if cell_type == 'LSTM':
            self.linear_c = nn.Linear(input_size, output_size)
        else:
            self.linear_h = nn.Linear(input_size, output_size)

        self.cell_type = cell_type

    def forward(self, state_encoder, state_affect):
        if self.cell_type == 'LSTM':
            state_eh, state_ec = state_encoder  # [layer, batch, dim]
            state_ah, state_ac = state_affect
            state_h = self.linear_h(torch.cat([state_eh, state_ah], 2))
            state_c = self.linear_c(torch.cat([state_ec, state_ac], 2))
            return tuple(state_h, state_c)
        else:
            state_h = self.linear_h(torch.cat([state_encoder, state_affect], 2))
            return state_h


