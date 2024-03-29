import torch
import torch.nn as nn
from Embedding import WordEmbedding
from Encoder import Encoder
from Decoder import Decoder


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        self.config = config

        # 定义嵌入层
        self.embedding = WordEmbedding(config.num_vocab,  # 词汇表大小
                                       config.embedding_size,  # 嵌入层维度
                                       config.pad_id)  # pad_id

        # 编码器
        self.encoder = Encoder(config.encoder_decoder_cell_type,  # rnn类型
                               config.embedding_size,  # 输入维度
                               config.encoder_decoder_output_size,  # 输出维度
                               config.encoder_decoder_num_layers,  # rnn层数
                               config.encoder_bidirectional,  # 是否双向
                               config.dropout)  # dropout概率

        # 解码器
        self.decoder = Decoder(config.encoder_decoder_cell_type,  # rnn类型
                               config.embedding_size,  # 输入维度
                               config.encoder_decoder_output_size,  # 输出维度
                               config.encoder_decoder_num_layers,  # rnn层数
                               config.dropout)  # dropout概率

        # 输出层
        self.projector = nn.Sequential(
            nn.Linear(config.encoder_decoder_output_size, config.num_vocab),
            nn.Softmax(-1)
        )

    def forward(self, input,
                inference=False,  # 是否测试
                max_len=60):  # 解码的最大长度

        if not inference:  # 训练

            id_posts = input['posts']  # [batch, seq]
            len_posts = input['len_posts']  # [batch]
            id_responses = input['responses']  # [batch, seq]

            embed_posts = self.embedding(id_posts)  # [batch, seq, embed_size]
            embed_responses = self.embedding(id_responses)  # [batch, seq, embed_size]

            # 解码器的输入为回复去掉end_id
            decoder_input = embed_responses[:, :-1, :].transpose(0, 1)  # [seq-1, batch, embed_size]
            len_decoder = decoder_input.size()[0]  # 解码长度 seq-1
            decoder_input = decoder_input.split([1] * len_decoder, 0)  # 解码器每一步的输入 seq-1个[1, batch, embed_size]

            # state = [layers, batch, dim]
            _, state_encoder = self.encoder(embed_posts.transpose(0, 1), len_posts)

            outputs = []

            for idx in range(len_decoder):

                if idx == 0:
                    state = state_encoder  # 解码器初始状态

                input = decoder_input[idx]  # 当前时间步输入 [1, batch, embed_size]

                # output: [1, batch, dim_out]
                # state: [num_layer, batch, dim_out]
                output, state = self.decoder(input, state)

                outputs.append(output)

            outputs = torch.cat(outputs, 0).transpose(0, 1)  # [batch, seq-1, dim_out]

            output_vocab = self.projector(outputs)  # [batch, seq-1, num_vocab]

            return output_vocab

        else:  # 测试

            id_posts = input['posts']  # [batch, seq]
            len_posts = input['len_posts']  # [batch]
            batch_size = id_posts.size()[0]
            device = id_posts.device.type

            embed_posts = self.embedding(id_posts)  # [batch, seq, embed_size]

            # state = [layers, batch, dim]
            _, state_encoder = self.encoder(embed_posts.transpose(0, 1), len_posts)

            outputs = []

            done = torch.BoolTensor([0] * batch_size)
            first_input_id = (torch.ones((1, batch_size)) * self.config.start_id).long()
            if device == 'cuda':
                done = done.cuda()
                first_input_id = first_input_id.cuda()

            for idx in range(max_len):

                if idx == 0:  # 第一个时间步
                    state = state_encoder  # 解码器初始状态
                    input = self.embedding(first_input_id)  # 解码器初始输入 [1, batch, embed_size]

                # output: [1, batch, dim_out]
                # state: [num_layers, batch, dim_out]
                output, state = self.decoder(input, state)

                outputs.append(output)

                vocab_prob = self.projector(output)  # [1, batch, num_vocab]
                next_input_id = torch.argmax(vocab_prob, 2)  # 选择概率最大的词作为下个时间步的输入 [1, batch]

                _done = next_input_id.squeeze(0) == self.config.end_id  # 当前时间步完成解码的 [batch]
                done = done | _done  # 所有完成解码的

                if done.sum() == batch_size:  # 如果全部解码完成则提前停止

                    break

                else:

                    input = self.embedding(next_input_id)  # [1, batch, embed_size]

            outputs = torch.cat(outputs, 0).transpose(0, 1)  # [batch, seq, dim_out]

            output_vocab = self.projector(outputs)  # [batch, seq, num_vocab]

            return output_vocab

    # 统计参数
    def print_parameters(self):

        def statistic_param(params):
            total_num = 0  # 参数总数
            for param in params:
                num = 1
                if param.requires_grad:
                    size = param.size()
                    for dim in size:
                        num *= dim
                total_num += num
            return total_num

        print("嵌入层参数个数: %d" % statistic_param(self.embedding.parameters()))
        print("编码器参数个数: %d" % statistic_param(self.encoder.parameters()))
        print("解码器参数个数: %d" % statistic_param(self.decoder.parameters()))
        print("输出层参数个数: %d" % statistic_param(self.projector.parameters()))
        print("参数总数: %d" % statistic_param(self.parameters()))

    # 保存模型
    def save_model(self, epoch, global_step, path):

        torch.save({'embedding': self.embedding.state_dict(),
                    'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'projector': self.projector.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step}, path)

    # 载入模型
    def load_model(self, path):

        checkpoint = torch.load(path)
        self.embedding.load_state_dict(checkpoint['embedding'])
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.projector.load_state_dict(checkpoint['projector'])
        epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']

        return epoch, global_step







