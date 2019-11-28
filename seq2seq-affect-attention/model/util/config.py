"""
    模型参数的类
"""


class Config(object):

    # 这些不需要改，由数据的预处理决定了，如果改预处理再改这里
    pad_id = 0
    start_id = 1
    end_id = 2
    unk_id = 3

    # 词汇表大小，根据预处理截取的词汇表设置
    num_vocab = 39000

    # 嵌入层参数，如果载入预训练的词向量，就由词向量的维度决定
    affect_embedding_size = 3
    embedding_size = 300

    affect_encoder_output_size = 100

    # 编码器解码器层数/输出大小
    encoder_decoder_cell_type = 'LSTM'  # in ['GRU', 'LSTM']
    encoder_decoder_num_layers = 2
    encoder_decoder_output_size = 300

    decoder_input_size = 300

    # 编码器参数
    encoder_bidirectional = True  # 是否是双向rnn

    attention_type = 'dot'  # in ['dot', 'general', 'concat', 'perceptron']
    attention_size = 100

    # 优化参数
    batch_size = 64
    method = 'adam'  # in ['sgd', 'adam']
    lr = 0.0001  # 初始学习率
    lr_decay = 1.0  # 学习率衰减，每过1个epoch衰减的百分比
    weight_decay = 0  # 权值decay
    max_grad_norm = 5
    dropout = 0  # 这里只有编解码器设置了dropout
