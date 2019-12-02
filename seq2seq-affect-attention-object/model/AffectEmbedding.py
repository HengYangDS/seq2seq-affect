import torch.nn as nn


class AffectEmbedding(nn.Module):

    def __init__(self, num_vocab,
                 embedding_size,
                 pad_id=0):
        super(AffectEmbedding, self).__init__()

        self.embedding = nn.Embedding(num_vocab, embedding_size, padding_idx=pad_id)
        self.embedding.weight.requires_grad = False

    def forward(self, input):  # [batch, seq]

        embeded = self.embedding(input)  # [batch, seq, embedding_size]

        return embeded
