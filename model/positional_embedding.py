from model import *
import matplotlib.pyplot as plt


class PositionalEmbedding(nn.Module):
    # TODO( define max_len in config)
    def __init__(self, dictionary_size: int, embedding_size: int, max_len: int, dropout=None, base=1e4):
        super(PositionalEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.embedding = Embedding(dictionary_size=dictionary_size, embedding_size=embedding_size)
        self.dropout = nn.Dropout(dropout) if dropout is not None else dropout
        self.base = base
        self.max_len = max_len

        self.freq = self.frequency()
        # buffers are saved in state_dict but not trained by the optimizer
        self.register_buffer('positional_embedding', self.freq)

    def forward(self, input_token: str):

        # token_seq = input_token
        # print(token_seq)
        # print(token_seq.shape)
        token_seq = input_token  # torch.Size([16, 512])
        embedding = self.embedding(token_seq)  # torch.Size([16, 512, 64])
        # print(embedding.shape)
        # print(embedding)
        # exit()
        input_token_pe = embedding + self.freq[:, : embedding.shape[1], :].requires_grad_(False)
        return self.dropout(input_token_pe) if self.dropout is not None else input_token_pe

    def frequency(self):
        frequency = torch.zeros(self.max_len, self.embedding_size)
        div_term = torch.exp(torch.arange(0, self.embedding_size, 2) * -(np.log(self.base) / self.embedding_size))

        positions = torch.arange(0, self.max_len).unsqueeze(1)
        frequency[:, 0::2] = torch.sin(positions * div_term)
        frequency[:, 1::2] = torch.cos(positions * div_term)
        frequency = frequency.unsqueeze(0)
        return frequency

    def visualize_pe(self):
        plt.imshow(self.frequency().squeeze(0), aspect="auto")
        plt.title("Positional Embedding")
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Position Index")

        # set the tick marks for the axes
        if self.embedding_size < 10:
            plt.xticks(torch.arange(0, self.embedding_size))
        if self.max_len < 20:
            plt.yticks(torch.arange(self.max_len - 1, -1, -1))

        plt.colorbar()
        plt.savefig('positional embedding.png')
        # plt.show()


class Embedding(nn.Module):
    def __init__(self, dictionary_size: int, embedding_size: int):
        super(Embedding, self).__init__()
        self.embedding_size = embedding_size
        self.embedding_layer = nn.Embedding(dictionary_size, embedding_size)

    def forward(self, x: torch.Tensor):
        return self.embedding_layer(x) * np.sqrt(self.embedding_size)
