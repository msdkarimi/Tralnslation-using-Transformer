from encoder import *


class Encoder(nn.Module):
    def __init__(self, encoder_input_vocab_size: int, embedding_size: int, ff_hidden_layer: int, head: int, dropout: float = None, N=6):
        super(Encoder, self).__init__()

        self.N = N
        self.pe = PositionalEmbedding(dictionary_size=encoder_input_vocab_size, embedding_size=embedding_size)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(embedding_size=embedding_size, ff_hidden_layer=ff_hidden_layer, head=head, dropout=dropout)
            for _ in range(self.N)
            ])

    def forward(self, x, mask):
        x = self.pe(x)
        for an_encoder in self.encoder_blocks:
            x = an_encoder(x, mask)
        return x
