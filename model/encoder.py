from model import *


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


class EncoderBlock(nn.Module):
    def __init__(self, embedding_size: int, ff_hidden_layer: int, head: int, dropout: float = None, mask=None):
        super(EncoderBlock, self).__init__()
        self.mhsa = MultiHeadSelfAttention(embedding_size=embedding_size, dropout=dropout, head=head)
        self.ffn = FeedForward(embedding_size=embedding_size, ff_hidden_layer=ff_hidden_layer, dropout=dropout)
        self.add_and_norm_mhsa = AddAndNorm(embedding_size=embedding_size)
        self.add_and_norm_ffn = AddAndNorm(embedding_size=embedding_size)

        self.mask = mask

    def forward(self, x, mask):
        output_mhsa = self.mhsa(x, x, x, mask)
        output_mhsa_plus_residual = self.add_and_norm_mhsa(x, output_mhsa)
        output_ffn = self.ffn(output_mhsa_plus_residual)
        output_ffn_plus_residual = self.add_and_norm_ffn(output_ffn, output_mhsa_plus_residual)
        return output_ffn_plus_residual