from model import *


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
