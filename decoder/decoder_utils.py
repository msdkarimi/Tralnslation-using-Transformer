from decoder import *

class DecoderBlock(nn.Module):
    def __init__(self, embedding_size: int, ff_hidden_layer: int, head: int, dropout: float = None):
        super(DecoderBlock, self).__init__()

        self.masked_mhsa = MultiHeadSelfAttention(embedding_size=embedding_size, dropout=dropout, head=head)
        self.mhca = MultiHeadSelfAttention(embedding_size=embedding_size, dropout=dropout, head=head)

        self.ffn = FeedForward(embedding_size=embedding_size, ff_hidden_layer=ff_hidden_layer, dropout=dropout)

        self.add_and_norm_masked_mhsa = AddAndNorm(embedding_size=embedding_size)
        self.add_and_norm_mhca = AddAndNorm(embedding_size=embedding_size)
        self.add_and_norm_ffn = AddAndNorm(embedding_size=embedding_size)

    def forward(self, x, encoder_output, mask):
        output_masked_mhsa = self.masked_mhsa(x, x, x, mask)
        output_masked_mhsa_plus_residual = self.add_and_norm_masked_mhsa(x, output_masked_mhsa)

        cross_attention_output = self.mhca(output_masked_mhsa_plus_residual, encoder_output, encoder_output)
        output_mhca_plus_residual = self.add_and_norm_mhca(cross_attention_output, output_masked_mhsa_plus_residual)

        output_ffn = self.ffn(output_mhca_plus_residual)
        output_ffn_plus_residual = self.add_and_norm_ffn(output_ffn, output_mhca_plus_residual)

        return output_ffn_plus_residual
