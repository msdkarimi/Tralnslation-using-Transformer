from model import *


class Decoder(nn.Module):
    def __init__(self, decoder_input_vocab_size: int, embedding_size: int, max_seq_len: int, ff_hidden_layer: int, head: int, dropout: float = None, N=6):
        super(Decoder, self).__init__()
        self.N = N
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(embedding_size=embedding_size,
                         ff_hidden_layer=ff_hidden_layer,
                         head=head,
                         dropout=dropout)
            for _ in range(self.N)
            ])

    def forward(self, decoder_input, encoder_output, mask_tgt, mask_src):
        for a_decoder_block in self.decoder_blocks:
            decoder_output = a_decoder_block(decoder_input, encoder_output, mask_tgt, mask_src)
            decoder_input = decoder_output
        return decoder_output


class DecoderBlock(nn.Module):
    def __init__(self, embedding_size: int, ff_hidden_layer: int, head: int, dropout: float = None):
        super(DecoderBlock, self).__init__()

        self.masked_mhsa = MultiHeadSelfAttention(embedding_size=embedding_size, dropout=dropout, head=head)
        self.mhca = MultiHeadSelfAttention(embedding_size=embedding_size, dropout=dropout, head=head)

        self.ffn = FeedForward(embedding_size=embedding_size, ff_hidden_layer=ff_hidden_layer, dropout=dropout)

        self.add_and_norm_masked_mhsa = AddAndNorm(embedding_size=embedding_size)
        self.add_and_norm_mhca = AddAndNorm(embedding_size=embedding_size)
        self.add_and_norm_ffn = AddAndNorm(embedding_size=embedding_size)

    def forward(self, x, encoder_output, mask_tgt, mask_src):
        output_masked_mhsa, _ = self.masked_mhsa(x, x, x, mask_tgt)
        output_masked_mhsa_plus_residual = self.add_and_norm_masked_mhsa(x, output_masked_mhsa)

        cross_attention_output, _ = self.mhca(output_masked_mhsa_plus_residual, encoder_output, encoder_output, mask_src)
        output_mhca_plus_residual = self.add_and_norm_mhca(cross_attention_output, output_masked_mhsa_plus_residual)

        output_ffn = self.ffn(output_mhca_plus_residual)
        output_ffn_plus_residual = self.add_and_norm_ffn(output_ffn, output_mhca_plus_residual)

        return output_ffn_plus_residual
