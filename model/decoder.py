from model import *


class Decoder(nn.Module):
    def __init__(self, decoder_input_vocab_size: int, embedding_size: int, ff_hidden_layer: int, head: int, dropout: float = None, N=6):
        super(Decoder, self).__init__()

        self.N = N
        self.pe = PositionalEmbedding(dictionary_size=decoder_input_vocab_size, embedding_size=embedding_size)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(embedding_size=embedding_size, ff_hidden_layer=ff_hidden_layer, head=head, dropout=dropout)
            for _ in range(self.N)
            ])

    def forward(self, decoder_input, encoder_output, mask):
        decoder_input = self.pe(decoder_input)
        for a_decoder in self.decoder_blocks:
            decoder_output = a_decoder(decoder_input, encoder_output, mask)
            decoder_input = decoder_output
        return decoder_output
