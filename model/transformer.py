from model import *


class Transformer(nn.Module):
    def __init__(self, encoder_input_vocab_size: int, decoder_input_vocab_size: int, embedding_size: int, max_seq_len: int, ff_hidden_layer: int, head: int = 6, dropout: float = None, N=6):
        super(Transformer, self).__init__()

        self.source_embedding = Embedding(encoder_input_vocab_size, embedding_size)
        self.target_embedding = Embedding(decoder_input_vocab_size, embedding_size)

        self.pe = PositionalEmbedding(embedding_size, max_seq_len, base=1e4)

        self.encoder = Encoder(encoder_input_vocab_size=encoder_input_vocab_size, embedding_size=embedding_size, max_seq_len=max_seq_len, ff_hidden_layer=ff_hidden_layer, head=head, dropout=dropout, N=N)
        self.decoder = Decoder(decoder_input_vocab_size=decoder_input_vocab_size, embedding_size=embedding_size, max_seq_len=max_seq_len, ff_hidden_layer=ff_hidden_layer, head=head, dropout=dropout, N=N)
        self.linear = nn.Linear(embedding_size, decoder_input_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, encoder_input, decoder_input, mask_encoder, mask_decoder):

        encoder_output = self.encoding(encoder_input, mask_encoder)
        decoder_output = self.decoding(decoder_input, encoder_output, mask_decoder, mask_encoder)
        logits = self.projection(decoder_output)
        return logits

    def encoding(self, encoder_input, mask_encoder):
        emb = self.source_embedding(encoder_input)
        pe = self.pe(emb)
        res = self.encoder(pe, mask_encoder)
        return res

    def decoding(self, decoder_input, encoder_output, mask_decoder, mask_encoder):
        emb = self.target_embedding(decoder_input)
        pe = self.pe(emb)
        res = self.decoder(pe, encoder_output, mask_decoder, mask_encoder)
        return res

    def projection(self, decoder_output):
        return self.linear(decoder_output)
