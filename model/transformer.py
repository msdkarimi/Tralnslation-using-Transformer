from model import *


class Transformer(nn.Module):
    def __init__(self, encoder_input_vocab_size: int, decoder_input_vocab_size: int, embedding_size: int, max_seq_len: int, ff_hidden_layer: int, head: int = 6, dropout: float = None, N=6):
        super(Transformer, self).__init__()

        self.encoder = Encoder(encoder_input_vocab_size=encoder_input_vocab_size, embedding_size=embedding_size, max_seq_len=max_seq_len, ff_hidden_layer=ff_hidden_layer, head=head, dropout=dropout, N=N)
        self.decoder = Decoder(decoder_input_vocab_size=decoder_input_vocab_size, embedding_size=embedding_size, max_seq_len=max_seq_len, ff_hidden_layer=ff_hidden_layer, head=head, dropout=dropout, N=N)
        self.linear = nn.Linear(embedding_size, decoder_input_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, encoder_input, decoder_input, mask_encoder, mask_decoder):

        encoder_output = self.encoder(encoder_input, mask_encoder)
        decoder_output = self.decoder(decoder_input, encoder_output, mask_decoder)
        logits = self.linear(decoder_output)  # torch.Size([16, 512, 22463]) (B, max_seq_len, vocab_size)
        return logits
