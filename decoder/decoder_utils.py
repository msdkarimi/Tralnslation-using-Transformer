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


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_size: int, dropout: float = None, head: int = 1):
        super(MultiHeadSelfAttention, self).__init__()

        assert embedding_size % head == 0, 'it is not dividable by number of heads'

        self.embedding_size = embedding_size
        self.head = head
        self.head_embedding = self.embedding_size // self.head

        self.wq = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.wk = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.wv = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.output = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout

    def make_head_chunks(self, input):
        return input.view(input.shape[0], input.shape[1], self.head, self.head_embedding).transpose(1, 2)

    def stick_head_chunks(self, output):
        transposed_tensor = output.transpose(1, 2).contiguous()
        return transposed_tensor.view(transposed_tensor.shape[0], transposed_tensor.shape[1],  self.embedding_size)

    def scaled_dot_product_attention(self, qs, ks, vs,  mask=None):
        d_k = float(ks.shape[-1])
        transposed_ks = ks.transpose(2, 3)
        similarity_scores = torch.einsum('bhse,bhel->bhsl', qs, transposed_ks)
        scaled_similarity_scores = similarity_scores / np.sqrt(d_k)
        if mask is not None:
            scaled_similarity_scores.masked_fill(mask == 0, -1e10)

        squeezed_scores = self.softmax(scaled_similarity_scores)

        if self.dropout is not None:
            squeezed_scores = self.dropout(squeezed_scores)

        context_aware_scores = torch.einsum('bhke,bhel->bhkl', squeezed_scores, vs)

        return context_aware_scores, squeezed_scores

    def forward(self, q, k, v, mask=None):

        Q = self.wq(q)
        K = self.wk(k)
        V = self.wv(v)

        qs = self.make_head_chunks(Q)
        ks = self.make_head_chunks(K)
        vs = self.make_head_chunks(V)

        context_aware_scores, weights = self.scaled_dot_product_attention(qs, ks, vs, mask=mask)
        output = self.stick_head_chunks(context_aware_scores)
        return self.output(output)


class FeedForward(nn.Module):
    def __init__(self, embedding_size: int, ff_hidden_layer: int, dropout: float = None):
        super(FeedForward, self).__init__()

        self.fcl1 = nn.Linear(embedding_size, ff_hidden_layer)
        self.fcl2 = nn.Linear(ff_hidden_layer, embedding_size)
        self.relu = nn.ReLU()

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout

    def forward(self, mhsa_add_and_norm):
        output_fcl1 = self.fcl1(mhsa_add_and_norm)

        if self.dropout is not None:
            hidden = self.dropout(self.relu(output_fcl1))
        else:
            hidden = self.relu(output_fcl1)
        output_fcl2 = self.fcl2(hidden)

        return output_fcl2


class AddAndNorm(nn.Module):
    def __init__(self, embedding_size: int):
        super(AddAndNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(embedding_size)

    def forward(self, residual_input, output_of_sublayer):
        return self.layer_norm(residual_input + output_of_sublayer)




