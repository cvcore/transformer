from turtle import forward
from typing import Dict, Optional
from numpy import source
import torch
import torch.nn as nn
from torch import Tensor


"""
as input we have x in shape (batch_size, )
"""


def scaled_dp_attention(q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
    """Scaled dot-product attention.
    If mask is given, the tensor elements with a positive mask will be kept, and others filled with -inf before the softmax function."""

    d_k = torch.tensor(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attention = torch.softmax(scores, dim=-1)
    return torch.matmul(attention, v), attention


class FeedForward(nn.Module):

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self._linear_0 = nn.Linear(in_features, hidden_features, bias=True)
        self._linear_1 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x: Tensor):
        x = self._linear_0(x)
        x = torch.relu(x)
        x = self._linear_1(x)

        return x


class InputEmbedding(nn.Module):
    """Input x has dimension (B, MAX_SEQ_LEN, in_features)
    Output has dimension (B, MAX_SEQ_LEN, out_features)

    The input tensor is expected to be encoded from the tokens with an one-hot encoding."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self._embedding_layer = nn.Linear(in_features, out_features)  # Question: include bias or not for this layer?

    def forward(self, x: Tensor):
        return self._embedding_layer(x)


class LayerNorm(nn.Module):
    """Normalizes an input tensor of shape (..., in_features) by first computing the mean and standard derivation over the feature dimension.
    Then, apply the learnable parameters alpha and beta individually for each feature dimensions."""

    def __init__(self, in_features) -> None:
        super().__init__()
        self._in_features = in_features
        self._eps = 1e-9
        self._alpha = nn.Parameter(torch.rand(in_features), requires_grad=True)
        self._beta = nn.Parameter(torch.rand(in_features), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        mu = torch.mean(x, dim=-1, keepdim=True)
        sigma = torch.std(x, dim=-1, unbiased=False, keepdim=True)

        x = (x - mu) / (sigma + self._eps)
        x = x * self._alpha + self._beta

        return x


class MultiHeadAttention(nn.Module):
    """For an input tensor (B x seq_len x in_features), output a tensor with same dimension.
    n_heads is the number of heads for the multi-head attention module."""

    def __init__(self, in_features: int, n_heads: int):
        super().__init__()

        assert in_features % n_heads == 0, f"the in_features = {in_features} is not divisible by n_heads = {n_heads}"
        self._n_heads = n_heads
        self._in_features = in_features

        # x -> q, k, v
        self.w_q = nn.Linear(in_features, in_features)
        self.w_k = nn.Linear(in_features, in_features)
        self.w_v = nn.Linear(in_features, in_features)

        # multi-head attention
        self.v_q = nn.Linear(in_features, in_features)
        self.v_k = nn.Linear(in_features, in_features)
        self.v_v = nn.Linear(in_features, in_features)

        # final linear mapping
        self.linear_out = nn.Linear(in_features, in_features)


    def forward(self, x: Tensor, apply_mask: bool = False):
        len_b, len_seq, len_features = x.shape[-3:]
        assert len_features == self._in_features

        q = self.v_q(self.w_q(x))
        k = self.v_k(self.w_k(x))
        v = self.v_v(self.w_v(x))

        q = torch.split(q, self._n_heads, dim=-1)  # list of B x seq x (in_features / h)
        k = torch.split(k, self._n_heads, dim=-1)
        v = torch.split(v, self._n_heads, dim=-1)

        q = torch.concat(q, dim=0)  # concatenate the multi-head dimensions into the batch dimension.
        k = torch.concat(k, dim=0)
        v = torch.concat(v, dim=0)

        x, attn = scaled_dp_attention(q, k, v, None)  # (h x B) x seq x (in_features / h)
        x = x.view(self._n_heads, len_b, len_seq, len_features // self._n_heads)
        x = x.unsqueeze(-2).swapaxes(0, -2).reshape(len_b, len_seq, len_features).squeeze(0)  # B x seq x in_features

        x = self.linear_out(x)
        return x


class MultiHeadAttentionPure(nn.Module):
    """For an input tensor (B x seq_len x in_features), output a tensor with same dimension.
    n_heads is the number of heads for the multi-head attention module."""

    def __init__(self, in_features: int, n_heads: int):
        super().__init__()

        assert in_features % n_heads == 0, f"the in_features = {in_features} is not divisible by n_heads = {n_heads}"
        self._n_heads = n_heads
        self._in_features = in_features

        # x -> q, k, v
        # self.w_q = nn.Linear(in_features, in_features)
        # self.w_k = nn.Linear(in_features, in_features)
        # self.w_v = nn.Linear(in_features, in_features)

        # multi-head attention
        self.v_q = nn.Linear(in_features, in_features)
        self.v_k = nn.Linear(in_features, in_features)
        self.v_v = nn.Linear(in_features, in_features)

        # final linear mapping
        self.linear_out = nn.Linear(in_features, in_features)


    def forward(self, q: Tensor, k: Tensor, v: Tensor, apply_mask: bool = False):
        assert q.shape == k.shape
        assert q.shape[:-1] == v.shape[:-1]
        len_b, len_seq, len_features = q.shape[-3:]
        assert len_features == self._in_features

        # q = self.v_q(self.w_q(x))
        # k = self.v_k(self.w_k(x))
        # v = self.v_v(self.w_v(x))

        q = torch.split(q, self._n_heads, dim=-1)  # list of B x seq x (in_features / h)
        k = torch.split(k, self._n_heads, dim=-1)
        v = torch.split(v, self._n_heads, dim=-1)

        q = torch.concat(q, dim=0)  # concatenate the multi-head dimensions into the batch dimension.
        k = torch.concat(k, dim=0)
        v = torch.concat(v, dim=0)

        if apply_mask is True:
            mask = torch.tril(torch.ones((len_seq, len_seq)).to(q))
        else:
            mask = None

        x, attn = scaled_dp_attention(q, k, v, mask)  # (h x B) x seq x (in_features / h)
        x = x.view(self._n_heads, len_b, len_seq, len_features // self._n_heads)
        x = x.unsqueeze(-2).swapaxes(0, -2).reshape(len_b, len_seq, len_features).squeeze(0)  # B x seq x in_features

        x = self.linear_out(x)
        return x


class PositionalEncoding(nn.Module):
    """Module that provides position encoding to an input tensor x of shape (B, MAX_SEQ_LEN, in_features)
    by calculating the sin / cos function value with different time t along the SEQ_LEN dimension
    and different frequency f along the in_features dimension"""

    def __init__(self) -> None:
        super().__init__()
        # question: should positional encoding remember the feature size it has been trained on?

    def forward(self, x: Tensor):
        len_feat = x.shape[-1]
        len_pos = x.shape[-2]
        # x is expected to have shape (B, MAX_SEQ_LEN, in_features)
        # where the second and third dimensions correspond to the pos and i variables in the original paper:
        # PE(pos, 2i) = sin(pos / (10000^(2 * i / in_features)))
        # PE(pos, 2i+1) = cos(pos / (10000^(2 * i / in_features)))
        pe = torch.zeros((len_pos, len_feat)).to(x)
        feat_idx = torch.arange(0, len_feat)[None, :].float().to(x)  # 1 x len_feat
        pos = torch.arange(0, len_pos)[:, None].float().to(x)        # len_pos x 1
        pe[:, 0::2] = torch.sin(pos / (torch.pow(10000, 2 * feat_idx[:, 0::2] / len_feat)))
        pe[:, 1::2] = torch.cos(pos / (torch.pow(10000, 2 * feat_idx[:, 1::2] / len_feat)))
        self.register_buffer('pe', pe, persistent=True)
        x += pe

        return x


class EncoderBlock(nn.Module):

    def __init__(self, in_features: int, hidden_features: int, n_attn_heads: int = 8):
        super().__init__()

        self._in_features = in_features
        self._n_attn_heads = n_attn_heads
        self._w_k = nn.Linear(in_features, in_features)
        self._w_v = nn.Linear(in_features, in_features)
        self._w_q = nn.Linear(in_features, in_features)

        self._mh_attn = MultiHeadAttentionPure(in_features, n_attn_heads)
        self._layer_norm_0 = LayerNorm(in_features)

        self._feed_forward = FeedForward(in_features, hidden_features, in_features)
        self._layer_norm_1 = LayerNorm(in_features)

    def forward(self, x: Tensor):
        x_0 = x
        k, q, v = self._w_k(x), self._w_q(x), self._w_v(x)
        x = self._mh_attn(k, q, v, apply_mask=False)
        x = self._layer_norm_0(x + x_0)
        del x_0
        x_1 = x
        x = self._feed_forward(x)
        x = self._layer_norm_1(x + x_1)

        return x


class DecoderBlock(nn.Module):

    def __init__(self, in_features: int, hidden_features, n_attn_heads: int = 8):
        super().__init__()
        self._mh_attn_0 = MultiHeadAttentionPure(in_features, n_attn_heads)
        self._mh_attn_1 = MultiHeadAttentionPure(in_features, n_attn_heads)
        self._feed_forward = FeedForward(in_features, hidden_features, in_features)
        self._layer_norm_0 = LayerNorm(in_features)
        self._layer_norm_1 = LayerNorm(in_features)
        self._layer_norm_2 = LayerNorm(in_features)

        self._w_k_0 = nn.Linear(in_features, in_features)
        self._w_v_0 = nn.Linear(in_features, in_features)
        self._w_q_0 = nn.Linear(in_features, in_features)

        self._w_k_1 = nn.Linear(in_features, in_features)
        self._w_v_1 = nn.Linear(in_features, in_features)
        self._w_q_1 = nn.Linear(in_features, in_features)

    def forward(self, input: Dict):
        """Forward function of the decoder block. enc_out is expected from the encoder block."""

        x, enc_out = input['x'], input['enc_out']

        x_0 = x
        q, k, v = self._w_q_0(x), self._w_k_0(x), self._w_v_0(x)
        x = self._mh_attn_0(q, k, v, apply_mask=True)
        x = self._layer_norm_0(x + x_0)
        del x_0

        x_1 = x
        v, k = self._w_v_1(enc_out), self._w_k_1(enc_out)
        q = self._w_q_1(x)
        x = self._mh_attn_1(q, k, v, apply_mask=False)
        x = self._layer_norm_1(x + x_1)
        del x_1

        x_2 = x
        x = self._feed_forward(x)
        x = self._layer_norm_2(x + x_2)
        del x_2

        return {'x': x, 'enc_out': enc_out}  # we keep the enc_out for the chained connection of decoder block


class Transformer(nn.Module):

    def __init__(
        self,
        dictionary_len: int,
        embedding_dim: int,
        embedding_padding_idx: int = 0,
        ff_hidden_features: int = 2048,
        n_encoder_blocks: int = 8,
        n_decoder_blocks: int = 8,
        n_attn_heads: int = 8,
    ):
        """Parameters:
        dictionary_len: the length of the dictionary, i.e. the number of tokens in the vocabulary
        embedding_dim: number of channels after the input & output embedding
        ff_hidden_features: the number of hidden features in the feed-forward layer
        n_encoder_blocks: number of encoder blocks
        n_decoder_blocks: number of decoder blocks
        n_attn_heads: number of attention heads
        """
        super().__init__()

        self._embedding_padding_idx = embedding_padding_idx
        self._embedding_dim = embedding_dim
        self._word_embedding = nn.Embedding(dictionary_len, embedding_dim, embedding_padding_idx)  # padding_idx depends on which index we use in the encoder for padding.
        self._encoder_blocks = nn.Sequential(*[EncoderBlock(embedding_dim, ff_hidden_features, n_attn_heads) for _ in range(n_encoder_blocks)])
        self._decoder_blocks = nn.Sequential(*[DecoderBlock(embedding_dim, ff_hidden_features, n_attn_heads) for _ in range(n_decoder_blocks)])
        self._pos_encoding = PositionalEncoding()

    def forward(self, input: Dict):
        """input is a dictionary containing the following keys:
        'source': batched input containing tokenized source language sentences stored as one-hot encodings, shape (B, seq_len)
        'target': (Optional) batched input containing tokenized target language sentences stored as one-hot encodings, shape (B, seq_len)
            in the inference time, target should be None, since the model will use it's own output at time t to condition its prediction at time t+1
        Note both source and target tensor have the same shape, because they share a joint vocabulary and also the embedding layer.

        The output is a tensor of shape (B, seq_len, dictionary_len) containing the probability distribution of the next token at each time step.
        """

        if self.training:
            src, tgt = input['source'], input['target']
            del input

            # in training time, we shift the target tensor by one time step to the right
            # and pad the first token with 0, which is the start of sequence token
            tgt = torch.cat([torch.zeros_like(tgt[:, 0:1]), tgt[:, :-1]], dim=-1)

            rescale_factor = torch.sqrt(torch.tensor(self._embedding_dim).to(src))  # make it larger: we don't want the pe later to be louder than the words
            src = self._word_embedding(src) * rescale_factor
            tgt = self._word_embedding(tgt) * rescale_factor

            src = self._pos_encoding(src)
            tgt = self._pos_encoding(tgt)

            src_enc = self._encoder_blocks(src)
            del src
            tgt_dec = self._decoder_blocks({'x': tgt, 'enc_out': src_enc})['x']
            del src_enc

            tgt_dec = tgt_dec @ self._word_embedding.weight.T
            # tgt_dec = torch.softmax(tgt_dec, dim=-1)

            return tgt_dec

        else:
            src = input['source']
            tgt = torch.zeros_like(src)
            del input
            src = self._word_embedding(src)
            src = self._pos_encoding(src)
            src_enc = self._encoder_blocks(src)
            del src

            tgt[:, 0] = self._embedding_padding_idx
            for i in range(tgt.shape[1]):
                tgt_dec = self._word_embedding(tgt)
                tgt_dec = self._pos_encoding(tgt_dec)
                tgt_dec = self._decoder_blocks({'x': tgt_dec, 'enc_out': src_enc})['x']
                tgt_dec = tgt_dec @ self._word_embedding.weight.T
                # tgt_dec = torch.softmax(tgt_dec, dim=-1)
                next_tokens = tgt_dec[:, i, :].argmax(dim=-1)  # shape: (B,)
                if i != tgt.shape[1] - 1:
                    tgt[:, i+1] = next_tokens

            return tgt_dec


def test_attention():
    B, T, L = 2, 4, 4
    q = torch.ones((B, T, L))
    k = torch.ones_like(q)
    v = torch.ones_like(q)
    mask = torch.tril(torch.ones((T, T)))

    print(mask)

    _, attn = scaled_dp_attention(q, k, v, mask)
    print(attn.shape)
    print(attn[0, ...])


def test_multihead_attn():
    B, T, L, = 2, 4, 4
    x = torch.ones((B, T, L))
    attn = MultiHeadAttention(4, 2)
    x = attn(x)
    print(x.shape)
    print(x[0])

def test_multihead_attn_pure():
    B, T, L, = 2, 4, 4
    k = torch.ones((B, T, L))
    v = torch.ones((B, T, L))
    q = torch.ones((B, T, L))
    attn = MultiHeadAttentionPure(L, 2)
    x = attn(k, v, q, False)
    print(x.shape)
    print(x[0])

def test_pe():
    B, T, L = 2, 3, 4
    x = torch.ones((B, T, L))
    pe = PositionalEncoding(L)
    x = pe(x)
    print(x.shape)
    print(x[0, ...])


def test_layernorm():
    B, T, L = 2, 3, 4
    x = torch.ones((B, T, L))
    layer_norm = LayerNorm(L)
    x = layer_norm(x)
    print(x.shape)
    print(x[0])


def test_enc_block():
    B, T, L = 2, 3, 4
    x = torch.ones((B, T, L))
    enc_blk = EncoderBlock(L, 8, 2)
    x = enc_blk(x)
    print(x.shape)
    print(x[0])

def test_dec_block():
    B, T, L = 2, 3, 4
    x = torch.ones((B, T, L))
    enc_blk = EncoderBlock(L, 8, 2)
    dec_blk = DecoderBlock(L, 8, 2)
    enc_out = enc_blk(x)
    dec_out = dec_blk(x, enc_out)
    print(dec_out.shape)
    print(dec_out[0])


def test_transformer():
    B, T, L = 2, 3, 4
    x = torch.ones((B, T)).long()
    y = torch.ones((B, T)).long()
    transformer = Transformer(
        dictionary_len=L,
        embedding_dim=4,
        ff_hidden_features=8,
        n_encoder_blocks=2,
        n_decoder_blocks=2,
        n_attn_heads=2,
    )
    transformer.train()
    out = transformer({'source': x, 'target': y})
    print(out.shape)
    print(out[0])
    transformer.eval()
    out = transformer({'source': x})
    print(out.shape)
    print(out[0])


if __name__ == "__main__":
    # test_pe()
    # test_multihead_attn()
    # test_layernorm()
    # test_enc_block()
    # test_multihead_attn_pure()
    test_transformer()
