from turtle import forward
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor


"""
as input we have x in shape (batch_size, )
"""


def attention(q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
    d_k = torch.tensor(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attention = torch.softmax(scores, dim=-1)
    return torch.matmul(attention, v), attention


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


    def forward(self, x: Tensor):
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

        x, attn = attention(q, k, v, None)  # (h x B) x seq x (in_features / h)
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
        pe = torch.zeros((len_pos, len_feat))
        feat_idx = torch.arange(0, len_feat)[None, :].float()  # 1 x len_feat
        pos = torch.arange(0, len_pos)[:, None].float()        # len_pos x 1
        pe[:, 0::2] = torch.sin(pos / (torch.pow(10000, 2 * feat_idx[:, 0::2] / len_feat)))
        pe[:, 1::2] = torch.cos(pos / (torch.pow(10000, 2 * feat_idx[:, 1::2] / len_feat)))
        self.register_buffer('pe', pe, persistent=True)
        x += pe

        return x


class EncoderBlock(nn.Module):

    def __init__(self, in_features: int, n_attn_heads: int = 8):
        super().__init__()

        self._in_features = in_features
        self._n_attn_heads = n_attn_heads

        self._mh_attn = MultiHeadAttention(in_features, n_attn_heads)
        self._layer_norm_0 = LayerNorm(in_features)

        self._feed_forward = nn.Linear(in_features, in_features)
        self._layer_norm_1 = LayerNorm(in_features)

    def forward(self, x: Tensor):
        x_0 = x
        x = self._mh_attn(x)
        x = self._layer_norm_0(x + x_0)
        x_1 = x
        x = self._feed_forward(x)
        x = self._layer_norm_1(x + x_1)

        return x


class DecoderBlock(nn.Module):
    pass


class TransformerBlock(nn.Module):


    def forward(self, x: Tensor):
        pass



def test_attention():
    B, T, L = 2, 4, 4
    q = torch.ones((B, T, L))
    k = torch.ones_like(q)
    v = torch.ones_like(q)
    mask = torch.tril(torch.ones((T, T)))

    print(mask)

    _, attn = attention(q, k, v, mask)
    print(attn.shape)
    print(attn[0, ...])


def test_multihead_attn():
    B, T, L, = 2, 4, 4
    x = torch.ones((B, T, L))
    attn = MultiHeadAttention(4, 2)
    x = attn(x)
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
    enc_blk = EncoderBlock(L, 2)
    x = enc_blk(x)
    print(x.shape)
    print(x[0])


if __name__ == "__main__":
    # test_pe()
    # test_multihead_attn()
    # test_layernorm()
    test_enc_block()
