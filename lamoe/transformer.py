import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional
from .moe import MoE
from .utils import (precompute_theta_pos_frequencies, SimpleInputMetadata, apply_rotary_embeddings)
from .config import ModelArgs


def repeat_kv(x: torch.Tensor, repeat: int, dim: int) -> torch.Tensor:
    x = torch.repeat_interleave(x, repeats = repeat, dim = dim)
    return x

class MHA(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        """
        Multi-Head Grouped Query Attention with cache and RoPE.
        Args:
            args (ModelArgs): Model arguments
        Returns:
            out (tuple): Attention values, Attention scores

        """
        assert args.n_kv_heads <= args.n_heads, "KV heads must be equal or less than Q heads."
        self.n_heads_q = args.n_heads
        self.n_heads_kv = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.repeat = self.n_heads_q // self.n_heads_kv
        self.head_dim = args.dim // args.n_heads

        self.Wq = nn.Linear(args.dim, self.head_dim * self.n_heads_q, bias = True)
        self.Wk = nn.Linear(args.dim, self.head_dim * self.n_heads_kv, bias = True)
        self.Wv = nn.Linear(args.dim, self.head_dim * self.n_heads_kv, bias = True)
        self.Wo = nn.Linear(args.dim, args.dim, bias = False)

        self.register_buffer("mask", torch.tril(torch.ones([args.max_seq_length, args.max_seq_length], dtype = torch.bool))
                                                .view(1, 1, args.max_seq_length, args.max_seq_length))
        self.softmax_scale = 1 / np.sqrt(args.dim)
        self.cache = args.cache
        if self.cache:
            self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_length, self.n_heads_kv, self.head_dim), device = args.device)
            self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_length, self.n_heads_kv, self.head_dim), device = args.device)

    def forward(
        self, 
        x: torch.Tensor,
        freqs_complex: torch.Tensor,
        start_pos: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:

        batch_size, seq_len, _ = x.shape # (batch_size, seq_len, dim)

        q = self.Wq(x) # (batch_size, seq_len, head_dim * n_heads_q)
        k = self.Wk(x) # (batch_size, seq_len, head_dim * n_heads_kv)
        v = self.Wv(x) # (batch_size, seq_len, head_dim * n_heads_kv)

        q = q.view(batch_size, seq_len, self.n_heads_q, self.head_dim) # (batch_size, seq_len, n_heads_q, head_dim)
        k = k.view(batch_size, seq_len, self.n_heads_kv, self.head_dim) # (batch_size, seq_len, n_heads_kv, head_dim)
        v = v.view(batch_size, seq_len, self.n_heads_kv, self.head_dim) # (batch_size, seq_len, n_heads_kv, head_dim)

        q = apply_rotary_embeddings(q, freqs_complex, x.device) # (batch_size, seq_len, n_heads_q, head_dim)
        k = apply_rotary_embeddings(k, freqs_complex, x.device) # (batch_size, seq_len, n_heads_kv, head_dim)

        if self.cache:
            assert start_pos is not None, "Start position is not given. Give the start position."
            self.cache_k[: batch_size, start_pos : start_pos + seq_len] = k 
            self.cache_v[: batch_size, start_pos : start_pos + seq_len] = v 
            keys = self.cache_k[: batch_size, : start_pos + seq_len] # (batch_size, seq_len, n_heads_kv, head_dim)
            values = self.cache_v[: batch_size, : start_pos + seq_len] # (batch_size, seq_len, n_heads_kv, head_dim)
        else: 
            keys, values = k, v
        
        keys = repeat_kv(keys, self.repeat, 2) # (batch_size, seq_len, n_heads, head_dim)
        values = repeat_kv(values, self.repeat, 2) # (batch_size, seq_len, n_heads, head_dim)

        """
        Actual calculation:
        query (batch_size, seq_len, n_heads, head_dim), keys (batch_size, kv_seq_len, n_heads, head_dim)
                                                      || (Transpose)
        query (batch_size, n_heads, seq_len, head_dim), keys (batch_size, n_heads, kv_seq_len, head_dim) 
                                                      || 
        query (batch_size, n_heads, seq_len, head_dim), keys (batch_size, n_heads, head_dim, kv_seq_len)
                                                      || (Matrix Multiplication, Softmax)
                            Attention scores (batch_size, n_heads, seq_len, kv_seq_len)                                             
        """
        attn_scores = torch.einsum("bshd, bthd -> bhst", q, keys) * self.softmax_scale # (batch_size, n_heads, seq_len, kv_seq_len)
        attn_scores = attn_scores.masked_fill(self.mask[:, :, : seq_len, : seq_len] == 0, float("-inf"))
        attn_scores = F.softmax(attn_scores, dim = -1) 

        """
        Actual calculation:
                                    values (batch_size, kv_seq_len, n_heads, head_dim)
                                                      || (Transpose)
                                    values (batch_size, n_heads, kv_seq_len, head_dim) 
                                                      || 
        attn_scores (batch_size, n_heads, seq_len, kv_seq_len), values (batch_size, n_heads, kv_seq_len, head_dim)
                                                      || (Matrix Multiplication)
                            Attention values (batch_size, n_heads, seq_len, head_dim)
                                                      || (Transpose, reshape)
                            Attention values (batch_size, seq_len, n_heads * head_dim)
        """     
        out = torch.einsum("bhst, bthd -> bshd", attn_scores, values).contiguous().view(batch_size, seq_len, self.n_heads_q * self.head_dim) # (batch_size, seq_len, n_heads * head_dim)
        out = self.Wo(out) # (batch_size, seq_len, dim)

        return out, attn_scores
    
class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        r"""
        Root Mean Square Normalization focuses on re-scaling invariance and regularizes the summed inputs simply according to Root Mean Square.
        Args:
            eps (float): Epsilon to avoid division error
            dim (int): Embedding dimension
        Returns:
            out (torch.Tensor): Tensor

        """
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, dim) / (batch_size, seq_len, 1) -> (batch_size, seq_len, dim)
        norm = (x / (torch.sqrt(x.pow(2).mean(-1, keepdim = True)) + self.eps)).type_as(x)
        return self.weight * norm
    
class Block(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super(Block, self).__init__()
        r"""
        Individual Decoder layer.
        Args:
            args (ModelArgs): Model arguments
        Returns:
            out (tuple): Output tensor from layer, auxloss of MoE

        """
        self.attention = MHA(args)
        self.moe = MoE(args)

        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
    
    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor, start_pos: Optional[int] = None) -> torch.Tensor:
        
        residue = x
        out = self.attention_norm(x) # RMS normalization before activation
        out = residue = self.attention(out, freqs_complex, start_pos)[0] + residue # MHA and Addition
        out = self.ffn_norm(out) # RMS normalization before feedforward neural network
        out, loss = self.moe(out) # Feedforward neural network (MoE)
        out = out + residue # Addition

        return out, loss

class Transformer(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super(Transformer, self).__init__()
        r"""
        Sparse-MoE based Decoder. 
        Args:
            args (ModelArgs): Model arguments
        Returns:
            out (tuple): Output tensor from decoder, total auxloss of MoE

        """
        assert args.vocab_size != -1, "Vocab size must be set."
        
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(Block(args))

        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias = False)

        self.head_dim = args.dim // args.n_heads
        self.freqs_complex = precompute_theta_pos_frequencies(self.head_dim, args.max_seq_length * 2, device = args.device)
        self.aux = args.aux_loss
        self.inference = args.inference
    
    def forward(self, x: torch.Tensor, start_pos: Optional[int] = None) -> torch.Tensor:

        batch_size, seq_len = x.shape
        out = self.tok_embeddings(x)
        aux_loss = []

        if self.inference: # During Inference
            assert start_pos is not None, "Start position is not given. Give the start position."
            assert seq_len == 1, "Only one token will be processed during inference."
            freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]
        else: # During Training
            seq_lens = [x[i].size(0) for i in range(batch_size)]
            seq_lens_metadata = [SimpleInputMetadata.from_seqlens(seq_lens, x.device) for i in range(len(self.layers))]
            positions = seq_lens_metadata[0].positions
            freqs_complex = self.freqs_complex[positions]

        for layer in self.layers:
            out, loss = layer(out, freqs_complex, start_pos)
            if self.aux and loss is not None:
                aux_loss.append(loss)
        
        out = self.norm(out)
        out = self.output(out).float()

        if self.aux: # During training
            total_aux_loss = sum(aux_loss) / len(aux_loss)
            return out, total_aux_loss
        else: # During inference
            return out, None

if __name__ == "__main__":

    args = ModelArgs()
    args.vocab_size = 30000
    args.max_seq_length = 100
    args.device = "cpu"

    tfr = Transformer(args)

    Num_of_parameters = sum(p.numel() for p in tfr.parameters())
    print("Model Parameters : {:.3f} M".format(Num_of_parameters / 1e6)) # Prints Total number of Model Parameters.