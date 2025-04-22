import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

class Expert(nn.Module):

    def __init__(self, args: dataclass):
        super().__init__()
        r"""
        An Expert is a FeedForward Neural Network with SwiGLU activation.
        Args:
            args (dataclass): Model arguments.

        Returns:
            out (torch.Tensor): Output of expert.

        """
        self.w1 = nn.Linear(args.dim, args.ffn_hidden_dim, bias = False)
        self.w2 = nn.Linear(args.ffn_hidden_dim, args.dim, bias = False)
        self.w3 = nn.Linear(args.dim, args.ffn_hidden_dim, bias = False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, hidden_dim)
        x_W = self.w1(x)
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, hidden_dim)
        x_V = self.w3(x)
        # (batch_size, seq_len, hidden_dim) * (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, hidden_dim)
        out = F.silu(x_W) * x_V
        # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, dim)
        out = self.w2(out)

        return out
    
class NoisyTopkRouter(nn.Module):

    def __init__(self, args: dataclass) -> None:
        super(NoisyTopkRouter, self).__init__()
        r"""
        A noisy router that creates sparse gate to route the tokens to the experts.
        Args:
            args (dataclass): Model arguments

        Returns:
            out (tuple): Output of expert

        """
        self.topk = args.k
        self.gate = nn.Linear(args.dim, args.num_experts)
        self.noisy_gate = nn.Linear(args.dim, args.num_experts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        gate_logits = self.gate(x) # (batch_size, seq_len, num_experts)
        noisy_logits = self.noisy_gate(x) # (batch_size, seq_len, num_experts)

        noise = torch.randn_like(noisy_logits) * F.softplus(noisy_logits)
        noisy_logits = noise + gate_logits

        topklogits, topexperts = noisy_logits.topk(self.topk, dim = -1) # (batch_size, seq_len, topk)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, topexperts, topklogits)
        router_output = F.softmax(sparse_logits, dim = -1) # (batch_size, seq_len, num_experts)

        return router_output, topexperts

class MoE(nn.Module):

    def __init__(self, args: dataclass) -> None:
        super(MoE, self).__init__()
        self.topk = args.k
        self.router = NoisyTopkRouter(args)
        self.experts = nn.ModuleList([Expert(args) for _ in range(args.num_experts)])
        self.aux_loss = args.aux_loss
        self.aux_loss_coeff = args.aux_loss_coeff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_scores, top_experts = self.router(x)
        out = torch.zeros_like(x)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_gate_score = gate_scores.view(-1, gate_scores.size(-1))

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (top_experts == i).any(dim = -1)
            flat_expert_mask = expert_mask.view(-1)

            if flat_expert_mask.any():
                expert_input = flat_x[flat_expert_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_gate_score[flat_expert_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                out[expert_mask] += weighted_output.squeeze(1)
        
        # Auxiliary Loss
        if self.aux_loss:
            imp = gate_scores.sum(1)
            cv = imp.var() / (imp.mean() ** 2)
            cv *= self.aux_loss_coeff
            return out, cv
        else:
            return out, None