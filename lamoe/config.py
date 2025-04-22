from typing import Optional
from dataclasses import dataclass, field
import torch

def get_default_device():
    """Use GPU if available, else CPU"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(i))
        return torch.device('cuda')
    else:
        return torch.device('cpu')



@dataclass
class ModelArgs:
    dim: int = 512
    ffn_hidden_dim: int = 4 * dim
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: Optional[int] = 4
    vocab_size: int = -1
    norm_eps: float = 1e-5
    num_experts: int = 8
    k: int = 2
    eos: str = "<eos>"
    pad: str = "<pad>"
    unk: str = "<unk>"
    aux_loss: Optional[bool] = True
    aux_loss_coeff: Optional[float] = 1e-2
    inference: Optional[bool] = False
    cache: bool = field(init = False)
    
    max_batch_size: int = 32
    max_seq_length: int = 300

    device: str = get_default_device()

    def __post_init__(self):
        self.cache = True if self.inference else False

@dataclass
class TrainEvalArgs: 
    lr: float = 1e-3
    max_train_iter: int = 1000
    max_eval_iter: int = 300
    eval_interval: int = 100
    weight_decay: float = 0.01