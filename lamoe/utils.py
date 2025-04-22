import torch
import numpy as np
from typing import List
from dataclasses import dataclass
import regex as re
import h5py
import json
from pathlib import Path
import os
from tabulate import tabulate
import pandas as pd


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0) -> torch.Tensor:
    r"""
    Computing Thetha_i = 10000 ** -2(i-1)/d, i = [1,2,...,d/2].

    Parameters:
        head_dim (int): Head dim of self-attention
        seq_len (int): Maximum sequence length
        device (str): Device to compute
        theta (float): Fixed theta (default: 1000.0)
    Returns:
        out (torch.Tensor): Frequency complex

    """
    assert head_dim % 2 == 0, "Head Dimension must be divisble by 2"

    # (head_dim / 2)
    theta_pow = torch.arange(0, head_dim, 2)[: (head_dim // 2)].float()
    # (head_dim / 2)
    theta_i = theta ** (-theta_pow / head_dim).to(device)
    # (seq_len)
    seq_pos = torch.arange(seq_len).to(device)
    # (seq_len) outer_product(theta_i) -> (seq_len, head_dim / 2)
    freqs = torch.outer(seq_pos, theta_i).float()
    # (seq_len, head_dim / 2) -> (seq_len, head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str) -> torch.Tensor:
   
   r"""
   Rotary positional Encodings[RPoE] is a relative positional encoding applied between two tokens, which indicates the intensity of relationship between them, in terms of Distance parameter [2].
   RPoE are only applied to the Query and the Keys, but not the Values. It is applied after the vector q and k are multiplied with respective
   W matrices in the attention mechanism.
   
   Parameters:
        x (torch.Tensor): Input tensor
        freqs_complex (torch.Tensor): Pre-computed freqs complex
        device (str): Device to compute
    Returns:
        out (torch.Tensor): Position encoded inputs

   """
   # (batch_size, seq_len, head, head_dim) -> (batch_size, seq_len, head, head_dim / 2)
   x_complex =  torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
   # (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
   # freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
   freqs_complex =  freqs_complex.view(x.shape[0], x.shape[1], 1, -1) # (seq_len, head_dim / 2) -> (batch_size, seq_len, 1, head_dim / 2)
   # (batch_size, seq_len, head, Head_dim / 2) * (1, seq_len, 1, head_dim / 2) -> (batch_size, seq_len, head, head_dim / 2)
   x_rotated = x_complex * freqs_complex
   # (batch_size, seq_len, head, head_dim / 2) -> (batch_size, seq_len, head, head_dim / 2, 2)
   x_rotated = torch.view_as_real(x_rotated)
   # (batch_size, seq_len, head, head_dim / 2) -> (batch_size, seq_len, head, head_dim)
   x_out = x_rotated.reshape(*x.shape)

   return x_out.type_as(x).to(device)

@dataclass
class SimpleInputMetadata:
    # Class to create positions for training tokens.
    positions: torch.Tensor

    @staticmethod
    def from_seqlens(seqlens: List[int], device: torch.device) -> "SimpleInputMetadata":
        return SimpleInputMetadata(
            positions = torch.cat([torch.arange(0, seqlen) for seqlen in seqlens]).to(device = device, dtype = torch.long)
        )
    
def decontractions(phrase) -> str:
    r"""
    Decontracted takes text and convert contractions into natural form.
    
    Parameters:
        phrase (str): Text to be decontracted
    Returns:
        out (str): Decontracted text
    
    """
     # Ref: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/47091490#47091490
    # Specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"won\’t", "will not", phrase)
    phrase = re.sub(r"can\’t", "can not", phrase)

    # General
    phrase = re.sub(r"'", '', phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
    
def preprocess_Eng(text) -> str: # Preprocessing of English sentences
    r"""
    Function to preprocess text.
    
    Parameters:
        text (str): Input raw text
    Returns:
        out (str): Processed text
    
    """
    txt = decontractions(text)
    txt = re.sub('\n', ' ', txt)
    txt = re.sub('\t\t', ' ', txt)
    txt = re.sub('\t', ' ', txt)
    txt = re.sub('[^A-Za-z0-9.% ]+', '', txt)
    txt = re.sub('[$)\?"’°!;\'€:,(/]', '', txt)
    txt = re.sub('\u200b', ' ', txt)
    txt = re.sub('\xa0', ' ', txt)
    txt = re.sub('-', ' ', txt)
    txt = txt.strip() 
    txt = re.sub('\.\.', '.', txt)
    return txt

def get_vocab_size(tokenizer_filename: Path = os.path.join("Saved", "Tokenizer.json")) -> int:
    r"""
    Function to get vocabulary size.
    
    Parameters:
        tokenizer_filename (Path): Path of tokenizer.json (default: Saved\Tokenizer.json)
    Returns:
        out (int): Size of vocbulary
    
    """
    try:
        with open(f'{tokenizer_filename}', 'r') as fp:
            assert os.path.getsize(tokenizer_filename) > 0, "Invalid size"
            print(f"'{tokenizer_filename}' exists. Loading dictionary values from '{tokenizer_filename}'.")
            tokenizer_dict = json.load(fp)
            print("Size of Vocabulary: ", len(tokenizer_dict['vocab_dict']))
            return len(tokenizer_dict['vocab_dict'])
    except Exception as e:
        print(e, f"\nCreate {tokenizer_filename} using Dataset.ipynb.")
    
def get_data(filename: Path) -> np.ndarray:
    r"""
    Function to extract the saved tokens.
    
    Parameters:
        filename (Path): Path of tokens saved in .h5
    Returns:
        out (np.ndarray): Array of tokens
    
    """
    try:
        assert Path(filename).suffix == '.h5', "Invalid file extension"
        assert os.path.getsize(filename) > 0, "Invalid size"
        with h5py.File(filename, "r") as f:
            # print("Keys: %s" % f.keys())
            a_group_key = list(f.keys())[0]
            # print(type(f[a_group_key]))
            data = f[a_group_key][()] 
            return data
    except Exception as e:
        print(e, f"{filename} doesn't exist or invalid. Check for filename or create {filename} using Dataset.ipynb.")

def get_batch(data: np.ndarray, context_len: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Function to create batch of data for the given context length.
    
    Parameters:
        data (np.ndarray): Source data
        context_len (int): Maximum context\sequence length
        batch_size (int): Maximum batch size
    Returns:
        out (tuple): Input tokens, Output tokens
    
    """
    data = torch.tensor(data, dtype = torch.long)
    ix = torch.randint(len(data) - context_len, (batch_size, ))
    x = torch.stack([data[i : i + context_len] for i in ix])
    y = torch.stack([data[i + 1 : i + context_len + 1] for i in ix])
    return x, y

def get_vocab(tokenizer_filename: Path = os.path.join("Saved", "Tokenizer.json")) -> dict:
    r"""
    Function to get tokenizer dictionary.
    
    Parameters:
        tokenizer_filename (Path): Path of tokenizer.json (default: Saved\Tokenizer.json)
    Returns:
        out (dict): Tokenizer dictionary
    
    """
    try:
        with open(f'{tokenizer_filename}', 'r') as fp:
            assert os.path.getsize(tokenizer_filename) > 0, "Invalid size"
            print(f"'{tokenizer_filename}' exists. Loading dictionary values from '{tokenizer_filename}'.")
            tokenizer_dict = json.load(fp)
            print("Size of Vocabulary: ", len(tokenizer_dict['vocab_dict']))
            return tokenizer_dict
    except Exception as e:
        print(e, f"\nCreate {tokenizer_filename} using Dataset.ipynb.")


def df_to_lines(df, header = True) -> tabulate:
    """Convert DataFrame to list of formatted strings (lines)."""
    return tabulate(df, headers='keys' if header else (), tablefmt='plain', showindex=False).splitlines()

def print_comparison(left_title, left_df, right_title, right_df) -> None:
    """Print two dataframes side by side as a comparison."""
    left_lines = df_to_lines(left_df)
    right_lines = df_to_lines(right_df)

    max_lines = max(len(left_lines), len(right_lines))
    left_lines += [''] * (max_lines - len(left_lines))
    right_lines += [''] * (max_lines - len(right_lines))

    print(f"{left_title:<60} || {right_title}")
    print("-" * 130)
    for l, r in zip(left_lines, right_lines):
        print(f"{l:<60} || {r}")


def get_model_info(model, args: dataclass) -> None:
    r"""
    Function to display model information.
    
    Parameters:
        model (Transformer class): Object of transformer
        args (dataclass): Model arguments
    
    """
    Parameters = {}
    Num_of_parameters = sum(p.numel() for p in model.tok_embeddings.parameters())
    Parameters["Embeddings"] = round(Num_of_parameters / 1e6, 3)
    Num_of_parameters = sum(p.numel() for p in model.layers.parameters())
    Parameters[f"Layers - {args.n_layers}"] = round(Num_of_parameters / 1e6, 3)
    Num_of_parameters = sum(p.numel() for p in model.output.parameters())
    Parameters["LLM-Head"] = round(Num_of_parameters / 1e6, 3)

    Layers_Parameters = {}
    Num_of_parameters = sum(p.numel() for p in model.layers[0].attention.parameters())
    Layers_Parameters["MHA"] = round(Num_of_parameters / 1e6, 3)
    Num_of_parameters = sum(p.numel() for p in model.layers[0].moe.parameters())
    Layers_Parameters[f"MOE"] = round(Num_of_parameters / 1e6, 3)

    Moe_Parameters = {}
    Num_of_parameters = sum(p.numel() for p in model.layers[0].moe.router.parameters())
    Moe_Parameters["Router"] = round(Num_of_parameters / 1e6, 3)
    Num_of_parameters = sum(p.numel() for p in model.layers[0].moe.experts.parameters())
    Moe_Parameters[f"Experts"] = round(Num_of_parameters / 1e6, 3)

    Expert_Parameters = {}
    Num_of_parameters = sum(p.numel() for p in model.layers[0].moe.experts[0].parameters())
    Expert_Parameters["Expert"] = round(Num_of_parameters / 1e6, 3)
    
    total_params = pd.DataFrame(Parameters.items(), columns = ['Name', 'Parameters(M)'])
    layer_params = pd.DataFrame(Layers_Parameters.items(), columns = ['Name', 'Parameters(M)'])
    moe_params = pd.DataFrame(Moe_Parameters.items(), columns = ['Name', 'Parameters(M)'])
    expert_params = pd.DataFrame(Expert_Parameters.items(), columns = ['Name', 'Parameters(M)'])

    # Create initial versions
    total_params_load = total_params.copy()
    layer_params_load = layer_params.copy()
    moe_params_load = moe_params.copy()
    expert_params_load = expert_params.copy()

    # Update for inference
    moe_params_inf = moe_params.copy()
    moe_params_inf.loc[1, 'Parameters(M)'] = expert_params['Parameters(M)'][0] * args.k

    layer_params_inf = layer_params.copy()
    layer_params_inf.loc[1, 'Parameters(M)'] = moe_params_inf['Parameters(M)'].sum()

    total_params_inf = total_params.copy()
    total_params_inf.loc[1, 'Parameters(M)'] = round((layer_params_inf['Parameters(M)'].sum() * args.n_layers) + 0.006, 3)

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2

    # Print comparisons
    print("-------------------------------------------------------- Model Summary --------------------------------------------------------\n")
    print('Model size: {:.3f} MB'.format(size_all_mb))
    print(f"Number of Experts: {args.num_experts} (Loaded) vs {args.k} (Inference)")
    print_comparison(f"Total params: {round(total_params_load['Parameters(M)'].sum(), 3)} M (Loaded)", total_params_load, 
                     f"Total params: {round(total_params_inf['Parameters(M)'].sum(), 3)} M (Inference)", total_params_inf)
    print()
    print_comparison(f"Layer: {round(layer_params_load['Parameters(M)'].sum(), 3)} M", layer_params_load, 
                     f"Layer: {round(layer_params_inf['Parameters(M)'].sum(), 3)} M", layer_params_inf)
    print()
    print_comparison(f"MoE params: {round(moe_params_load['Parameters(M)'].sum(), 3)} M", moe_params_load, 
                     f"MoE params: {round(moe_params_inf['Parameters(M)'].sum(), 3)} M", moe_params_inf)
    print()
    print_comparison(f"Expert params: {round(expert_params_load['Parameters(M)'].sum(), 3)} M", expert_params, 
                     f"Expert params: {round(expert_params_load['Parameters(M)'].sum(), 3)} M", expert_params)
    print("-"*130)

