import argparse
from argparse import RawTextHelpFormatter
from typing import Optional
import os, sys
from pathlib import Path
from tqdm import tqdm
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lamoe.transformer import Transformer
from lamoe.config import ModelArgs
from lamoe.tokenizer import BPE
from lamoe.utils import preprocess_Eng, get_vocab, get_model_info



def sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:

        probs_sort, probs_idx = torch.sort(probs, dim = -1, descending = True)
        probs_sum = torch.cumsum(probs_sort, dim = -1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim = -1, keepdim = True))
        next_token = torch.multinomial(probs_sort, num_samples = 1)
        next_token = torch.gather(probs_idx, -1, next_token)

        return next_token

def text_completion(model: Transformer, 
                    args: ModelArgs, 
                    prompts: list[str], 
                    temperature: float = 0.6, 
                    top_p: float = 0.9, 
                    max_gen_len: Optional[int] = None) -> tuple[list, list]:

        if max_gen_len is None:
            max_gen_len = args.max_seq_length - 1

        prompts = [preprocess_Eng(prompt) for prompt in prompts]
        prompt_tokens = [tokenizer.encode(prompt, tokenizer_dict, None) for prompt in prompts]
        batch_size = len(prompt_tokens)
        assert batch_size <= args.max_batch_size, f"Batch size must be less than or equal to {args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= args.max_seq_length, f"Sequence length must be less than or equal to {args.max_seq_length}"
        total_len = min(args.max_seq_length, 64 + max_prompt_len)

        pad_id = tokenizer_dict['vocab_dict'].get(args.pad)
        eos_id = tokenizer_dict['vocab_dict'].get(args.eos)
        tokens = torch.full((batch_size, total_len), pad_id, dtype = torch.long, device = args.device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, :len(t)] = torch.tensor(t, dtype = torch.long, device = args.device)

        eos_reached = torch.tensor([False] * batch_size, device = args.device)
        prompt_tokens_mask = tokens != pad_id

        dot_cycle = ['.', '..', '...', '....'] * total_len

        for i, token in enumerate(tokens):
            with tqdm(total = total_len, desc = "Generating Text", bar_format="{desc} {postfix}") as pbar:
                for pos in range(1, total_len):
                    with torch.no_grad():
                            logits, _ = model.forward(token[pos - 1:pos].unsqueeze(0), start_pos = pos)

                    if temperature > 0:
                        probs = torch.softmax(logits[:, -1] / temperature, dim = -1)
                        next_token = sample_top_p(probs, top_p)
                    else:
                        next_token = torch.argmax(logits[:, -1], dim = -1)

                    next_token = next_token.reshape(-1)

                    next_token = torch.where(prompt_tokens_mask[i, pos], tokens[i, pos], next_token)
                    token[pos] = next_token

                    eos_reached |= (~prompt_tokens_mask[:, pos]) & (next_token == eos_id)
                    if all(eos_reached):
                        break
                    
                    dots = dot_cycle[pos-1 % len(dot_cycle)]
                    pbar.set_postfix_str(dots)
                    time.sleep(0.01)
                    pbar.update(1)

        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            if eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(tokenizer.decode(current_prompt_tokens, tokenizer_dict))
        return (out_tokens, out_text)



if __name__ == '__main__':

    doc_string = """
                                            Real-time generation of text for the given user prompt.
                    Working: First, it asks for the user prompt and model generates text and prints sequentially. 
                    Note: To exit the inference, type exit.
                 """
    print(doc_string)
    parser = argparse.ArgumentParser(description = "Text inference using Language model.", formatter_class = RawTextHelpFormatter)

    parser.add_argument("--vocab-file", type = Path, default = "Tokenizer.json", help = "Tokenizer file in .json format that contains vocabulary of training corpus created using Dataset.ipynb.")
    parser.add_argument("--model-path", type = Path, default = os.path.join("model", "MoE-LM.pth"), help = "Path to trained model file.")
    parser.add_argument("--temperature", type = float, default = 0.6, help = "Temperature value to adjust response of the model.")
    parser.add_argument("--top-p", type = float, default = 0.9, help = "Value of p in top-p sampling.")

    args = parser.parse_args()

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    tokenizer = BPE()
    tokenizer_dict = get_vocab(os.path.join(parent_dir, 'Saved', args.vocab_file))

    model_args = ModelArgs(inference = True)
    model_args.vocab_size = len(tokenizer_dict['vocab_dict'])
    model_args.aux_loss = False
    print("\nModel Args: ", model_args, "\n")


    model = Transformer(model_args) 

    model_path = os.path.join(parent_dir, 'Saved', args.model_path)

    try:
         model.load_state_dict(torch.load(model_path, weights_only = True, map_location = 'cpu'), strict = True)
         model.to(model_args.device)
         print("Model is loaded with trained weights successfully.\n")
    except Exception as e:
         print(e, f"\n{model_path} is not present. Train the model to get final {model_path}")
         exit(0)
         
    get_model_info(model, model_args)

    while True:
        text = input("User: ")
        if text == "exit":
             break
        prompts = [text]
        out_tokens, out_texts = (text_completion(model, model_args, prompts, temperature = args.temperature, top_p = args.top_p, max_gen_len = 150))
        assert len(out_texts) == len(prompts)       
        print("Model: ")
        for i in range(len(out_texts)):
            for word in f'{out_texts[i]}':
                print(word, end = '', flush = True)
                time.sleep(0.05)
        print()

    