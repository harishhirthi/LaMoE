import argparse
from argparse import RawTextHelpFormatter
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import os
import urllib.request
import glob
import sys

import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
import mlflow

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lamoe.transformer import Transformer
from lamoe.config import ModelArgs, TrainEvalArgs
from lamoe.utils import get_data, get_vocab_size, get_batch



def perplexity_score(loss: torch.Tensor) -> float:
    return torch.exp(loss).item()

@torch.no_grad()
def estimate_model(train_data: np.ndarray,
                   val_data: np.ndarray,
                   model: Transformer,
                   model_args: ModelArgs, 
                   train_eval_args: TrainEvalArgs, 
                   loss_criterion: nn.CrossEntropyLoss) -> defaultdict:
    
    out = defaultdict(dict)
    data = {"train": train_data, "val": val_data}
    model.eval()
    for key, split in data.items():
        losses = torch.zeros(train_eval_args.max_eval_iter)
        perplexities = torch.zeros(train_eval_args.max_eval_iter)
        for k in range(train_eval_args.max_eval_iter):

            x, y = get_batch(split, model_args.max_seq_length, model_args.max_batch_size)
            logits, aux_loss = model(x.to(model_args.device))
            task_loss = loss_criterion(logits.view(-1, logits.size(2)), y.view(-1).to(model_args.device))

            perplexity = perplexity_score(task_loss)
            perplexities[k] = perplexity

            total_loss = task_loss + aux_loss if aux_loss is not None else task_loss
            losses[k] = total_loss.item()

        out[key] = {"loss": losses.mean(), "perplexity": perplexities.mean().item()}

    return out


def train(model_args: ModelArgs,
          train_eval_args: TrainEvalArgs,
          model: Transformer,
          train_data: np.ndarray,
          val_data: np.ndarray,
          ckpt_dir: Path) -> None:
    
    try:
        urllib.request.urlopen("http://127.0.0.1:5000/").getcode()
        run_name = f"Experiment-{np.random.randint(1e6)}"
        mlflow.set_experiment("MoE Training")
        mlflow.set_tracking_uri(uri = "http://127.0.0.1:5000/") 
        print("Mlflow current run name inside MoE Training: ", run_name, "\n")
    except Exception as e:
        print("Mlflow tracking server is not initiated. Initiate the server to start the training.")
        print("To start the tracking server. Open terminal in current directory and type mlflow ui\n")
        exit(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr = train_eval_args.lr, weight_decay = train_eval_args.weight_decay, eps = 1e-8)
    loss_criterion = nn.CrossEntropyLoss()

    val_temp = 0
    print(".......................................Executing training of the model.......................................\n")
    with mlflow.start_run(run_name = run_name):
        params = {"Num_layers": model_args.n_layers, "Num_Q_heads": model_args.n_heads, "Num_KV_heads": model_args.n_kv_heads,
                  "Num_experts": model_args.num_experts, "Top_experts": model_args.k, "Vocab_size": model_args.vocab_size,
                  "Dimension": model_args.dim, "batch_size": model_args.max_batch_size , "context_length" : model_args.max_seq_length, 
                  "Max_iters": train_eval_args.max_train_iter, "eval_interval": train_eval_args.eval_interval,  
                  "Device": model_args.device, "eval_iters": train_eval_args.max_eval_iter, "aux_loss_coeff": model_args.aux_loss_coeff, 
                  "optimizer": "AdamW", "learning_rate": train_eval_args.lr, "weight_decay": train_eval_args.weight_decay}
        mlflow.log_params(params)

        for iter in tqdm(range(train_eval_args.max_train_iter)):

            x, y = get_batch(train_data, model_args.max_seq_length, model_args.max_batch_size)
            model.train()
            logits, aux_loss = model(x.to(model_args.device))
            task_loss = loss_criterion(logits.view(-1, logits.size(2)), y.view(-1).to(model_args.device))
            mlflow.log_metric("Task_Loss", task_loss.item(), step = iter)

            perplexity = perplexity_score(task_loss)
            mlflow.log_metric("Perplexity", perplexity, step = iter)

            if aux_loss is not None:
                total_loss = task_loss + aux_loss 
                mlflow.log_metric("Aux_Loss", aux_loss.item(), step = iter)
            else:
                total_loss = task_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (iter % train_eval_args.eval_interval == 0) or (iter == train_eval_args.max_train_iter - 1):
                estimates = estimate_model(train_data, val_data, model, model_args, train_eval_args, loss_criterion)
                print(f"\nStep {iter}: Train Loss - {estimates['train']['loss']:.4f}, Train Perplexity - {estimates['train']['perplexity']:.4f}, Val Loss - {estimates['val']['loss']:.4f}, Val Perplexity -  {estimates['val']['perplexity']:.4f}")
                
                if iter == 0:
                    files_list = glob.glob(os.path.join(ckpt_dir, "*.pth"))
                    if files_list:
                        for file in files_list:
                            os.remove(file)
                    val_temp = estimates['val']['loss']
                    training_state = dict()
                    training_state['ckpt_state_dict'] = model.state_dict()
                    training_state['optimizer_state'] = optimizer.state_dict()
                    ckpt_name = f"checkpoint-{iter}-{val_temp:.3f}.pth"
                    print(f"Saving first checkpoint in {os.path.join(ckpt_dir, ckpt_name)}")
                    torch.save(training_state, os.path.join(ckpt_dir, ckpt_name))
                    
                if (iter > 0) and (estimates['val']['loss'] < val_temp):
                    files_list = glob.glob(os.path.join(ckpt_dir, "*.pth"))
                    if files_list:
                        for file in files_list:
                            os.remove(file)
                    print(f"Val loss improved from {val_temp:.4f} to {estimates['val']['loss']:.4f}.")
                    val_temp = estimates['val']['loss']
                    training_state = dict()
                    training_state['ckpt_state_dict'] = model.state_dict()
                    training_state['optimizer_state'] = optimizer.state_dict()
                    ckpt_name = f"checkpoint-{iter}-{val_temp:.3f}.pth"
                    print(f"Saving checkpoint in {os.path.join(ckpt_dir, ckpt_name)}")
                    torch.save(training_state, os.path.join(ckpt_dir, ckpt_name))


                metrics = {"Train_Loss": float(estimates['train']['loss']), "Train_Perplexity": float(estimates['train']['perplexity']),
                           "Val_Loss": float(estimates['val']['loss']), "Val_Perplexity": float(estimates['val']['perplexity'])}
                mlflow.log_metrics(metrics, step = iter)

        print()
        model_name = "MoE-LM.pth"
        torch.save(model.state_dict(), os.path.join(Path(ckpt_dir).parent, model_name))
        print(f"Training is completed successfully. Final model is saved in {os.path.join(Path(ckpt_dir).parent, model_name)}")        
            
if __name__ == '__main__':

    string = """
                                            Training and Evaluation of LaMoE model.
                !!Primary Note!!: Make sure train.h5, val.h5 and Tokenizer.json is created using Dataset.ipynb.  
                !!Caution Note!!: This training and evluation uses mlflow for logging. So, make sure you started the mlfow
                              tracking server using mlflow ui in terminal opened in current directory and then run the training script.
                        Note: To make changes in model, training and evaluation configuration, edit config.py to make changes.
             """

    parser = argparse.ArgumentParser(description = string, formatter_class = RawTextHelpFormatter)

    parser.add_argument("--train-file", type = Path, default = "train.h5", help = "Train file in .h5 format that contains tokens of training data created using Dataset.ipynb.")
    parser.add_argument("--val-file", type = Path, default = "val.h5", help = "Val file in .h5 format that contains tokens of validation/test data created using Dataset.ipynb.")
    parser.add_argument("--vocab-file", type = Path, default = "Tokenizer.json", help = "Tokenizer file in .json format that contains vocabulary of training corpus created using Dataset.ipynb.")

    args = parser.parse_args()


    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(parent_dir, "Saved", "model")
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok = True)

    train_data = get_data(os.path.join(parent_dir, 'Saved', args.train_file))
    val_data = get_data(os.path.join(parent_dir, 'Saved', args.val_file))
    print(f"\nTokens in training data is {len(train_data)}. Tokens in validtion/test data is {len(val_data)}.\n")

    vocab_size = get_vocab_size(os.path.join(parent_dir, 'Saved', args.vocab_file))

    model_args = ModelArgs()
    model_args.vocab_size = vocab_size
    print("\nModel Args: ", model_args, "\n")

    model = Transformer(model_args)
    model.to(model_args.device)

    Num_of_parameters = sum(p.numel() for p in model.parameters())
    print("Model Parameters : {:.3f} M".format(Num_of_parameters / 1e6), "\n") # Prints Total number of Model Parameters.

    x, y = get_batch(train_data, model_args.max_seq_length, model_args.max_batch_size)
    print("Input shape: ", x.shape, "Output shape: ", y.shape, "\n")

    print("Summary of the model:\n", summary(model, [(x)], device = model_args.device, verbose = 0), "\n")

    train_eval_args = TrainEvalArgs()
    train(model_args, train_eval_args, model, train_data, val_data, checkpoint_dir)

    