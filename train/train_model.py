import os
import torch
from torch import nn, optim
from torch.nn import functional as nnf
import time
import copy
from tqdm import tqdm
from utils.util import save_model
from utils.metric import calculate_metrics
import json
def train_model(model: nn.Module, data_loaders: dict[str, torch.utils.data.DataLoader], 
                optimizer: optim.Optimizer, 
                model_dir: str, args, start_epoch: int = 0):
    # setting up training loop
    alpha = args.alpha
    num_epochs = args.epochs
    scaler = torch.amp.GradScaler()
    start_time = time.perf_counter()
    with open(model_dir + "/output.txt", "a") as f:
        f.write(str(model))
        f.write(f"\n\n")
    tokenizer = data_loaders['train'].dataset.dataset.tokenizer
    save_dict, best_cider = {'model_state_dict': copy.deepcopy(model.state_dict()), 'epoch': start_epoch}, float('-inf')
    # separating train and val is crucial
    try:
        for epoch in range(start_epoch, start_epoch + num_epochs):
            # training loop
            running_loss = 0.0
            model.train()
            steps, preds, tgts = 0, list(), list()
            tqdm_train_loader = tqdm(data_loaders['train'], mininterval=3)
            for images, caption_tokens, mask in tqdm_train_loader:
                # images : tensor of shape (B, C, H, W)
                # captions_token: tensor of shape (B, T, D)
                # mask: tensor of shape (B, prefix_len + T)
                model.zero_grad()
                steps += caption_tokens.shape[0]
                
                images = images.to(args.device)
                caption_tokens = caption_tokens.to(args.device)
                mask = mask.to(args.device)
                
                out, loss_align = model(images,caption_tokens)
                logits = out.logits[:, args.model_config['prefix_len'] - 1: -1]
                loss_ce = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), caption_tokens.flatten(),ignore_index=0)
                
                loss = loss_ce +  alpha * loss_align
                # backpropagate
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss.backward()
                
                running_loss += loss.item() * caption_tokens.shape[0]
                
                tqdm_train_loader.set_description(
                    f"Train epoch: {epoch}, loss: {(running_loss / steps) : .8f}, cross_entropy: {loss_ce.item() : .8f}, align_loss: {loss_align.item() : .8f}"
                )
                # torch.cuda.empty_cache()
            train_time = time.perf_counter() - start_time
            # logging train loss
            with open(model_dir + "/output.txt", "a") as f:
                f.write(f"Train epoch: {epoch}, train loss: {(running_loss / steps) : .8f}\n")
                f.write(f"Train time: {train_time} seconds\n")
                f.write("\n")
            
            # validation loop
            model.eval()
            preds, tgts = [], []
            with torch.no_grad():
                for images, caption_tokens, _ in tqdm(data_loaders['val'], mininterval=3,desc="Validating"):
                    # images : tensor of shape (B, C, H, W)
                    # captions_token: tensor of shape (B, T, D)
                    # mask: tensor of shape (B, prefix_len + T)
                    images = images.to(args.device)
                    # [str] * B
                    captions = tokenizer.batch_decode(caption_tokens, skip_special_tokens=True)
                    # [str] * B
                    outputs_texts = model(images) # inference should only be through prefix
                    
                    for refs, pred in zip(captions, outputs_texts):
                        tgts.append(refs if isinstance(refs, list) else [refs])
                        preds.append(pred)
            scores = calculate_metrics(preds, tgts) # this should calculate all metric we want --> BLEU, CIDER, METEOR, ....
            val_time = time.perf_counter() - start_time 
            # logging metrics
            with open(model_dir + "/output.txt", "a") as f:
                f.write(f"Val epoch: {epoch}\n") 
                f.write(str(scores))
                f.write("\n")
                f.write(f"Val time: {val_time} seconds\n")
                f.write("\n")
            # for this part we need to save the best model in case of multiple epoch training
            if scores['CIDEr'] > best_cider:
                best_cider = scores['CIDEr']
                save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()), epoch=epoch,optimzer_state_dict=copy.deepcopy(optimizer.state_dict()))
                save_model(f"{model_dir}/best_model.pkl", **save_dict)
    finally:
        time_elapsed = time.perf_counter() - start_time
        h, rem = divmod(time_elapsed, 3600)
        m, s   = divmod(rem, 60)
        print(f"Training complete in {h} hours {m} minutes {s} seconds")
        
        save_model(f"{model_dir}/final_model.pkl",
               **{
                   'model_state_dict': copy.deepcopy(model.state_dict()),
                   'epoch': start_epoch + num_epochs,
                   'optimizer_state_dict': copy.deepcopy(optimizer.state_dict())
               })
                    
def test_model(model: nn.Module, test_loader: torch.utils.data.DataLoader, args):
    result_dir = args.data_config['out_dir']
    os.makedirs(result_dir, exist_ok=True)
    pred_file = f"{result_dir}/predictions.jsonl"
    
    model.eval()
    tokenizer = test_loader.dataset.dataset.tokenizer
    preds, tgts = [], []
    with torch.no_grad():
        for images, caption_tokens, _ in tqdm(test_loader, mininterval=3,desc="Validating"):
            # images : tensor of shape (B, C, H, W)
            # captions_token: tensor of shape (B, T, D)
            # mask: tensor of shape (B, prefix_len + T)
            images = images.to(args.device)
            # [str] * B
            captions = tokenizer.batch_decode(caption_tokens, skip_special_tokens=True)
            # [str] * B
            outputs_texts = model(images) # inference should only be through prefix
            for refs, pred in zip(captions, outputs_texts):
                tgts.append(refs if isinstance(refs, list) else [refs])
                preds.append(pred)
    metric = calculate_metrics(preds, tgts)
    print(metric)
    
    with open(result_dir + "/output.txt", "a") as f:
        f.write(f"Test\n") 
        f.write(str(metric))
        f.write("\n")
        f.write("\n")
    # prediction append
    with open(pred_file, 'w', encoding='utf-8') as f:
        for i, (pred, refs) in enumerate(zip(preds, tgts)):
            entry = {
                "id": i,
                "prediction": pred,
                "references": refs
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        