import torch
from torch import nn, optim
import time
import copy
from tqdm import tqdm
def train_model(model: nn.Module, data_loaders: dict[str, torch.utils.data.DataLoader], 
                loss_fn: nn.Module, optimizer: optim.Optimizer, 
                model_dir: str, args, start_epoch: int = 0):
    # setting up training loop
    num_epochs = args.epochs
    start_time = time.perf_counter()
    phases = ['train', 'val']
    with open(model_dir + "/output.txt", "a") as f:
        f.write(f"Training start time: {start_time} seconds\n")
    
    save_dict, best_mae = {'model_state_dict': copy.deepcopy(model.state_dict()), 'epoch': start_epoch}, float('inf')
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        running_loss = {phase: 0.0 for phase in phases}
        
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            steps, preds, tgts = 0, list(), list()
            tqdm_loader = tqdm(data_loaders[phase], mininterval=3)
            # main loop start
            
            for frames, captions in tqdm_loader:
                steps += len(captions)
                frames = frames.to(args.device)
                tgts.extend(captions)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs