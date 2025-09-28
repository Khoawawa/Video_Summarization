import os
import torch
import shutil
from utils.util import create_model, load_datadict
from train.train_model import train_model, test_model

def train_main(args):
    # from args load dataloader
    dataloaders = load_datadict(args)
    
    model : torch.nn.Module = create_model(args)
    
    model_dir = f"{args.absPath}/data/save_models/{args.model}_{args.dataset}"
    args.model_dir = model_dir
    model  = model.to(args.device)
    # create optimizer
    if args.optim.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(f"Optimizer {args.optim} not implemented")

    if args.model == "train":
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir, ignore_errors=True)
        os.makedirs(model_dir, exist_ok=True)
    elif args.model == "resume":
        # load model checkpoint
        final_model = torch.load(os.path.join(model_dir, "final_model.pkl"), map_location=args.device)
        model.load_state_dict(final_model['model_state_dict'], strict=False)
        # load optim checkpoint
        optimizer.load_state_dict(final_model['optimizer_state_dict'])
        # epoch cycle
        start_epoch = final_model['epoch'] + 1
    
    train_model(
        model=model, 
        data_loaders=dataloaders,
        optimizer=optimizer,
        model_dir=model_dir,
        args=args,
        start_epoch=start_epoch if args.model == "resume" else 0
                )    
    
    test_model(model=model, test_loader=dataloaders['test'], args=args)
    