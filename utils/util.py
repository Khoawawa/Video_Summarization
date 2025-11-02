import os
import torch
import json
from models.Captioner import Captioner
def create_model(args):
    # create model based on args
    absPath = os.path.join(os.path.dirname(__file__), "model_config.json")
    with open(absPath, "r") as f:
        model_config = json.load(f)[args.model]
    args.model_config = model_config
    if "captioner" in args.model.lower():
        model = Captioner(**model_config)
    
    return model
def save_model(model_path: str, **save_dict):
    os.makedirs(os.path.split(model_path)[0], exist_ok=True)
    torch.save(save_dict, model_path)

