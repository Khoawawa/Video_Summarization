import os
import sys
import argparse
from train.train_main import train_main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="Captioner", help="model name", choices=["Captioner"])
    parser.add_argument('-d', '--dataset', type=str, default="MSRVTT", help="dataset name", choices=["MSRVTT"])
    parser.add_argument('-d', '--device', type=str, default="cuda", help="device name", choices=["cuda", "cpu"])
    parser.add_argument('-o', '--optim', type=str, default="Adam", help="optimizer name", choices=["Adam", "AdamW"])
    parser.add_argument('-e', '--epochs', type=int, default=10, help="number of epochs")
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help="learning rate")
    
    args = parser.parse_args()
    args.absPath = os.path.dirname(os.path.abspath(__file__))
    
    train_main(args)
