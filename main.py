import os
import sys
import argparse
from train.train_main import train_main, test_main, tester

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=4, help="batch size")
    parser.add_argument('-n', '--num_frames', type=int, default=8, help="number of frames")
    parser.add_argument('-M', '--mode', type=str, default="train", help="mode", choices=["train", "resume"])
    parser.add_argument('-s', '--seed', type=int, default=42, help="seed")
    parser.add_argument('-k', '--data_mode', type=str, default="keyframe", help="data mode", choices=["train", "test"])
    parser.add_argument('-a', '--alpha', type=float, default=0.5, help="alpha")
    parser.add_argument('-m', '--model', type=str, default="Captioner", help="model name")
    parser.add_argument('-d', '--dataset', type=str, default="MSRVTT", help="dataset name")
    parser.add_argument('-D', '--device', type=str, default="cuda", help="device name", choices=["cuda","cuda:0", "cpu"])
    parser.add_argument('-o', '--optim', type=str, default="AdamW", help="optimizer name", choices=["Adam", "AdamW"])
    parser.add_argument('-e', '--epochs', type=int, default=10, help="number of epochs")
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help="learning rate")
    
    args = parser.parse_args()
    args.absPath = os.path.dirname(os.path.abspath(__file__))
    # tester(args)
    if args.data_mode == "train":
        train_main(args)
    else:
        test_main(args)
