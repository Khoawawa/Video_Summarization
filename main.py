import os
import sys
import argparse
from train_main.train_main import train_main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m')
    
    args = parser.parse_args()
    args.absPath = os.path.dirname(os.path.abspath(__file__))
    
    train_main(args)
