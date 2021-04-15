import torch
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type= str,
        help="Directory to the dataset.",
        default="./dataset/",
    )
    args = parser.parse_args()
    return args

def read_train_data(args):

    train_path = args.data_dir + "train.json"
    context_path = args.data_dir + "context.json"
    print(train_path)
    # Opening JSON file
    f_train = open(train_path , encoding = "utf-8")
    train_data = json.load(f_train)
    f_context = open(context_path , encoding = "utf-8")
    context = json.load(f_context)
    return train_data , context

        

if __name__ == "__main__":
    args = parse_args()
    # args.output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset , context = read_train_data(args)

