from time import time
import numpy  as np
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertForQuestionAnswering
import QA_preprossing
import logging
from tqdm import trange
from argparse import ArgumentParser, Namespace
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'bert-base-chinese'

def main(args):
    # load data
    train_data , context = QA_preprossing.read_train_data(args)
    QA_preprossing.preprocess_data(args,train_data , context)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type= str,
        help="Directory to the dataset.",
        default="./dataset/",
    )
    parser.add_argument(
        "--cache_dir",
        type = Path,
        help="Directory to the preprocessed caches.",
        default="./cache/QA/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type = Path,
        help="Directory to save the model file.",
        default="./ckpt/QA/",
    )
    parser.add_argument(
        "--split_ratio",
        type = float,
        help = "split ratio for train_dataset",
        default = 0.9,
    )
    parser.add_argument(
        "--input_length",
        type= int,
        help= "BERT token maximum input length",
        default = 512,
    )
    # optimizer
    parser.add_argument("--lr", type=float, default=1e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default = 2)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default = "cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default = 2)

    args = parser.parse_args()
    # args = parser.parse_known_args()[0] # for colab
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)