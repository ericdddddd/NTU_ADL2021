import os

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import pickle

from sgdqa_model import SGDQAModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from pathlib import Path
from nemo.utils.exp_manager import exp_manager
from transformers import BertTokenizer, BertModel
from argparse import ArgumentParser, Namespace
import json
from torch.utils.data import DataLoader, Dataset
import torch

from utils_unseen import extract_test_data, read_data, DSTlabel, EmbedDataset, create_utterance, create_data, finalDataset, postprocess, serv_pred
from model_clf import serviceModel
from train_unseen import serv_pred

@hydra_runner(config_path="conf", config_name="sgdqa_config")
def main(cfg: DictConfig) -> None:
    # service classifier
    torch.cuda.set_device(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(torch.cuda.current_device())

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    print("--------------- {:15} ---------------".format("testing data"))
    test_data = read_data(args.test_file, mode=args.mode)
    print("Number of testing data:", len(test_data))
    test_data = extract_test_data(test_data)
    print("Number of testing data after processing:", len(test_data))
    # print("\nFirst train data:\n", test_data[0])
    # print("\nSecond train data:\n", test_data[1])

    with open("./data_after_pred_serv/schema.json") as f:
        label_file = json.load(f)
    label_fn = DSTlabel(label_file)
    label_dict = label_fn.collect_label()
    service_dict = label_fn.get_service()
    
    #Process utterance
    test_utterance = create_utterance(test_data, service_dict, tokenizer, 'test') #tokenize  
    testData = finalDataset(test_utterance, 'test')
    testloader = DataLoader(testData, batch_size = 1, shuffle = False)

    
    best_model_path = args.ckpt_dir / "service_best.pt"
    model = serviceModel()
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    model = model.to(device)

    pred_dict = serv_pred(model, testloader, test_data, device)
    
    if args.mode == 'test_unseen':
        path = 'test_unseen_servid'
    else:
        path = 'test_seen_servid'
    postprocess(pred_dict, args.test_file, args.processed_test_data_dir/path, args.mode)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Directory to the dataset.",
        default="./data-0625/test_seen/",
    )
    parser.add_argument(
        "--processed_test_data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data_after_pred_serv",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )


    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=5)
    parser.add_argument("--pretrained_model", type = str, default = 'bert-base-cased')
    parser.add_argument("--mode", type=str, default="test_seen")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)