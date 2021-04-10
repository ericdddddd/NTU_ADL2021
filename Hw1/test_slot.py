import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import os
import csv

import torch
import logging
from slot_dataset import SlotTagDataset
from slot_model import SlotClassifier
from utils import Vocab
from itertools import zip_longest


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    logging.info(f"--------- processing test data ---------")
    data = json.loads(args.test_file.read_text())
    test_id = [test_id['id'] for test_id in data]

    test_dataset = SlotTagDataset(data, vocab, tag2idx, args.max_len , True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = args.batch_size,
                                            shuffle = False,
                                            num_workers = 2)

    logging.info(f"--------- loading model ---------")
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = torch.load(args.ckpt_path)
    print(model)

    model.eval()
    predict_output = []

    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs.view(-1,args.max_len,outputs.shape[-1])
            predict = torch.argmax(outputs , dim = 2)
            predict = predict.int().tolist()
            for batch_data in predict:
                tags_label = [test_dataset.idx2label[tag] for tag in batch_data if tag != test_dataset.label2idx("Pad")]
                tags_predict = " ".join(elem for elem in tags_label)
                predict_output.append(tags_predict)
    
    # write predict csv
    print("save csv ...")
    d = [test_id, predict_output]
    export_data = zip_longest(*d, fillvalue = '')
    with open(args.pred_file ,'w', encoding="utf-8", newline='') as fp:
      wr = csv.writer(fp)
      wr.writerow(("id", "tags"))
      wr.writerows(export_data)
    fp.close()
    print("Finish Predicting")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/slot/test.json",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/slot/ckpt.model",
    )
    parser.add_argument("--pred_file", type=Path, default="./result/pred_slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=64)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)