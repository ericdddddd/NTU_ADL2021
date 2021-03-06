from time import time
from datetime import timedelta
import numpy  as np
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice
import logging
from tqdm import trange
from context_dataset import TestingDataset
import sec1_preprossing
from argparse import ArgumentParser, Namespace
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
import json
import os

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
        input_ids, attention_mask, token_type_ids = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True).transpose(1,2).contiguous()  # re-transpose
        attention_mask = pad_sequence(attention_mask, batch_first=True).transpose(1,2).contiguous()
        token_type_ids = pad_sequence(token_type_ids, batch_first=True).transpose(1,2).contiguous()
        return input_ids, attention_mask, token_type_ids

def main(args):
    # load data and processing
    pred_file = args.predict_file
    pred_file_path = args.predict_file[:args.predict_file.rfind('/')]
    if not os.path.exists(pred_file_path):
        print('mkdir ' + pred_file_path)
        os.mkdir(pred_file_path)
    test_data , context , ids = sec1_preprossing.read_test_data(args) # read data 
    test_instances = sec1_preprossing.preprocess_test_data(args,test_data,context)
    # test data format for BERT choose context model
    logging.info("generate dataloader....")
    test_dataset = TestingDataset(test_instances)
    test_dataloader = DataLoader(test_dataset, collate_fn = collate_fn, shuffle = False, \
                        batch_size = args.context_batch_size) # , num_workers = 2                 
    logging.info("dataloader OK!")

    model = AutoModelForMultipleChoice.from_pretrained(args.model_name)
    model.to(device)
    start_time = time()

    t_batch = len(test_dataloader)
    model.eval()
    pred_contexts = [] # test_dataloader???model???????????????????????????index ,???????????????index?????????tensor???QA
    logging.info("run BERT choose context model !!!")
    for i,batch in enumerate(test_dataloader):
        batch = (tensor.to(device) for tensor in batch)
        input_ids, attention_mask, token_type_ids = batch
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            cls_scores = outputs.logits
            highest_score = torch.argmax(cls_scores , dim = 1).cpu().numpy()
            pred_contexts.extend(highest_score)
            elapsed_time = time() - start_time
            elapsed_time = timedelta(seconds=int(elapsed_time))
            print("\r | Batch: %d/%d |%s" \
                        % (i, t_batch  ,elapsed_time), end='')
    logging.info("test data predict finished!!!")

    logging.info("write context slection json.....")
    predict_ids = {}
    for index in range(len(pred_contexts)):
        predict_ids[ids[index]] = int(pred_contexts[index])
    with open( args.pred_dir / 'context_predict.json', 'w', encoding='utf-8') as f:
        json.dump(predict_ids, f, ensure_ascii=False, indent=4)
    

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type= str,
        help="Directory to the dataset.",
        default="./dataset/public.json",
        required = True
    )
    parser.add_argument(
        "--context_file",
        type= str,
        help="Directory to the dataset.",
        default="./dataset/context.json",
        required = True
    )
    parser.add_argument(
        "--predict_file",
        type= str,
        help="store final result.",
        default = None ,
        required = True
    )
    parser.add_argument(
        "--pred_dir",
        type = Path,
        help="Directory to the predict files.",
        default="./pred_file/",
    )
    parser.add_argument(
        "--input_length",
        type= int,
        help= "BERT token maximum input length",
        default = 512,
    )
    parser.add_argument(
        "--model_name",
        type = str,
        help = "BERT model_name",
        default = './model/context',
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default= './tokenizer',
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    # batch size
    parser.add_argument("--context_batch_size", type=int, default = 3)
    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default = "cuda:0"
    )
    args = parser.parse_args()
    # args = parser.parse_known_args()[0] # for colab
    return args

if __name__ == "__main__":
    args = parse_args()
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    main(args)

    