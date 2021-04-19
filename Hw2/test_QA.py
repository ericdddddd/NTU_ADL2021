from time import time
from datetime import timedelta
import numpy  as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForQuestionAnswering, BertTokenizerFast
import QA_preprossing
from QA_dataset import TestingDataset
import logging
from test_context import choose_context
from tqdm import trange
from argparse import ArgumentParser, Namespace
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = './ckpt/QA/bert-base-chinese_epoch_2'

def collate_fn(batch):
        input_ids, attention_mask, token_type_ids, start , end = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True)
        return input_ids, attention_mask, token_type_ids

def generate_answers(pred_ans_tokens):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    ans_str = []
    for tokens in pred_ans_tokens:
        ans_tokens = tokenizer.convert_ids_to_tokens(tokens)
        ans = "".join( token for token in ans_tokens)
        ans = ans.replace('#','')
        ans_str.append(ans)
    return ans_str

def main(args):
    # load data and processing
    test_data , context = sec1_preprossing.read_test_data(args) # read data 
    test_instances = sec1_preprossing.preprocess_test_data(args,test_data,context)
    # test data format for BERT choose context model
    pred_contexts = choose_context(args , test_instances)
    # 取出QA所需要的data
    QA_test_instances = {}
    for index , instance in enumerate(test_instances):
        test_instance = {}
        choice = pred_contexts[index]
        test_instance['input_ids'] = instance['input_ids'].T[choice]
        test_instance['attention_mask'] = instance['attention_mask'].T[choice]
        test_instance['token_type_ids'] = instance['token_type_ids'].T[choice]
        QA_test_instances.append(test_instance)

    logging.info("generate dataloader....")
    QA_dataset = TestingDataset(QA_test_instances)
    test_dataloader = DataLoader(QA_dataset, collate_fn = collate_fn, shuffle = False, \
                        batch_size = args.QA_batch_size , num_workers = 2)                 
    logging.info("dataloader OK!")
    
    # model
    model = BertForQuestionAnswering.from_pretrained(model_name)
    model = model.to(device)
    # run model

    start_time = time()
    pred_ans_tokens = []
    t_batch = len(test_dataloader)
    model.eval()
    logging.info("run BERT QA model !!!")
    with torch.no_grad():
        for i,batch in enumerate(test_dataloader):
            ans_tokens = [] # 猜測的答案token，最後須變回string
            batch = (tensor.to(device) for tensor in batch)
            input_ids, attention_mask, token_type_ids = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            start_index = outputs.start_logits
            start_index = torch.argmax(start_index , dim = 1)
            end_index = outputs.end_logits
            end_index = torch.argmax(end_index , dim = 1)
            input_ids = input_ids.cpu().numpy()
            for  i , seq in enumerate(input_ids):
                ans_tokens.append(seq[start_index[i] : end_index[i]]) # 取得答案的部分
            pred_ans_tokens.extend(ans_tokens) # 將batch猜測的答案加進 pred_ans_tokens
            elapsed_time = time() - start_time
            elapsed_time = timedelta(seconds=int(elapsed_time))
            print("\r | Batch: %d/%d |%s" \
                    % (i, t_batch  ,elapsed_time), end='')
    logging.info("test data predict finished!!!")
    answers = generate_answers(pred_ans_tokens)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type= str,
        help="Directory to the dataset.",
        default="./dataset/",
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
    # batch size
    parser.add_argument("--QA_batch_size", type=int, default = 5)
    parser.add_argument("--context_batch_size", type=int, default = 1)
    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default = "cuda:0"
    )
    args = parser.parse_args()
    # args = parser.parse_known_args()[0] # for colab
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
