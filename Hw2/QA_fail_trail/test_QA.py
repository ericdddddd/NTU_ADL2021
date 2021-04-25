from time import time
from datetime import timedelta
import numpy  as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForQuestionAnswering, BertTokenizerFast
import QA_preprossing
from QA_dataset import TestingDataset , TrainingDataset
import logging
from test_context import choose_context
import sec1_preprossing
from argparse import ArgumentParser, Namespace
from pathlib import Path
import pickle
from torch.nn.utils.rnn import pad_sequence
import json

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'C:/Users/User/Desktop/bert/QA/epoch_3_loss_0.819'

def collate_fn(batch):
        input_ids, attention_mask, token_type_ids = zip(*batch)
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

def preprocess_data(args , train_data , context):
    
    ''' Preprocess Data into training instances for BERT. '''
    instances = []
    max_question_length = 40
    max_input_length = args.input_length
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    #processing data....
    logging.info("data processing to BERT Token")
    
    for i , data in enumerate(train_data):

        # Make question tokens for BERT
        question_tokens = tokenizer.tokenize(data['question']) # 轉換成BERT格式的token(str)
        if len(question_tokens) > max_question_length:  # truncation
            question_tokens = question_tokens[:max_question_length]
        question_token_ids = tokenizer.convert_tokens_to_ids(question_tokens) # 將token轉換成ids
        question_token_ids.insert(0, tokenizer.cls_token_id)
        question_token_ids.append(tokenizer.sep_token_id)

        # context tokens for BERT
        relevant_doc_id = data['relevant']
        context_tokens = tokenizer.tokenize(context[relevant_doc_id])
            
        # convert context ids and concat question_ids
        context_tokens_ids = tokenizer.convert_tokens_to_ids(context_tokens)
        context_tokens_ids.append(tokenizer.sep_token_id) # + [SEP]
        input_ids = question_token_ids + context_tokens_ids
        token_type_ids = [0 for token_id in question_token_ids]
        token_type_ids.extend(1 for token_id in context_tokens_ids)
        if len(input_ids) > max_input_length:  # truncation
            input_ids = input_ids[:max_input_length]
            if input_ids[-1] != tokenizer.sep_token_id:
                input_ids[-1] = tokenizer.sep_token_id
            token_type_ids = token_type_ids[:max_input_length]
        attention_mask = [1 for token_id in input_ids]
           
        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        token_type_ids = torch.LongTensor(token_type_ids)
        instance = {}
        instance['input_ids'] = input_ids
        instance['attention_mask'] = attention_mask
        instance['token_type_ids'] = token_type_ids
        instances.append(instance)
            
    print("Progress: %d/%d\r" % (i+1, len(train_data)), end='')

    return instances

def main(args):
    # load data and processing
    test_data , context = sec1_preprossing.read_test_data(args) # read data 
    # test_instances = sec1_preprossing.preprocess_test_data(args,test_data,context)
    # test data format for BERT choose context model
    # pred_contexts = choose_context(args , test_instances)
    #with open('pred_context', 'rb') as fp:
    #    pred_contexts = pickle.load(fp)
    # 取出QA所需要的data
    """
    QA_test_instances = []
    for index , instance in enumerate(test_instances):
        test_instance = {}
        choice = pred_contexts[index]
        test_instance['input_ids'] = instance['input_ids'].T[choice]
        test_instance['attention_mask'] = instance['attention_mask'].T[choice]
        test_instance['token_type_ids'] = instance['token_type_ids'].T[choice]
        QA_test_instances.append(test_instance)
    """
    QA_test_instances = preprocess_data(args , test_data , context)
    logging.info("generate dataloader....")
    QA_dataset = TestingDataset(QA_test_instances)
    test_dataloader = DataLoader(QA_dataset, collate_fn = collate_fn, shuffle = False, \
                        batch_size = args.QA_batch_size) # , num_workers = 2             
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
            for  index , seq in enumerate(input_ids):
                ans_tokens.append(seq[start_index[index] : end_index[index]]) # 取得答案的部分
            pred_ans_tokens.extend(ans_tokens) # 將batch猜測的答案加進 pred_ans_tokens
            elapsed_time = time() - start_time
            elapsed_time = timedelta(seconds=int(elapsed_time))
            print("\r | Batch: %d/%d |%s" \
                    % (i, t_batch  ,elapsed_time), end='')
    logging.info("test data predict finished!!!")
    answers = generate_answers(pred_ans_tokens)
    ans_dict = {}
    for  i ,data in enumerate(test_data):
        ans_dict[data['id']] = answers[i]
    with open('predict4.json', 'w', encoding='utf-8') as f:
        json.dump(ans_dict, f, ensure_ascii=False, indent=4)


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
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    main(args)