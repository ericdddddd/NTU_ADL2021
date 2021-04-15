import torch
import json
import logging
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from transformers import BertTokenizerFast
from torch.nn.utils.rnn import pad_sequence

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type= str,
        help="Directory to the dataset.",
        default="./dataset/",
    ),
    parser.add_argument(
        "--split_ratio",
        type= float,
        help= "split ratio for train_dataset",
        default = 0.95,
    ),
    parser.add_argument(
        "--input_length",
        type= int,
        help= "BERT token maximum input length",
        default = 512,
    ),
    args = parser.parse_args()
    return args

def read_train_data(args):

    #path
    train_path = args.data_dir + "train.json"
    context_path = args.data_dir + "context.json"
    # Opening JSON file
    logging.info("read train.json and context.json")
    f_train = open(train_path , encoding = "utf-8")
    f_context = open(context_path , encoding = "utf-8")
    all_data = json.load(f_train)
    context = json.load(f_context)
    logging.info("finished read!")
    #split train , validation data  => split ratio : ( 1 - split ratio)
    #data_len = int(len(all_data) * args.split_ratio)
    #train_data , validation_data =  all_data[:data_len] , all_data[data_len:]

    return all_data  ,  context

def preprocess_data(args , train_data , context):
    ''' Preprocess Data into training instances for BERT. '''
    instances = []
    max_question_length = 64
    max_input_length = args.input_length
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    #processing data....
    for i , data in enumerate(train_data):
        # Make question tokens for BERT
        question_tokens = tokenizer.tokenize(data['question']) # 轉換成BERT格式的token(str)
        if len(question_tokens) > max_question_length:  # truncation
            question_tokens = question_tokens[:max_question_length]
        question_token_ids = tokenizer.convert_tokens_to_ids(question_tokens) # 將token轉換成ids
        question_token_ids.insert(0, tokenizer.cls_token_id)
        question_token_ids.append(tokenizer.sep_token_id)

        paragraphs_ids = data['paragraphs']
        relevant_context_ids = data['relevant']
        paragraphs_ids.remove(relevant_context_ids) # 將正解先抽出，取出錯誤的三篇
        # 取出relevant context 及 三篇錯誤的context做多選

        # 尚須考慮negative context少於3的情況
        neg_context_ids = random.sample(paragraphs_ids, 3)  # random choose 3 context
        relevant_position = random.randint(0, 3) # relevant_context position
        rel_neg_context_ids = neg_context_ids # 將relevant插入
        rel_neg_context_ids.insert(relevant_position, relevant_context_ids)
        labeled_sample = (rel_neg_context_ids, relevant_position) # (list , label)
        # Make input instances for all question/context pairs
        paired_input_ids = []
        paired_attention_mask = []
        paired_token_type_ids = []

        for context_id in labeled_sample[0]:

            # get  4 context token_ids
            context_text = context[context_id]
            context_tokens = tokenizer.tokenize(context_text)
            context_tokens_ids = tokenizer.convert_tokens_to_ids(context_tokens)
            context_tokens_ids.append(tokenizer.sep_token_id)

            # make input sequences for BERT (串接question , context)
            input_ids = question_token_ids + context_tokens_ids
            token_type_ids = [0 for token_id in question_token_ids]
            token_type_ids.extend(1 for token_id in context_tokens_ids)
            if len(input_ids) > max_input_length:  # truncation
                input_ids = input_ids[:max_input_length]
                if input_ids[-1] != tokenizer.sep_token_id:
                    input_ids[-1] = tokenizer.sep_token_id
                token_type_ids = token_type_ids[:max_input_length]
            attention_mask = [1 for token_id in input_ids]
                
            # convert and collect inputs as tensors
            input_ids = torch.LongTensor(input_ids)
            attention_mask = torch.FloatTensor(attention_mask)
            token_type_ids = torch.LongTensor(token_type_ids)
            paired_input_ids.append(input_ids)
            paired_attention_mask.append(attention_mask)
            paired_token_type_ids.append(token_type_ids)

        label = torch.LongTensor([labeled_sample[1]]).squeeze()
        
        # Pre-pad tensor pairs for efficiency
        paired_input_ids = pad_sequence(paired_input_ids, batch_first=True)
        paired_attention_mask = pad_sequence(paired_attention_mask, batch_first=True)
        paired_token_type_ids = pad_sequence(paired_token_type_ids, batch_first=True)

        # collect all inputs as a dictionary
        instance = {}
        instance['input_ids'] = paired_input_ids.T  # transpose for code efficiency
        instance['attention_mask'] = paired_attention_mask.T
        instance['token_type_ids'] = paired_token_type_ids.T
        instance['label'] = label
        instances.append(instance)
        print("Progress: %d/%d\r" % (i+1, len(train_data)), end='')

    return instances

#train_instances = preprocess_df(train_df)
#dev_instances = preprocess_df(dev_df)

#print("num. train_instances: %d" % len(train_instances))
#print("num. dev_instances: %d" % len(dev_instances))
#print("input_ids.T shape:", train_instances[0]['input_ids'].T.shape)
#train_instances[0]['input_ids'].T

if __name__ == "__main__":
    args = parse_args()
    # args.output_dir.mkdir(parents=True, exist_ok=True)
    train_data , context = read_train_data(args)
    preprocess_data(args , train_data  , context)
