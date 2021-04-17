import torch
import json
import logging
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from transformers import BertTokenizerFast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from context_Train_dataset import TrainingDataset


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

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

    return all_data  ,  context

def preprocess_data(args , train_data , context):
    relevant = train_data[0]['relevant']
    answers = train_data[0]['answers']
    start = answers[0]['start']
    print(start)
    text = context[relevant]
    print(text)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    context_tokens = tokenizer.tokenize(text) # 轉換成BERT格式的token(str)
    print(context_tokens)
    """
    ''' Preprocess Data into training instances for BERT. '''
    instances = []
    max_question_length = 64
    max_input_length = args.input_length
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    #processing data....
    logging.info("data processing to BERT Token")
    """