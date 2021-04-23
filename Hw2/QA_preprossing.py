import torch
import json
import logging
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from transformers import BertTokenizerFast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

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

def read_test_data(args):
    
    #path
    train_path = args.data_dir + "public.json"
    context_path = args.data_dir + "context.json"
    # Opening JSON file
    logging.info("read public.json and context.json")
    f_train = open(train_path , encoding = "utf-8")
    f_context = open(context_path , encoding = "utf-8")
    all_data = json.load(f_train)
    context = json.load(f_context)
    logging.info("finished read!")

    return all_data  ,  context

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
        answers_data = data['answers']

        for answer in answers_data:

            ans_start = answer['start']
            # print(context[relevant_doc_id])
            ans_text = answer['text']
            ans_tokens = tokenizer.tokenize(ans_text)
            # 在context tokens中找尋答案的位置
            context_tokens_start = ans_start
            context_tokens_end = 0

            for start_index in range(ans_start , -1 , -1):
                if context_tokens_end != 0: # find answer
                    break
                context_tokens_start = start_index
                ans_tokens_index = 0
                for index in range(start_index,len(context_tokens)):
                    if ans_tokens_index == len(ans_tokens):
                        context_tokens_end = index
                        break
                    elif context_tokens[index] == ans_tokens[ans_tokens_index]:
                        ans_tokens_index += 1
                    else:
                        break

            if (context_tokens_end == 0): continue

            # 若context的長度大於最大值 ， 而答案超出最大值外，將前面的內容做刪除留答案的內文，position也需做更正
            current_context_length = len(context_tokens)
            context_max_length = max_input_length - len(question_token_ids)
            
            if(context_tokens_end > context_max_length):
                length_difference = current_context_length - context_max_length
                context_tokens = context_tokens[length_difference:]
                context_tokens_start -= length_difference
                context_tokens_end -= length_difference

            # 在ids中 ans position實際的位置
            ans_tokens_start = context_tokens_start + len(question_token_ids)
            ans_tokens_end = context_tokens_end + len(question_token_ids)
            
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
            ans_tokens_start = torch.LongTensor([ans_tokens_start]).squeeze()
            ans_tokens_end = torch.LongTensor([ans_tokens_end]).squeeze()
            instance = {}
            instance['input_ids'] = input_ids
            instance['attention_mask'] = attention_mask
            instance['token_type_ids'] = token_type_ids
            instance['start'] = ans_tokens_start
            instance['end'] = ans_tokens_end
            instances.append(instance)
            
        print("Progress: %d/%d\r" % (i+1, len(train_data)), end='')
        

    
    logging.info("Finishing convert to BERT Token!")
    #split train , validation data  => split ratio : ( 1 - split ratio)
    data_len = int(len(instances) * args.split_ratio)
    train_data , validation_data =  instances[:data_len] , instances[data_len:]

    return train_data , validation_data

# Dataloader collate_fn 
def collate_fn(batch):
        input_ids, attention_mask, token_type_ids, start , end = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True)
        start = torch.stack(start)
        end = torch.stack(end)
        return input_ids, attention_mask, token_type_ids, start , end
        
"""
        # ans tokens for BERT (先取一個答案)
        answers_data = data['answers']
        ans_start = answers_data[0]['start']
        ans_text = answers_data[0]['text']
        ans_tokens = tokenizer.tokenize(ans_text)

        # 在context tokens中找尋答案的位置
        ans_tokens_index = 0
        context_tokens_start = 0
        context_tokens_end = 0
        for index,token in enumerate(context_tokens):
            if ans_tokens_index == len(ans_tokens):
                context_tokens_end = index
                break
            elif token == ans_tokens[ans_tokens_index]:
                if ans_tokens_index == 0:
                    context_tokens_start = index
                ans_tokens_index += 1
            else:
                context_tokens_start = 0
                ans_tokens_index = 0

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
        # input_ids答案的正確位置
        ans_tokens_start = context_tokens_start + len(question_token_ids)
        ans_tokens_end = context_tokens_end + len(question_token_ids)
        # 若input_ids超過最大長度，則pass這筆資料
        if ans_tokens_end >= max_input_length:
            continue
        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        token_type_ids = torch.LongTensor(token_type_ids)
        ans_tokens_start = torch.LongTensor([ans_tokens_start]).squeeze()
        ans_tokens_end = torch.LongTensor([ans_tokens_end]).squeeze()
        instance = {}
        instance['input_ids'] = input_ids
        instance['attention_mask'] = attention_mask
        instance['token_type_ids'] = token_type_ids
        instance['start'] = ans_tokens_start
        instance['end'] = ans_tokens_end
        instances.append(instance)
        print("Progress: %d/%d\r" % (i+1, len(train_data)), end='')
"""