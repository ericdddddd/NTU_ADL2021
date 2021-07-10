import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from torch.utils.data import DataLoader, Dataset 
import pytorch_lightning as pl
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import torch.optim as optim
from torch.autograd import Variable
from random import sample
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer

import torch
from tqdm import tqdm
from tqdm import trange
import numpy as np
import torch.nn as nn

from utils_unseen import extract_need_data, read_data, DSTlabel, EmbedDataset, create_utterance, create_data, finalDataset
from model_clf import serviceModel



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data-0625/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        help="Save the processed data",
        default="./clf_data/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=64)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=1)
    parser.add_argument("--pretrained_model", type = str, default = 'bert-base-cased')

    args = parser.parse_args()
    return args



def main(args) -> None:
    torch.cuda.set_device(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.current_device())

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    emb_model = BertModel.from_pretrained(args.pretrained_model, output_hidden_states = True)
    # tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_model)
    # emb_model = RobertaModel.from_pretrained(args.pretrained_model, output_hidden_states = True)

    print("--------------- {:15} ---------------".format("training data"))
    train_data = read_data(args.data_dir, mode='train')
    print("Number of training data:", len(train_data))
    train_data = extract_need_data(train_data)
    print("Number of training data after processing:", len(train_data))
    print("--------------- {:^15} ---------------".format("dev data"))
    dev_data = read_data(args.data_dir, mode='dev')
    print("Number of development data:", len(dev_data))
    dev_data = extract_need_data(dev_data)
    print("Number of dev data after processing:", len(dev_data))
    print("\nFirst train data:\n", train_data[0])
    print("\nSecond train data:\n", train_data[1])

    train_data = sample(train_data, 10000)
    dev_data = sample(dev_data, 2000)




    with open(args.data_dir/"schema.json") as f:
        label_file = json.load(f)
    label_fn = DSTlabel(label_file)
    label_dict = label_fn.collect_label()
    service_dict = label_fn.get_service()
    

    #Process utterance
    trainset_path = args.save_dir/'train_utterance.pkl'
    devset_path = args.save_dir/'dev_utterance.pkl'
    if Path(trainset_path).exists() and Path(devset_path).exists():
        with open(trainset_path, 'rb') as f:
            train_utterance = pickle.load(f)
        with open(devset_path, 'rb') as f:
            dev_utterance = pickle.load(f) 
    
    else:
        train_utterance = create_utterance(train_data, service_dict, tokenizer) #tokenize
        dev_utterance = create_utterance(dev_data, service_dict, tokenizer) #tokenize  

        file = open(trainset_path, "wb")
        pickle.dump(train_utterance, file)
        file.close()
        
        file = open(devset_path, "wb")
        pickle.dump(dev_utterance, file)
        file.close()

    trainData = finalDataset(train_utterance)
    trainloader = DataLoader(trainData, batch_size = 1, shuffle = False)

    devData = finalDataset(dev_utterance)
    devloader = DataLoader(devData, batch_size = 1, shuffle = False)
    

    model = serviceModel()
    model = model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 1e-2,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr = args.lr,eps=1e-8, betas=(0.9, 0.999))
    # optimizer = optim.Adam(model.parameters(),lr=args.lr, eps=1e-8)

    gradient_accumulation_steps = 32
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(trainloader)// gradient_accumulation_steps * args.num_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()

    minloss = 100000
    
    train_cnt = len(train_data)
    dev_cnt = len(dev_data)
    del train_data
    del dev_data
    del train_utterance
    del dev_utterance


    print('Start training.........')
    for epoch in range(args.num_epoch):
        train_loss, train_pred = run_train(model, trainloader, device, optimizer, criterion, gradient_accumulation_steps, scheduler)
        dev_loss, dev_pred = run_valid(model, devloader, criterion, device)

        print("Epoch {:^3} | Training: loss = {} | accuracy = {} | Dev: loss = {} | accuracy = {}".format(epoch+1, train_loss, train_pred / train_cnt, dev_loss, dev_pred / dev_cnt))
        if dev_loss < minloss:
            trigger = 0
            minloss = dev_loss
            # torch.save(model.state_dict(), (args.ckpt_dir/"unseen_model.pt"))
            print("Loss降低了啦")
        else:
            trigger += 1
        
        if trigger > 10:
            break
        # model_name = 'clf_model_L_' + str(epoch+1) + '.pt'
        # torch.save(model.state_dict(), (args.ckpt_dir/model_name))
        # print("save new model!")



def run_train(model, trainloader, device, optimizer, criterion, gradient_accumulation_steps, scheduler):
    step = 0
    loss_accu = 0
    total_loss = 0
    model.train()
    pred = []
    correct = 0
    for batch_id, batch in enumerate(tqdm(trainloader, desc = 'Training')):
        dialogue_id, turn_id, text_set, services, labels = batch
        # labels = torch.tensor([0 for i in range(len(dialog_id))]).to(device)
        pred_list = list()
        output_list = list()
        for i, text in enumerate(text_set):
            step += 1
            input_ids = text['input_ids'].squeeze(0).to(device)
            token_type_ids = text['token_type_ids'].squeeze(0).to(device)
            attention_mask = text['attention_mask'].squeeze(0).to(device)
            label = torch.FloatTensor([labels[i]]).to(device)
            output = model(input_ids, token_type_ids, attention_mask)
            # label = torch.flatten(labels, start_dim=0, end_dim=-1)
            output = output.squeeze(0)
            loss = criterion(output, label)

            loss = loss/gradient_accumulation_steps
            loss.backward()
                
            if step == gradient_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step() # update all parameters
                scheduler.step()
                optimizer.zero_grad() # initialize the gradient so that it wont repeat accumulate itself(update the params)
                model.zero_grad()
                step = 0
                

            total_loss += loss
            pred_list.append(int(torch.round(torch.sigmoid(output)).item()))


        labels = [l.item() for l in labels]
        if pred_list == labels:
            correct += 1

        # if batch_id % 100 == 0:
        #     print('prediction : ', pred_list)
        #     print('label : ', labels)
        #     print('--------------------------------------------------------------------------------')

        # prediction = torch.argmax(pred_list)
        # pred.append(prediction.item())
        # # print(predition)
        # total_loss += loss_all

        # if gradient_accumulation_steps > 1:
        #     loss = loss / gradient_accumulation_steps
        #     total_loss += loss_all
        # else:
        #     total_loss += loss_all
        
        # loss.backward()
        # if (batch_id + 1) % gradient_accumulation_steps == 0:
        #     # Clip the norm of the gradients to 1.0.
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        #     optimizer.step() # update all parameters
        #     optimizer.zero_grad() # initialize the gradient so that it wont repeat accumulate itself(update the params)
        #     model.zero_grad()
            # print("Training: in batch:%s | loss:%s"%(str(batch_id),str(loss.item())),end = "\r")
    return total_loss/len(trainloader), correct 

def run_valid(model, devloader, criterion, device):
    total_loss = 0
    model.eval()
    pred = []
    correct = 0
    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(devloader, desc = 'Validation')):
            dialogue_id, turn_id, text_set, services, labels = batch
            pred_list = list()
            for i, text in enumerate(text_set):
                input_ids = text['input_ids'].squeeze(0).to(device)
                token_type_ids = text['token_type_ids'].squeeze(0).to(device)
                attention_mask = text['attention_mask'].squeeze(0).to(device)
                label = torch.FloatTensor([labels[i]]).to(device)
                output = model(input_ids, token_type_ids, attention_mask)
                output = output.squeeze(0)
                loss = criterion(output, label)
                loss = loss/len(text_set)
                total_loss += loss

                # print('output: ', output)
                # print('label: ', label)

                # pred_list.append(torch.argmax(output).item())
                pred_list.append(int(torch.round(torch.sigmoid(output)).item()))

            labels = [l.item() for l in labels]
            
            if pred_list == labels:
                correct += 1


    return total_loss/len(devloader), correct


def serv_pred(model, testloader, test_data, device):
    model.eval()
    pred = []
    dialog_id_list = list()
    turn_id_list = list()
    services_list = list()
    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(testloader)):
            dialogue_id, turn_id, text_set, services = batch
            services = test_data[batch_id]['services']     
            pred_list = list()
            for i, text in enumerate(text_set):
                input_ids = text['input_ids'].squeeze(0).to(device)
                token_type_ids = text['token_type_ids'].squeeze(0).to(device)
                attention_mask = text['attention_mask'].squeeze(0).to(device)
                output = model(input_ids, token_type_ids, attention_mask)
                output = output.squeeze(0)

                pred_list.append(int(torch.round(torch.sigmoid(output)).item()))

            pred_service = [services[i] for i,p in enumerate(pred_list) if p]
            if pred_service == []:
                pred_service = [services[0]]

            pred.append(pred_service)
            dialog_id_list += dialogue_id
            # turn_id_list += turn_id
            turn_id_list.append(turn_id)
            # print(turn_id)
            # print(turn_id_list)
            services_list.append(services)


    pred_dict = {} # {dialog_id:{turn_id:service}}
    for i in range(len(pred)):
        if dialog_id_list[i] not in pred_dict.keys():
            pred_dict[dialog_id_list[i]] = {}
        # pred_dict[dialog_id_list[i]][turn_id_list[i][0].item()] = services_list[i][pred[i]]
        # pred_dict[dialog_id_list[i]][turn_id_list[i][1].item()] = services_list[i][pred[i]] 
        pred_dict[dialog_id_list[i]][turn_id_list[i][0].item()] = pred[i]
        pred_dict[dialog_id_list[i]][turn_id_list[i][1].item()] = pred[i]
    # print(pred_dict['PMUL0320'])

    return pred_dict


    
def accuracy(pred):
    correct = 0
    for p in pred:
        if p == 0:
            correct += 1
    return correct/len(pred) 


if __name__ == "__main__":
    args = parse_args()
    main(args)

