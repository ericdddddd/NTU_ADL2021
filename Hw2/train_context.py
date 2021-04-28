from time import time
from datetime import timedelta
import numpy  as np
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice
import sec1_preprossing
from context_Train_dataset import  TrainingDataset
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

def evaluation(outputs, labels):
    predict = torch.argmax(outputs , dim = 1)
    correct = torch.sum(torch.eq(predict, labels)).item()
    return correct

# Dataloader collate_fn 
def collate_fn(batch):
        input_ids, attention_mask, token_type_ids, labels = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True).transpose(1,2).contiguous()  # re-transpose
        attention_mask = pad_sequence(attention_mask, batch_first=True).transpose(1,2).contiguous()
        token_type_ids = pad_sequence(token_type_ids, batch_first=True).transpose(1,2).contiguous()
        labels = torch.stack(labels)
        return input_ids, attention_mask, token_type_ids, labels

def main(args):
    # load data
    train_data , context = sec1_preprossing.read_train_data(args)
    # processing data
    train_instances , dev_instances = sec1_preprossing.preprocess_data(args , train_data , context)
    # load dataloader
    logging.info("generate dataloader....")
    train_dataset = TrainingDataset(train_instances)
    dev_dataset = TrainingDataset(dev_instances)
    train_dataloader = DataLoader(train_dataset, collate_fn = collate_fn, shuffle=True, \
                            batch_size = args.batch_size) # num_workers = 2
    dev_dataloader = DataLoader(dev_dataset, collate_fn = collate_fn, shuffle=True, \
                            batch_size = args.batch_size) # num_workers = 2  
    # on windows , dataloader can't add num_workers may cause some problems !         
    logging.info("dataloader OK!")
    # model
    model = AutoModelForMultipleChoice.from_pretrained(args.model_name)
    print(model)
    model.to(device)
    # model parameters
    total = sum(p.numel() for p in model.parameters())
    print('\nstart training, parameter total:{}\n'.format(total))
    # optimizer
    optimizer = AdamW(model.parameters(), lr = args.lr)
    optimizer.zero_grad()

    # patience, best_dev_loss = 0, 1e10
    # best_state_dict = model.state_dict()
    start_time = time()

    t_batch = len(train_dataloader) 
    v_batch = len(dev_dataloader)


    for epoch in range(1, args.num_epoch + 1):
        total_loss, total_acc, best_acc = 0, 0, 0
        model.train()
        # train step
        for i, batch in enumerate(train_dataloader, start=1):

            batch = (tensor.to(device) for tensor in batch)
            input_ids, attention_mask, token_type_ids, labels = batch
            optimizer.zero_grad()
            # Backpropogation
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            correct = evaluation(outputs[1], labels)
            total_acc += (correct / input_ids.shape[0])
            total_loss += loss.item()
            # Progress bar with timer ;-)
            elapsed_time = time() - start_time
            elapsed_time = timedelta(seconds=int(elapsed_time))
            print("Epoch: %d/%d | Batch: %d/%d | loss=%.5f | %s      \r" \
                % (epoch, args.num_epoch, i, len(train_dataloader), loss, elapsed_time), end='')

        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/t_batch, total_acc/t_batch*100))
        # Save parameters of each epoch
        filename = "%s_epoch_%d" % (model_name, epoch)
        model.save_pretrained(args.ckpt_dir / filename)

        # Get avg. loss on development set
        print("Epoch: %d/%d | Validating...                           \r" % (epoch, args.num_epoch), end='')
        dev_total_loss = 0
        dev_total_acc = 0
        model.eval()
        for batch in dev_dataloader:
            batch = (tensor.to(device) for tensor in batch)
            input_ids, attention_mask, token_type_ids, labels = batch
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
                loss = outputs[0]
                correct = evaluation(outputs[1], labels)
                dev_total_acc += (correct / input_ids.shape[0])
                dev_total_loss += loss
        dev_avg_loss = dev_total_loss / v_batch

        elapsed_time = time() - start_time
        elapsed_time = timedelta(seconds=int(elapsed_time))
        print("Epoch: %d/%d | dev_loss=%.5f | dev_acc=%.3f |%s                      " \
            % (epoch, args.num_epoch , dev_avg_loss ,dev_total_acc/v_batch*100, elapsed_time))

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--train_file",
        type= str,
        help="Directory to the dataset.",
        default="./dataset/train.json",
    )
    parser.add_argument(
        "--context_file",
        type= str,
        help="Directory to the dataset.",
        default="./dataset/context.json",
    )
    parser.add_argument(
        "--cache_dir",
        type = Path,
        help="Directory to the preprocessed caches.",
        default="./cache/choose_context/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type = Path,
        help="Directory to save the model file.",
        default="./ckpt/choose_context/",
    )
    parser.add_argument(
        "--model_name",
        type = str,
        help = "BERT model_name",
        default = 'hfl/chinese-roberta-wwm-ext',
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default= 'hfl/chinese-roberta-wwm-ext',
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--split_ratio",
        type = float,
        help = "split ratio for train_dataset",
        default = 0.95,
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