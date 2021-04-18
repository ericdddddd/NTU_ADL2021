from time import time
from datetime import timedelta
import numpy  as np
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertForQuestionAnswering
import QA_preprossing
from QA_dataset import TrainingDataset
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
model_name = './ckpt/QA/bert-base-chinese_epoch_2'

def main(args):
    # load data
    train_data , context = QA_preprossing.read_train_data(args)
    train_instances , dev_instances = QA_preprossing.preprocess_data(args, train_data , context)
    # load dataloader
    logging.info("generate dataloader....")
    train_dataset = TrainingDataset(train_instances)
    dev_dataset = TrainingDataset(dev_instances)
    train_dataloader = DataLoader(train_dataset, collate_fn = QA_preprossing.collate_fn, shuffle=True, \
                            batch_size = args.batch_size) # num_workers = 2
    dev_dataloader = DataLoader(dev_dataset, collate_fn = QA_preprossing.collate_fn, shuffle=True, \
                            batch_size = args.batch_size) # num_workers = 2
    # on windows , dataloader can't add num_workers , otherwise may cause some problems !                        
    logging.info("dataloader OK!")
    # model
    model = BertForQuestionAnswering.from_pretrained(model_name)
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

    for epoch in range(3, 4):
        total_loss, total_acc = 0, 0
        model.train()
        # train step
        for i, batch in enumerate(train_dataloader, start=1):

            batch = (tensor.to(device) for tensor in batch)
            input_ids, attention_mask, token_type_ids, start , end = batch
            optimizer.zero_grad()
            # Backpropogation
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                start_positions = start , end_positions = end)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # Progress bar with timer ;-)
            elapsed_time = time() - start_time
            elapsed_time = timedelta(seconds=int(elapsed_time))
            print("Epoch: %d/%d | Batch: %d/%d | loss=%.5f | %s      \r" \
                % (epoch, args.num_epoch, i, len(train_dataloader), loss, elapsed_time), end='')

        print('\nTrain | Loss:{:.5f}'.format(total_loss/t_batch))
        # Save parameters of each epoch
        filename = "%s_epoch_%d" % (model_name, epoch)
        model.save_pretrained(args.ckpt_dir / filename)

        # Get avg. loss on development set
        print("Epoch: %d/%d | Validating...                           \r" % (epoch, args.num_epoch), end='')
        dev_total_loss = 0
        model.eval()
        for batch in dev_dataloader:
            batch = (tensor.to(device) for tensor in batch)
            input_ids, attention_mask, token_type_ids, start , end = batch
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                 start_positions = start , end_positions = end)
                loss = outputs.loss
                dev_total_loss += loss
        dev_avg_loss = dev_total_loss / v_batch

        elapsed_time = time() - start_time
        elapsed_time = timedelta(seconds=int(elapsed_time))
        print("Epoch: %d/%d | dev_loss=%.5f |%s                      " \
            % (epoch, args.num_epoch , dev_avg_loss , elapsed_time))


    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type= str,
        help="Directory to the dataset.",
        default="./dataset/",
    )
    parser.add_argument(
        "--cache_dir",
        type = Path,
        help="Directory to the preprocessed caches.",
        default="./cache/QA/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type = Path,
        help="Directory to save the model file.",
        default="./ckpt/QA/",
    )
    parser.add_argument(
        "--split_ratio",
        type = float,
        help = "split ratio for train_dataset",
        default = 0.9,
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
    parser.add_argument("--batch_size", type=int, default = 5)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default = "cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default = 1)

    args = parser.parse_args()
    # args = parser.parse_known_args()[0] # for colab
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)