from time import time
import numpy  as np
import torch
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForMultipleChoice
import preprossing 
import logging
from tqdm import trange

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    # load data
    train_data , context = preprossing.read_train_data(args)
    # processing data
    train_data , dev_data = preprocessing.preprocess_data(args , train_data , context)
    # load dataloader
    logging.info("generate dataloader....")
    train_dataset = TrainingDataset(train_instances)
    dev_dataset = TrainingDataset(dev_instances)
    train_dataloader = DataLoader(train_dataset, collate_fn = preprossing.collate_fn, shuffle=True, \
                            batch_size = args.batch_size , num_workers = 2)
    dev_dataloader = DataLoader(train_dataset, collate_fn = preprossing.collate_fn, shuffle=True, \
                            batch_size = args.batch_size , num_workers = 2)                        
    logging.info("dataloader OK!")
    # model
    model = BertForMultipleChoice.from_pretrained('bert-base-chinese')
    model.to(device)
    # model parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # optimizer
    optimizer = AdamW(model.parameters(), lr = args.lr)
    optimizer.zero_grad()

    patience, best_dev_loss = 0, 1e10
    best_state_dict = model.state_dict()
    start_time = time()

    for epoch in range(1, args.num_epoch + 1):
        model.train()
        # train step
        for i, batch in enumerate(dataloader, start=1):
            batch = (tensor.to(device) for tensor in batch)
            input_ids, attention_mask, token_type_ids, labels = batch

            # Backpropogation
            loss = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Progress bar with timer ;-)
            elapsed_time = time() - start_time
            elapsed_time = timedelta(seconds=int(elapsed_time))
            print("Epoch: %d/%d | Batch: %d/%d | loss=%.5f | %s      \r" \
                % (epoch, args.num_epoch, i, len(train_dataloader), loss, elapsed_time), end='')
            
        # Save parameters of each epoch
        if save_model_path is not None:
            save_checkpoint_path = "%s/epoch_%d" % (save_model_path, epoch)
            model.save_pretrained(save_checkpoint_path)
            
        # Get avg. loss on development set
        print("Epoch: %d/%d | Validating...                           \r" % (epoch, args.num_epoch), end='')
        total_loss = 0
        model.eval()
        for batch in dev_dataloader:
            batch = (tensor.to(device) for tensor in batch)
            input_ids, attention_mask, token_type_ids, labels = batch
            with torch.no_grad():
                loss = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)[0]
            curr_batch_size = input_ids.shape[0]
            total_loss += loss * curr_batch_size
        dev_avg_loss = total_loss / len(instances)

        elapsed_time = time() - start_time
        elapsed_time = timedelta(seconds=int(elapsed_time))
        print("Epoch: %d/%d | dev_loss=%.5f | %s                      " \
            % (epoch, args.num_epoch , dev_avg_loss, elapsed_time))
        
        # Track best checkpoint and earlystop patience
        if dev_loss < best_dev_loss:
            patience = 0
            best_dev_loss = dev_loss
            best_state_dict = deepcopy(model.state_dict())
            if save_model_path is not None:
                model.save_pretrained(save_model_path)
        else:
            patience += 1
        
        if patience > max_patience:
            print('Earlystop at epoch %d' % epoch)
            break
            
    # Restore parameters with best loss on development set
    model.load_state_dict(best_state_dict)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type= str,
        help="Directory to the dataset.",
        default="./dataset/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/choose_context/",
    )
    parser.add_argument(
        "--split_ratio",
        type= float,
        help= "split ratio for train_dataset",
        default = 0.95,
    )
    parser.add_argument(
        "--input_length",
        type= int,
        help= "BERT token maximum input length",
        default = 512,
    )
    # optimizer
    parser.add_argument("--lr", type=float, default=3e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default = 2)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default = "cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default = 3)

    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)