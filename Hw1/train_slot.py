import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import operator
import logging
from tqdm import trange

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2

from slot_model import SlotClassifier
from slot_dataset import SlotTagDataset
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# token Accuracy
def evaluation(outputs, labels , tag_pad_idx):
    predict = torch.argmax(outputs , dim = 1)
    non_pad_elements = (labels != tag_pad_idx)
    correct = torch.sum(torch.eq(predict[non_pad_elements], labels[non_pad_elements])).item()
    return correct / torch.sum(non_pad_elements == True)
# generate report and joint Accuracy
def test(outputs , labels, predict_output , truth_labels , dev_dataset):

    predict = torch.argmax(outputs , dim = 2)
    predict = predict.int().tolist()
    # predict labels
    for batch_data in predict:
        tags_label = [dev_dataset.idx2label[tag] for tag in batch_data if tag != dev_dataset.label2idx("Pad")]
        predict_output.append(tags_label)

    labels = labels.view(-1,args.max_len)
    labels = labels.int().tolist()
    # ground truth
    for batch_label in labels:
        tags_label = [dev_dataset.idx2label[tag] for tag in batch_label if tag != dev_dataset.label2idx("Pad")]
        truth_labels.append(tags_label)

    return predict_output , truth_labels

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    # shape : dict , devide train , dev 
    print(len(data['train']))
    print(len(data['eval']))
    """
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    """
    # split : train , dev split_data : train data and dev data (text, intent, id) 15000 , 3000

    # TODO: create DataLoader for train / dev datasets

    logging.info(f"--------- processing train data ---------")
    train_dataset = SlotTagDataset(data['train'], vocab, tag2idx, args.max_len,False)
    logging.info(f"--------- processing dev data ---------")
    dev_dataset = SlotTagDataset(data['eval'], vocab, tag2idx, args.max_len,False)

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = args.batch_size,
                                            shuffle = True,
                                            num_workers = 2)

    dev_loader = torch.utils.data.DataLoader(dataset = dev_dataset,
                                            batch_size = args.batch_size,
                                            shuffle = False,
                                            num_workers = 2)


    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    print(embeddings.shape)
    # TODO: init model and move model to target device(cpu / gpu)
    model = SlotClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        train_dataset.num_classes,
        vocab.pad_id)

    print(model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # TODO: init optimizer
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    t_batch = len(train_loader) 
    v_batch = len(dev_loader)
    total_loss, total_acc, best_acc = 0, 0, 0
    batch_size = args.batch_size

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        total_loss, total_acc = 0, 0
        # TODO: Training loop - iterate over train dataloader and update model weights
        # 這段做 training
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, dtype=torch.long) # device 為 "cuda"，將 inputs 轉成 torch.cuda.LongTensor
            labels = labels.to(device, dtype=torch.long) # device為 "cuda"，將 labels 轉成 torch.cuda.FloatTensor，因為等等要餵進 criterion，所以型態要是 float
            
            optimizer.zero_grad() # 由於 loss.backward() 的 gradient 會累加，所以每次餵完一個 batch 後需要歸零
            outputs = model(inputs) # 將 input 餵給模型
            outputs = outputs.view(-1, outputs.shape[-1]) # [batch size * sent len, output dim]
            #print(outputs.size())
            labels = labels.view(-1) # [batch size * sent len]
            #print(labels.size())
            loss = criterion(outputs, labels) # 計算此時模型的 training loss
            #outputs.view(args.batch_size , -1 ,outputs.shape[-1]), labels.view(args.batch_size,-1)
            loss.backward() # 算 loss 的 gradient
            optimizer.step() # 更新訓練模型的參數
            correct = evaluation(outputs, labels,train_dataset.label2idx("Pad")) # 計算此時模型的 training accuracy
            total_acc += correct
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
            	epoch+1, i+1, t_batch, loss.item(), correct * 100), end='\r')
        print('\nTrain | Loss:{:.5f} token accuracy: {:.3f}'.format(total_loss/t_batch, total_acc/t_batch * 100))

        # 這段做 validation
        model.eval() # 將 model 的模式設為 eval，這樣 model 的參數就會固定住
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            y_pred = []
            y_true = []
            for i, (inputs, labels) in enumerate(dev_loader):
                inputs = inputs.to(device, dtype=torch.long) 
                labels = labels.to(device, dtype=torch.long)
                outputs = model(inputs) # 將 input 餵給模型
                outputs = outputs.view(-1, outputs.shape[-1]) # [batch size * sent len, output dim]
                labels = labels.view(-1) # [batch size * sent len]
                loss = criterion(outputs, labels) # 計算此時模型的 validation loss
                correct = evaluation(outputs, labels,train_dataset.label2idx("Pad")) # 計算此時模型的 validation accuracy
                y_pred , y_true = test(outputs.view(-1,args.max_len,outputs.shape[-1]),
                labels, y_pred , y_true , dev_dataset)
                total_acc += correct
                total_loss += loss.item()
        
            print("Valid | Loss:{:.5f}".format(total_loss/v_batch))
            joint_acc = 0
            for i in range(len(y_true)):
                if str(y_pred[i]) == str(y_true[i]):
                    joint_acc += 1
            print("Valid | token accuracy: {:.3f} ".format(total_acc/v_batch*100))
            print("Valid | Joint Accuracy : {:.1f}%".format(joint_acc / len(y_true) * 100))
            print(classification_report(y_true, y_pred, mode='strict', scheme=IOB2))
            """
            if total_acc > best_acc:
                # 如果 validation 的結果優於之前所有的結果，就把當下的模型存下來以備之後做預測時使用
                best_acc = total_acc
                #torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir,total_acc/v_batch*100))
                torch.save(model, "{}/ckpt.model".format(args.ckpt_dir))
                print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
            """
        print('-----------------------------------------------')
        model.train() # 將 model 的模式設為 train，這樣 optimizer 就可以更新 model 的參數（因為剛剛轉成 eval 模式）
    
    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
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
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default= 64)

    # model
    parser.add_argument("--hidden_size", type=int, default = 512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default = True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default = "cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default = 10)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
