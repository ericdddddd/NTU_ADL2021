from typing import List, Dict

from torch.utils.data import Dataset
import torch
from utils import Vocab
import torch.nn.functional as F


class SlotTagDataset(Dataset):
    def __init__(
        self,
        data: List[Dict], # split_data 存放dict {text,intent,id}
        vocab: Vocab, #utils.py
        label_mapping: Dict[str, int], # intent2idx
        max_len: int, # sequence max len
        test_data : bool
    ):
        self.origin_data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self.idx2label = {idx: tag for tag, idx in self.label_mapping.items()} # convert idx to intent
        self.max_len = max_len
        self.test_data = test_data
        self.data , self.labels  = self.collate_fn(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        if self.test_data == True:
            return self.data[index]
        else:
            return self.data[index] , self.labels[index]

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]):
        # TODO: implement collate_fn
        # 最後要轉成 torch.tensor
        text = [data['tokens'] for data in samples] # get batch sequence
        word2idx = self.vocab.encode_batch(batch_tokens = text , to_len = self.max_len) #統一至self.max_len , 可對不同batch再優化
        word2idx_tensor = torch.LongTensor(word2idx)
        if self.test_data == False :
            all_tags = [data['tags'] for data in samples] #得到所有train or dev的資料中的tags 每個tags下面還有對應的tag
            labels = [] # 存所有資料的tags
            for tags in all_tags:
                current_labels = [self.label2idx('Pad') for i in range(self.max_len)]
                for i,tag in enumerate(tags):
                    if i >= len(current_labels):
                        break
                    current_labels[i] = self.label2idx(tag)
                labels.append(current_labels)
            labels_tensor = torch.LongTensor(labels)
            print(labels_tensor.size())
            return word2idx_tensor , labels_tensor
        return word2idx_tensor , None

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self.idx2label[idx]