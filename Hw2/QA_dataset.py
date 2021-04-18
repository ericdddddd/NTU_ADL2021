from torch.utils.data import Dataset

class TrainingDataset(Dataset):
    def __init__(self, instances):
        self.instances = instances
    
    def __len__(self):
        return len(self.instances)
        
    def __getitem__(self, index):
        instance = self.instances[index]
        input_ids = instance['input_ids']
        attention_mask = instance['attention_mask']
        token_type_ids = instance['token_type_ids']
        start = instance['start']
        end = instance['end']
        return input_ids, attention_mask, token_type_ids, start , end