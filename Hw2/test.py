"""
import pickle

with open ('pred_context', 'rb') as fp:
    itemlist = pickle.load(fp)
print(itemlist)

"""
"""
import QA_preprossing
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
train_data , context = QA_preprossing.read_train_data(args)
QA_preprossing.preprocess_data(args, train_data[:10] , context)
"""
for i in range(5,-1,-1):
    print(i)