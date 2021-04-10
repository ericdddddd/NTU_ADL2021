# Homework 1 ADL NTU 109 Spring

## file explain
dataset , model for  intent classification
slot_dataset , slot_model for  slot tagging
util.py 產生單字對應的index及生成context vector

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl"
make
# otherwise
pip install -r requirements.txt
```

## Preprocessing
```shell
# To preprocess intent detection and slot tagging datasets
bash preprocess.sh
# without preprocessing and training directly :
bash download.sh
# download embedding matrix , vocabulary pickle and class2idx for two tasks from dropbox
```

## training
```shell
python train_intent.py
python train_slot.py
```
## testing 
```shell
bash ./intent_cls.sh /path/to/test.json /path/to/pred.csv
# python3.8 test_intent.py --test_file "${1}" --ckpt_path ckpt/intent/best.model --pred_file "${2}"
bash ./slot_tag.sh /path/to/test.json /path/to/pred.csv
#python3.8 test_slot.py --test_file "${1}" --ckpt_path ckpt/slot/best.model --pred_file "${2}"
# enter Absolute path recommand
```
