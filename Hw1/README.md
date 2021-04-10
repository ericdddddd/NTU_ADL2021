# Homework 1 ADL NTU 109 Spring

## File explain
dataset , model for  intent classification
<br>
slot_dataset , slot_model for  slot tagging
<br>
utils.py 產生單字對應的index及生成context vector

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

## Training
```shell
python train_intent.py
python train_slot.py
```
## Testing 
```shell
bash ./intent_cls.sh /path/to/test.json /path/to/pred.csv
# python3.8 test_intent.py --test_file "${1}" --ckpt_path ckpt/intent/best.model --pred_file "${2}"
bash ./slot_tag.sh /path/to/test.json /path/to/pred.csv
#python3.8 test_slot.py --test_file "${1}" --ckpt_path ckpt/slot/best.model --pred_file "${2}"
# 建議使用絕對路徑，使用相對路徑請加上.
```
