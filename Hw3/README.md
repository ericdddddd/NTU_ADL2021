# Homework 3 ADL NTU 109 Spring - NLG

## Task

給定文章及標題，訓練出一個能生成文章標題的模型，使用mt5-small model

## File explain

train.py : 訓練 mt5-small 模型，詳細參數見argparse ，並有使用accelerate , datasets套件。
<br>sum_dataset.py  : transformers datasets 格式，讀取 jsonl格式的訓練及測試檔案。
<br>test.py  : 讀取訓練好的 model , tokenizer，對public.jsonl , private.jsonl 產生標題 ，詳細參數見argparse。
<br>
test_dataset.py : 讀取測試檔案，放入test.py進行預測。

## Environment

```shell
pytorch == 1.7.1
python == 3.8
transfomers == 4.5.0
datasets == 1.6.0 
accelerate == 0.2.1
SpaCy  == 3.0.5
rouge, spacy, nltk, ckiptagger, tqdm, pandas, jsonlines
```

## demo

``` shell
bash ./download.sh
# 如果遇到 bash ./download.sh 無法執行的問題，再請回報通知，謝謝!!
bash ./run.sh 測試檔案路徑  預測檔案路徑
# 建議使用絕對路徑，使用相對路徑請加上./
# 預測檔案路徑的目錄請先建立，否則會造成錯誤
# then you can get the title
```

## Training

```shell
# train mt5-small model
python train.py # 需放入訓練、驗證集檔案的路徑
```
