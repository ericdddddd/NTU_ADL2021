# Homework 2 ADL NTU 109 Spring - transformers
## Task
給定一問題，並給予多篇文章，需在當中選出最為相關的一篇作為輸出，再從輸出文章中尋找出問題的正解。
## File explain
sec1_preprossing.py : 處理context selection，讀取訓練及測試資料，並且將原始資料轉換成transfomers可用的資料型態。
<br>
context_dataset.py : 用於context selection ， 存放 pytorch dataset 資料的格式，在dataloader時需存取。
<br>
train_context.py , test_context.py : 訓練及測試context model ，詳細參數參見argparse。
<br>
QA_trainingDataset.py , QA_testingDataset.py : transfomers-datasets格式，其中QA_testingDataset需先執行context selection model得到結果才可使用。
<br>
train_QA , train_QA_v2 :皆為訓練QA model，v2加了每經過幾個steps會紀錄chechpoint及驗證集的輸出結果。
<br>
test_QA : 需先執行test_context.py，得到選取context得結果，才可執行此檔案獲得作業2的輸出。
## Environment
```shell
pytorch == 1.7.1
python == 3.8
transfomers == 4.5.0
datasets == 1.6.0 
accelerate == 0.2.1
SpaCy  == 3.0.5
```

## demo
``` shell
bash ./download.sh
# 如果遇到 bash ./download.sh 無法執行的問題，再請回報通知，謝謝!!
bash ./run.sh context路徑 public(privete)路徑 預測檔案路徑
# 建議使用絕對路徑，使用相對路徑請加上./
# 預測檔案路徑的目錄請先建立，否則會造成錯誤
# then you can get the answers
```

## Training
```shell
# train context selection model
python train_context.py
# train QA model
python train_QA.py
```
