# ADL NTU 109 Spring Final Project
R09725025 陳鈺淇
R08725048 王語萱
M10915045 施信宏	
M10915080 羅笠程

## Introduction
在 Task-oriented dialogue system 中 Dialogue State Tracking 負責非常重要的任務，Dialogue State Tracking 會從 user 和 system 的dialogue中提取重要的資訊，以進行後續的處理及回應，通常會使用 Large-scale human-human conversational corpus來訓練，目前常見之 Large-scale human-human conversational corpus 有 MultiWOZ 和 SGD。 在Project中，我們會先預測每個 turn 的 service，接著再使用 DST model 來產生 turn 的 state。

## Download data and model
model 會被下載至 ckpt 資料夾，需注意如果train過sgd-qa model，model就會被覆蓋，需要重新執行download.sh
```shell
bash download.sh
```

## Predict test seen data
- /path/to/data_dir 為存放所有 test seen data 的資料夾
- /path/to/output.csv 為預測結果的存放路徑
```shell
bash run_seen.sh /path/to/data_dir /path/to/output.csv
```

## Predict test unseen data
- /path/to/data_dir 為存放所有 test unseen data 的資料夾
- /path/to/output.csv 為預測結果的存放路徑
```shell
bash run_unseen.sh /path/to/data_dir /path/to/output.csv
```

## Train service prediction model
```shell
python train_unseen.py --data_dir /path/to/data
```

## Train sgd-qa model
- /path/to/data 為train,dev和test data所在的資料夾，需注意在該資料夾內必須有三個子資料夾"train","dev"和"test_seen"，分別存放三種data
- train 完的 model 會被存在 ckpt 資料夾中，檔案名為 best.pt
```shell
python dst.py --data_dir /path/to/data
```

## Predict service
- /path/to/data_dir 為存放所有 test data 的資料夾
- /path/to/output_dir 為預測結果的存放資料夾
```shell
python test_unseen.py --data_dir /path/to/data --output_dir /path/to/output_dir 
```