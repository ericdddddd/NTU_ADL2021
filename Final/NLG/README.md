# ADL-FinalProject-NLG

## Task

學習生成Task-oriented dialogue system的回應內容，並加上chit chat使回覆更加生動。

## File explain

gen_train.py : 前處理資料的內容，生成GPT-2所使用的向量格式，並結合chit chat和response。
<br>gen_train_2.py : 前處理資料的內容，生成GPT-2所使用的向量格式，加上user的state，輔助生成回應，但輸出結果需再做後處理。
<br>gen_train.py : 前處理資料的內容，生成GPT-2所使用的向量格式，只生成chit-chat內容，並決定加在response的前面or後面。
<br>run_language_modeling.py  : 訓練GPT-2 LM model。
<br>gen_eval.py : 產生測試格式資料，原始資料格式須為SGD。
<br>lm.input.test_seen.txt :由gen_eval.py所生成，將test set做處理，run_generation.py所需要的測試檔案。
<br>run_generation.py  : 讀取訓練好的 model , tokenizer，將lm.input.test_seen.txt做預測，是否加上chit chat和生成回應。

## Requirement

```shell
pytorch == 1.7.1
python == 3.8
transfomers == 4.6.0 
accelerate == 0.2.1
apex
rouge, spacy, nltk, ckiptagger, tqdm, pandas, jsonlines
```

## Testing

``` shell
bash ./download.sh # download pretrained GPT-2 chit chat model
if not have processing test file (.txt):
python gen_eval.py --data /* test file dir */ (Dialigue format : SGD)
bash ./run.sh  /*processing test file*/ /*out file(.json)*/
```

## Training

```shell
# train gpt-2 LM model
python run_language_modeling.py \
--output_dir= /* your save model path */ \
--model_type=gpt2 \
--model_name_or_path=gpt2 \
--do_train \
--train_data_file= /* your train file path */ \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 16 \
--num_train_epochs 10 \
--learning_rate 1e-3 \
--fp16 \
--save_strategy epoch \
```
