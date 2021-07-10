wget https://www.dropbox.com/s/ub85r44lbdudjt9/model_0630_e1_data92.pt?dl=1 -O ckpt/best.pt
wget https://www.dropbox.com/s/husgfu5wyv4cy8h/service_best.pt?dl=1 -O ckpt/service_best.pt
# wget https://www.dropbox.com/s/mnuc1msyopdy1la/dialogues_001.json?dl=1 -O data_after_pred_serv/train/dialogues_001
# wget https://www.dropbox.com/s/mnuc1msyopdy1la/dialogues_001.json?dl=1 -O data_after_pred_serv/dev/dialogues_001

python -c "from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, BertConfig, BERT_PRETRAINED_MODEL_ARCHIVE_LIST, ALL_PRETRAINED_CONFIG_ARCHIVE_MAP, RobertaConfig;tokenizer=BertTokenizer.from_pretrained('bert-base-cased');model=BertModel.from_pretrained('bert-base-cased');tokenizer = AutoTokenizer.from_pretrained('bert-base-cased');automodel = AutoModel.from_pretrained('bert-base-cased');"
pip install -r requirements.txt
pip install git+https://github.com/NVIDIA/NeMo.git