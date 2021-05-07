mkdir -p ./model/context && wget -O ./model/context/config.json "https://www.dropbox.com/s/fe1y2w3uopvz8sh/context_config.json?dl=1" 
mkdir -p model/context && wget -O ./model/context/pytorch_model.bin "https://www.dropbox.com/s/ybx4iu3736y2rxn/context_pytorch_model.bin?dl=1"

mkdir -p model/QA && wget -O ./model/QA/config.json "https://www.dropbox.com/s/5gbmqllmbd2476k/QA_config.json?dl=1" 
mkdir -p model/QA && wget -O ./model/QA/pytorch_model.bin "https://www.dropbox.com/s/atgbowec9mjf4ke/QA_pytorch_model.bin?dl=1" 

mkdir -p ./tokenizer && wget -O ./tokenizer/config.json "https://www.dropbox.com/s/z1l1k2nz5pdn24o/config.json?dl=1" 
mkdir -p ./tokenizer && wget -O ./tokenizer/special_tokens_map.json "https://www.dropbox.com/s/d1i13zizhwbhcxc/special_tokens_map.json?dl=1"  
mkdir -p ./tokenizer && wget -O ./tokenizer/tokenizer_config.json "https://www.dropbox.com/s/ft8qgtv5ivf3nou/tokenizer_config.json?dl=1"  
mkdir -p ./tokenizer && wget -O ./tokenizer/vocab.txt "https://www.dropbox.com/s/9p3fxdwxmxbx7q4/vocab.txt?dl=1"  
