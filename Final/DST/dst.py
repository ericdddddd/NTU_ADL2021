# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script contains an example of how to train and test the NeMo SGD-QA Model (https://arxiv.org/abs/2105.08049).
The SGD-QA model is a fast multi-pass schema-guided state-tracking model, that is trained on the Google schema-guided state tracking dataset (https://arxiv.org/abs/1909.05855).
The model takes dialogue as input and outputs the dialogue state, which includes slot-value pairs. 
The model consists of two components: a neural natural language understanding model (NLU), and a rule-based state tracker.
The NLU takes in a dialogue turn and different schema (entity) information options and outputs their match score. The state tracker takes the highest rated entities and composes
the dialogue state across turns.

***Data format***
The SGD-QA requires a JSON schema and dialogue files for each dataset split. 
In the following we will show an example for a service entry in the schema file.
* service_name
* description
* slots
    * name
    * description
    * is_categorical
    * possible values
* intents
    * name
    * description
    * required_slots (not used)
    * is_transactional (not used)
    * optional_slots (not used)
    * result_slots (not used)


In the following we will show an example for a dialogue. More information at https://arxiv.org/abs/1909.05855
* dialogue_id
* services
* turns
    * frames
        * actions
            * act
            * slot
            * values
        * service
        * slots
            * exclusive_end
            * slot
            * start
        * state
            * active_intent
            * requeste_slots
            * slot_values 
    * speaker - [USER, SYSTEM]
    * utterance


***Downloading the dataset***
#   git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git

***Setting the configs***
The model and the PT trainer are defined in a config file that declares multiple important sections.
The most important ones are:
    model: All arguments that are related to the Model - language model, SGD-QA encoder and decoder, loss, optimizer,
            schedulers, and datasets/data loaders.
    trainer: Any argument to be passed to PyTorch Lightning including number of epochs, number of GPUs,
            precision level, etc.

This script uses the `/examples/nlp/dialogue_state_tracking/conf/sgdqa_config.yaml` config file
by default. You may update the config file from the file directly. The other option is to set another config file via command-line arguments by `--config-name=CONFIG_FILE_PATH'.


***Model Training***
# python sgd_qa.py
    do_training=True
    model.dataset.data_dir=<DATA_DIR_WITH_JSON_DATA>
    model.dataset.dialogues_example_dir=<DATA_DIR_WITH_PREPROCESSED_DATA>
    model.validation_ds.ds_item=<LIST_OF_SPLITS>
    trainer.max_epochs=<NUM_EPOCHS>
    trainer.gpus=[<CHANGE_TO_GPU_YOU_WANT_TO_USE>]


***Model Evaluation***
#   python sgd_qa.py
    do_training=False
    model.test_ds.ds_item=<LIST_OF_SPLITS>

To load a pretrained checkpoint from the cloud prior to training (e.g. for fine-tuning) or evaluation you can set cfg.from_pretrained=<MODEL_NAME>. You can find all pretrained model names by using 
SGDQAModel.list_available_models(). To load a local checkpoint use model.restore_from(<PATH_TO_CHECKPOINT>)

"""

import os
from utils_seen import re_dataid_check, check_schema
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from sgdqa_model import SGDQAModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from shutil import copyfile
import shutil


@hydra_runner(config_path="conf", config_name="sgdqa_config")
def main(cfg:DictConfig) -> None:
    logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    # build dir and reid data
    # data_dir_path = Path("/home/sharon/ADL/Project/data-0625") 
    data_dir_path = Path("./data_after_pred_serv")
    process_data_dir_path = Path("./processed_data")
    if os.path.exists(process_data_dir_path):
        shutil.rmtree(process_data_dir_path)
        logging.info("Remove old processed data!")
    os.mkdir(process_data_dir_path)
    process_train_data_dir_path = Path(process_data_dir_path/"train")
    process_dev_data_dir_path = Path(process_data_dir_path/"dev")
    process_test_data_dir_path = Path(process_data_dir_path/"test")
    if not os.path.exists(process_train_data_dir_path):
        os.mkdir(process_train_data_dir_path)
    if not os.path.exists(process_dev_data_dir_path):
        os.mkdir(process_dev_data_dir_path)
    if not os.path.exists(process_test_data_dir_path):
        os.mkdir(process_test_data_dir_path)
    # reid data & check
    if cfg.model.dataset.task_name == "unseen_domain":
        copyfile(Path(data_dir_path/"train"/"dialogues_001.json"), Path(process_train_data_dir_path/"dialogues_001.json"))
        copyfile(Path(data_dir_path/"dev"/"dialogues_001.json"), Path(process_dev_data_dir_path/"dialogues_001.json"))
        logging.info("Rebuild test unseen data id and remove dialogues with empty frame")
        re_dataid_check(data_dir_path, process_test_data_dir_path, "test_unseen_servid")
    elif cfg.model.dataset.task_name == "seen_domain":
        copyfile(Path(data_dir_path/"train"/"dialogues_001.json"), Path(process_train_data_dir_path/"dialogues_001.json"))
        copyfile(Path(data_dir_path/"dev"/"dialogues_001.json"), Path(process_dev_data_dir_path/"dialogues_001.json"))
        logging.info("Rebuild test seen data id and remove dialogues with empty frame")
        re_dataid_check(data_dir_path, process_test_data_dir_path, "test_seen_servid")
    else:
        logging.info("Rebuild train data id and remove dialogues with empty frame")
        re_dataid_check(data_dir_path, process_train_data_dir_path, "train")
        logging.info("Rebuild dev data id and remove dialogues with empty frame")
        re_dataid_check(data_dir_path, process_dev_data_dir_path, "dev")
        logging.info("Rebuild test seen data id and remove dialogues with empty frame")
        re_dataid_check(data_dir_path, process_test_data_dir_path, "test_seen_servid")
    
    # copy schema
    # check_schema(data_dir_path, process_data_dir_path)
    if not os.path.isdir(process_train_data_dir_path/"schema.json"):
        copyfile(Path(data_dir_path/"schema.json"), Path(process_train_data_dir_path/"schema.json"))
    if not os.path.isdir(process_dev_data_dir_path/"schema.json"):
        copyfile(Path(data_dir_path/"schema.json"), Path(process_dev_data_dir_path/"schema.json")) 
    if not os.path.isdir(process_test_data_dir_path/"schema.json"):
        copyfile(Path(data_dir_path/"schema.json"), Path(process_test_data_dir_path/"schema.json"))

    if cfg.pretrained_model or (cfg.model.nemo_path and os.path.exists(cfg.model.nemo_path)):
        if cfg.pretrained_model:
            logging.info(f'Loading pretrained model {cfg.pretrained_model}')
            model = SGDQAModel.from_pretrained(cfg.pretrained_model)
        else:
            logging.info(f'Restoring model from {cfg.model.nemo_path}')
            model = SGDQAModel.restore_from(cfg.model.nemo_path)
        if cfg.do_training:
            model.setup_training_data(train_data_config=cfg.model.train_ds)
            model.setup_multiple_validation_data(val_data_config=cfg.model.validation_ds)
    else:
        logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
        model = SGDQAModel(cfg.model, trainer=trainer)

    # print(cfg.dataset.data_dir)

    if cfg.do_training:
        trainer.fit(model)
        if cfg.model.nemo_path:
            model.save_to(cfg.model.nemo_path)
    else:
        data_dir = cfg.model.dataset.get('data_dir', None)
        dialogues_example_dir = cfg.model.dataset.get('dialogues_example_dir', None)

        if data_dir is None or dialogues_example_dir is None:
            raise ValueError('No dataset directory provided. Skipping evaluation. ')
        elif not os.path.exists(data_dir):
            raise ValueError(f'{data_dir} is not found, skipping evaluation on the test set.')
        else:
            model.update_data_dirs(data_dir=data_dir, dialogues_example_dir=dialogues_example_dir)
            model._cfg.dataset = cfg.model.dataset

        if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.ds_item is not None:
            gpu = 1 if cfg.trainer.gpus != 0 else 0
            trainer = pl.Trainer(gpus=gpu)
            model.setup_multiple_test_data(test_data_config=cfg.model.test_ds)
            if model.prepare_test(trainer):
                trainer.test(model)


if __name__ == '__main__':
    main()
