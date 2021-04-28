"""SQUAD: The Stanford Question Answering Dataset."""


import json

import datasets

import random

logger = datasets.logging.get_logger(__name__)


class QA_testConfig(datasets.BuilderConfig):
    """BuilderConfig"""

    def __init__(self, **kwargs):
        """
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(QA_testConfig, self).__init__(**kwargs)


class QA_testDataset(datasets.GeneratorBasedBuilder):
    """SQUAD: The Stanford Question Answering Dataset. Version 1.1."""

    def _info(self):
       return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None
        )

    def _split_generators(self, filepath):
        return [
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": self.config.data_files['test']})
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        random.seed(10)
        # Opening JSON file
        file_path = filepath[0] # include questions and ids file
        context_path = filepath[1] # context file
        predict_id_path = filepath[2] # choose context
        f_train = open(file_path , encoding = "utf-8")
        f_context = open(context_path , encoding = "utf-8")
        f_predict_ids = open(predict_id_path , encoding = "utf-8")
        all_data = json.load(f_train)
        context = json.load(f_context)
        predict_ids = json.load(f_predict_ids)
        
        for data in all_data:
          id_ = data['id']
          choice = predict_ids[id_]
          question = data['question']
          
          if (choice + 1) > len(data['paragraphs']): # 可能為猜錯文章，超出範圍，則random choose 一篇
              random_choice = random.randint(0, len(data['paragraphs'])-1) # relevant_context position
              choose_id = data['paragraphs'][random_choice]
          else:
              choose_id = data['paragraphs'][choice]

          context_ = context[choose_id]
          answer_starts = []
          answers = []
          yield id_,{
            "context": context_,
            "question": question,
            "id": id_,
            "answers": {
                "answer_start": answer_starts,
                "text": answers,
                },
            }