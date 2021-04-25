"""SQUAD: The Stanford Question Answering Dataset."""


import json

import datasets


logger = datasets.logging.get_logger(__name__)


class Hw2_QAConfig(datasets.BuilderConfig):
    """BuilderConfig"""

    def __init__(self, **kwargs):
        """
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Hw2_QAConfig, self).__init__(**kwargs)


class Hw2_QAdataset(datasets.GeneratorBasedBuilder):
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

        print(self.config.data_files)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": self.config.data_files['train']}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": self.config.data_files['validation']}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        context_path = "./dataset/context.json"
        # Opening JSON file
        file_path = filepath[0] # include questions and ids file
        context_path = filepath[1] # context file
        f_train = open(file_path , encoding = "utf-8")
        f_context = open(context_path , encoding = "utf-8")
        all_data = json.load(f_train)
        context = json.load(f_context)
        for data in all_data:
          id_ = data['id']
          question = data['question']
          relevant_id = data['relevant']
          context_ = context[relevant_id]
          answers_data = data['answers']
          answer_starts = [answer["start"] for answer in data['answers']]
          answers = [answer["text"] for answer in data['answers']]
          yield id_,{
            "context": context_,
            "question": question,
            "id": id_,
            "answers": {
                "answer_start": answer_starts,
                "text": answers,
                },
            }