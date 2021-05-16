import jsonlines
import os
import datasets

logger = datasets.logging.get_logger(__name__)

class Hw3_Config(datasets.BuilderConfig):
    """BuilderConfig"""

    def __init__(self, **kwargs):
        """
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Hw3_Config, self).__init__(**kwargs)



class Hw3_sum(datasets.GeneratorBasedBuilder):
    

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    'id': datasets.Value("string"),
                    'document': datasets.Value("string"),
                    'summary': datasets.Value("string"),
                }
            ),
            supervised_keys=None
        )
        
        
    def _split_generators(self, filepath):
 
        print(self.config.data_files)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": self.config.data_files['train']}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": self.config.data_files['validation']}),
        ]

    def _generate_examples(self, filepath):
        with jsonlines.open(filepath, mode='r') as reader:
            for row in reader:
                maintext = row['maintext'].strip().replace('\n', '').replace('\r', '')
                title = row['title'].strip()
                id_ = row['id']
                yield id_,{
                "id": id_,
                "document": maintext,
                "summary": title,
                }