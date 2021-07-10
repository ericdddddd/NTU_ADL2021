from typing import Dict, Optional

import torch
from torch import nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel


class serviceModel(nn.Module):
    def __init__(self):
        """Get logits for elements by conditioning on input embedding.
        Args:
          num_classes: An int containing the number of classes for which logits are to be generated.
          embedding_dim: hidden size of the BERT
    
        Returns:
          A tensor of shape (batch_size, num_classes) containing the logits.
        """
        super().__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-cased', output_hidden_states = True)
        self.encoder = SGDEncoder(hidden_size=768, dropout=0.1)
        self.decoder = SGDDecoder(embedding_dim = 768, num_classes = 1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        Args:
            encoded_utterance: [CLS] token hidden state from BERT encoding of the utterance
        """

        token_embedding = self.bert_model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask)[0]
        encoded_utterance = self.encoder(hidden_states = token_embedding)
        logit = self.decoder(encoded_utterance = encoded_utterance)
        return logit

class SGDEncoder(nn.Module):
    """
    Neural module which encodes BERT hidden states
    """


    def __init__(
        self, hidden_size: int, activation: str = 'tanh', dropout: float = 0.0, use_transformer_init: bool = True,
    ) -> None:

        """
        Args:
            hidden_size: hidden size of the BERT model
            activation: activation function applied
            dropout: dropout ratio
            use_transformer_init: use transformer initialization
        """
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)


        self.activation = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: bert output hidden states
        """
        first_token_hidden_states = hidden_states[:, 0]
        logits = self.fc(first_token_hidden_states)
        logits = self.activation(logits)
        logits = self.dropout1(logits)
        return logits

class SGDDecoder(nn.Module):

    def __init__(self, embedding_dim : int, num_classes : int) -> None:

        super().__init__()

        self.utterance_proj = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.functional.gelu

        self.layer1 = nn.Linear(embedding_dim, num_classes)

    def forward(self, encoded_utterance):
        utterance_embedding = self.utterance_proj(encoded_utterance)
        utterance_embedding = self.activation(utterance_embedding)

        logits = self.layer1(utterance_embedding)
        return logits
 

