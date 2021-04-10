from typing import Dict

import torch
from torch import nn
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        pad_idx
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embeddings = Embedding.from_pretrained(embeddings, freeze=False , padding_idx= pad_idx) 
        # TODO: model architecture
        self.embedding_dim = embeddings.size(1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = num_class
        self.rnn = nn.LSTM(embeddings.size(1), hidden_size, num_layers = num_layers,bidirectional = bidirectional ,
        batch_first= True)
        self.classifier = nn.Sequential( nn.Dropout(dropout),
                                         nn.Linear(self.hidden_size * 2, self.output_dim)
                                        )

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch):
        inputs = self.embeddings(batch)
        x, _  = self.rnn(inputs,None)
        # x  dimension (batch, seq_len, hidden_size)
        # get rnn final layer hidden state
        #x = x[:, -1, :]
        x = torch.mean(x, dim = 1)
        x = self.classifier(x)
        return x
