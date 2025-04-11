import torch
import os
import fasttext
from typing import List, Tuple  
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from ner_dataset import load_ner_data, collate_fn, NERDataset
import time
from sklearn.metrics import f1_score
import numpy as np

class NERModel(torch.nn.Module):
    def __init__(self, embedding_model, hidden_dim, num_classes):
        super(NERModel, self).__init__()
        self.embedding_model = embedding_model  # FastText por ejemplo
        self.hidden_dim = hidden_dim

        self.lstm = torch.nn.LSTM(input_size=300, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.linear = torch.nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, tokens: List[List[str]], lengths: torch.Tensor):
        # Convertimos listas de palabras en embeddings usando FastText
        embedded = []
        for sent in tokens:
            vectors = [self.embedding_model.get_word_vector(word) for word in sent]
            embedded.append(torch.tensor(vectors, dtype=torch.float32, device=lengths.device))

        padded = pad_sequence(embedded, batch_first=True, padding_value=0.0)

        packed = torch.nn.utils.rnn.pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(packed)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        logits = self.linear(outputs)
        return logits
    

