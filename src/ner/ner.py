import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Any
from sklearn.metrics import f1_score
import numpy as np


class NERDataset(Dataset):
    def __init__(self, tokens: List[List[str]], ner_tags: List[List[int]]):
        self.tokens = tokens
        self.ner_tags = ner_tags

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.ner_tags[idx]


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
            vectors = np.array([self.embedding_model.get_word_vector(word) for word in sent])
            embedded.append(torch.tensor(vectors, dtype=torch.float32, device=lengths.device))

        padded = pad_sequence(embedded, batch_first=True, padding_value=0.0)

        packed = torch.nn.utils.rnn.pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(packed)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        logits = self.linear(outputs)
        return logits
    

