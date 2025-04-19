import torch
from torch.utils.data import Dataset
import numpy as np
import fasttext
import fasttext.util
import torch.nn as nn


class SADataset(Dataset):
    def __init__(self, texts, labels, fasttext_model, max_len=50):
        self.texts = texts
        self.labels = labels
        self.fasttext_model = fasttext_model
        self.max_len = max_len  # Para padding

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx].split()
        label = self.labels[idx]

        embeddings = [self.get_embedding(w) for w in text]
        # Padding
        if len(embeddings) < self.max_len:
            pad_length = self.max_len - len(embeddings)
            embeddings += [np.zeros(self.fasttext_model.get_dimension())] * pad_length
        else:
            embeddings = embeddings[:self.max_len]

        return torch.tensor(embeddings, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def get_embedding(self, word):
        try:
            return self.fasttext_model.get_word_vector(word)
        except:
            return np.zeros(self.fasttext_model.get_dimension())
        

class SentimentClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes=2):
        super(SentimentClassifier, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])  # Último hidden state
        return out

