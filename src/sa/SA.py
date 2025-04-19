import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np


class SADataset(Dataset):
    def __init__(self, texts, labels, embedding_model):
        self.texts = texts
        self.labels = labels
        self.embedding_model = embedding_model

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx].split()
        vectors = np.array([self.embedding_model.get_word_vector(tok) for tok in tokens])
        embeddings = torch.tensor(vectors, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return embeddings, label

class SentimentClassifier(nn.Module):
    def __init__(self, embedding_dim=300, hidden_dim=128, dropout_prob=0.5):
        super(SentimentClassifier, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.hidden_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        mask = torch.arange(unpacked.size(1), device=lengths.device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(-1).float()

        masked_out = unpacked * mask
        mean_pooled = masked_out.sum(dim=1) / lengths.unsqueeze(1)

        expanded_mask = mask.expand_as(masked_out)
        masked_out[~expanded_mask.bool()] = float('-inf')
        max_pooled, _ = masked_out.max(dim=1)

        pooled = torch.cat([mean_pooled, max_pooled], dim=1)
        hidden = F.relu(self.hidden_proj(pooled))
        hidden = self.dropout(hidden)
        logits = self.output_layer(hidden)
        return logits

    def predict(self, text, embedding_model, device):
        self.eval()
        tokens = text.split()
        vectors = np.array([embedding_model.get_word_vector(tok) for tok in tokens])
        x = torch.tensor(vectors, dtype=torch.float32).unsqueeze(0).to(device)
        lengths = torch.tensor([len(tokens)]).to(device)
        with torch.no_grad():
            output = self.forward(x, lengths)
            prob = torch.sigmoid(output).item()
            label = int(prob > 0.5)
        return label, prob