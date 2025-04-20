import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Any
import numpy as np


class SADataset(Dataset):
    """
    This class is the dataset loading the data for sentiment analysis.

    Attr:
        texts (List[str]): List of input text strings.
        labels (List[int]): List of sentiment labels (e.g., 0 or 1).
        embedding_model (Any): Embedding model used.
    """

    def __init__(self, texts:List[str], labels:List[int], embedding_model:Any):
        """
        Constructor of SADataset.

        Args:
            texts (List[str]): List of input texts.
            labels (List[int]): List of corresponding sentiment labels.
            embedding_model (Any): Embedding model (e.g., FastText) used to obtain word vectors.
        """
        self.texts = texts
        self.labels = labels
        self.embedding_model = embedding_model

    def __len__(self):
        """
        This method returns the length of the dataset.

        Returns:
            int: Total number of text-label pairs.
        """
        return len(self.texts)

    def __getitem__(self, idx:int):
        """
        Returns the embedding tensor and label tensor for the sample at idx.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[Tensor, Tensor]:
                - embeddings (Tensor)
                - label (Tensor)
        """
        tokens = self.texts[idx].split()
        vectors = np.array([self.embedding_model.get_word_vector(tok) for tok in tokens])
        embeddings = torch.tensor(vectors, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return embeddings, label

class SentimentClassifier(nn.Module):
    """
    A sentiment classification model based on a bidirectional LSTM.

    It combines mean and max pooling over LSTM outputs to generate a rich
    sentence representation, followed by a projection and a sigmoid output.

    Attributes:
        lstm (nn.LSTM): Bidirectional LSTM .
        hidden_proj (nn.Linear): Linear layer to project pooled features.
        dropout (nn.Dropout): Dropout layer for regularization.
        output_layer (nn.Linear): Final linear layer producing one logit.
    """
    def __init__(self, embedding_dim=300, hidden_dim=128, dropout_prob=0.5):
        """
        Initializes the SentimentClassifier model.

        Args:
            embedding_dim (int): Dimensionality of input word embeddings.
            hidden_dim (int): Number of hidden units in the LSTM and projection layers.
            dropout_prob (float): Dropout probability.
        """
        super(SentimentClassifier, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.hidden_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim).
            lengths (Tensor): Lengths of the original sequences (before padding), shape (batch_size,).

        Returns:
            Tensor: Output logits of shape (batch_size, 1).
        """
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
        """
        Predicts the sentiment of a given text string.

        Args:
            text (str): Input sentence to classify.
            embedding_model (Any): Embedding model.
            device (torch.device): Device.

        Returns:
            Tuple[int, float]:
                - Predicted label (0 or 1).
                - Predicted probability.
        """
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