import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Any
from sklearn.metrics import f1_score
import numpy as np


class NERDataset(Dataset):
    """
    Dataset class for Named Entity Recognition (NER).

    Each item is a tuple containing a list of tokens and a list of corresponding NER tag indices.
    """
    def __init__(self, tokens: List[List[str]], ner_tags: List[List[int]]):
        """
        Constructor of NERDataset.

        Args:
            tokens (List[List[str]]): List of token sequences (sentences).
            ner_tags (List[List[int]]): List of NER tag sequences corresponding to each sentence.
        """
        self.tokens = tokens
        self.ner_tags = ner_tags

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Total number of sentences.
        """
        return len(self.tokens)

    def __getitem__(self, idx):
        """
        Retrieves the tokens and NER tags for the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[List[str], List[int]]: A sentence and its tag sequence.
        """
        return self.tokens[idx], self.ner_tags[idx]


class NERModel(torch.nn.Module):
    """
    Bidirectional LSTM-based model for Named Entity Recognition.

    Takes sequences as input and outputs per-token class logits.
    """
    
    def __init__(self, embedding_model, hidden_dim, num_classes):
        """
        Constructor of the NER model.

        Args:
            embedding_model (Any): Embedding model.
            hidden_dim (int): Number of hidden units in the LSTM.
            num_classes (int): Number of output classes (NER tags).
        """
        super(NERModel, self).__init__()
        self.embedding_model = embedding_model
        self.hidden_dim = hidden_dim

        self.lstm = torch.nn.LSTM(input_size=300, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.linear = torch.nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, tokens: List[List[str]], lengths: torch.Tensor):
        """
        Forward pass of the NER model.

        Args:
            tokens (List[List[str]]): Batch of token sequences.
            lengths (Tensor): Lengths of the original sequences (before padding).

        Returns:
            Tensor: Logits for each token in the batch [batch_size, max_seq_len, num_classes].
        """

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
    

