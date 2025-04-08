import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import ast

def load_sa_data(csv_path):
    df = pd.read_csv(csv_path)
    sentences = []
    sentiments = []

    for i, row in df.iterrows():
        tokens = ast.literal_eval(row['tokens'])  # convierte de string a lista
        sentiment = int(row['sentiment'])
        sentences.append(tokens)
        sentiments.append(sentiment)

    return sentences, sentiments


def build_vocab(sentences, min_freq=1):
    from collections import Counter
    counter = Counter(word for sent in sentences for word in sent)
    vocab = {word: idx+2 for idx, (word, count) in enumerate(counter.items()) if count >= min_freq}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab