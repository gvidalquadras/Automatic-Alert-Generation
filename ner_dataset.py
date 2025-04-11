import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Any
import pandas as pd
import ast
import fasttext  
from gensim.models import KeyedVectors


def load_ner_data(file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Load data from a specified file path, extract tokens and NER tags.

    Parameters:
    file_path (str): The path to the NER dataset file.

    Returns:
    Tuple[List[List[str]], List[List[str]]]: Lists of tokenized texts and their corresponding NER tags.
    """
    try:
        data: pd.DataFrame = pd.read_csv(file_path)

        # Función para convertir las listas de texto a listas reales, reemplazando los espacios por comas
        def convert_to_list(x: str) -> List[str]:
            # Reemplazar los espacios por comas y luego evaluar como una lista
            x = x.replace(" ", ",")
            return ast.literal_eval(x)

        tokens: List[List[str]] = data['tokens'].apply(convert_to_list).tolist()
        ner_tags: List[List[str]] = data['ner_tags'].apply(convert_to_list).tolist()

        return tokens, ner_tags 
    except FileNotFoundError:
        print(f"{file_path} not found. Please check the file path.")
        return [], []

def collate_fn(batch: List[Tuple[List[str], List[str]]]) -> Tuple[List[List[str]], torch.Tensor, torch.Tensor]:
    tokens, ner_tags = zip(*batch)

    ner_tags_padded = pad_sequence(
        [torch.tensor(tags, dtype=torch.long) for tags in ner_tags],
        batch_first=True,
        padding_value=0
    )
    
    lengths = torch.tensor([len(seq) for seq in tokens], dtype=torch.long)

    return list(tokens), ner_tags_padded, lengths



class NERDataset(Dataset):
    def __init__(self, tokens: List[List[str]], ner_tags: List[List[int]]):
        self.tokens = tokens
        self.ner_tags = ner_tags

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.ner_tags[idx]
