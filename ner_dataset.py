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
        data: pd.DataFrame = pd.read_csv(file_path)  # Leer el archivo CSV
       
        # Función para convertir las listas de texto a listas reales, reemplazando los espacios por comas
        def convert_to_list(x: str) -> List[str]:
            # Reemplazar los espacios por comas y luego evaluar como una lista
            x = x.replace(" ", ",")
            return ast.literal_eval(x)
       
        # Convertir las cadenas de texto a listas reales
        tokens: List[List[str]] = data['tokens'].apply(convert_to_list).tolist()
        ner_tags: List[List[str]] = data['ner_tags'].apply(convert_to_list).tolist()
       
        return tokens, ner_tags  # Devolver los tokens y las etiquetas de NER
    except FileNotFoundError:
        print(f"{file_path} not found. Please check the file path.")
        return [], []


def word2idx(embedding_model: Any, tweet: List[str]) -> torch.Tensor:
    """
    Converts a tweet to a list of word indices based on an embedding model.

    Args:
        embedding_model (Any): The embedding model (FastText in this case).
        tweet (List[str]): A list of words representing the tweet.

    Returns:
        torch.Tensor: A tensor of word indices corresponding to the words in the tweet.
    """
    word_indices = [embedding_model.get_word_vector(word) for word in tweet]  # FastText usa get_word_vector
    word_indices_tensor = torch.tensor(word_indices, dtype=torch.float32)  # Convertimos a tensor flotante
    return word_indices_tensor


def collate_fn(batch: List[Tuple[List[str], List[str]]], embedding_model: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepares and returns a batch for training/testing in a torch model.

    Args:
        batch (List[Tuple[List[str], List[str]]]): A list of tuples, where each tuple contains a
                                                   list of words (representing a text) and a list of NER tags.
        embedding_model (Any): The embedding model (FastText in this case).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing three elements:
            - texts_padded (torch.Tensor): A tensor of padded word indices of the text.
            - ner_tags_padded (torch.Tensor): A tensor of padded NER tags.
            - lengths (torch.Tensor): A tensor representing the lengths of each text sequence.
    """
    # Sort the batch by the length of text sequences in descending order
    batch.sort(key=lambda x: len(x[0]), reverse=True)
   
    # Unzip texts and ner_tags from the sorted batch
    texts, ner_tags = zip(*batch)
    texts, ner_tags = list(texts), list(ner_tags)
   
    # Convert texts to indices using the word2idx function and the FastText model
    texts_idx = [word2idx(embedding_model, text) for text in texts]
   
    # Calculate the lengths of each text sequence
    lengths = torch.tensor([max(len(text), 1) for text in texts_idx], dtype=torch.long)

    # Pad the text sequences to have uniform length
    texts_padded = pad_sequence(texts_idx, batch_first=True, padding_value=0)

    # Convert ner_tags to tensor using the updated label_map
    label_map = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    ner_tags_int = [[label_map.get(tag, 0) for tag in tags] for tags in ner_tags]
   
    # Pad the ner_tags sequences to match the padded texts
    ner_tags_padded = [torch.tensor(tags, dtype=torch.long) for tags in ner_tags_int]
    ner_tags_padded = pad_sequence(ner_tags_padded, batch_first=True, padding_value=0)

    return texts_padded, ner_tags_padded, lengths


class NERDataset(Dataset):
    """
    A PyTorch Dataset for Named Entity Recognition (NER) tasks.

    Attributes:
        tokens (List[List[str]]): List of tokenized sentences.
        ner_tags (List[List[str]]): List of NER tags corresponding to each token.
    """

    def __init__(self, tokens: List[List[str]], ner_tags: List[List[str]]):
        self.tokens = tokens
        self.ner_tags = ner_tags

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tuple[List[str], List[str]]:
        """
        Returns the tokenized sentence and corresponding NER tags for the specified index.

        Args:
            idx (int): Index of the item.

        Returns:
            Tuple[List[str], List[str]]: A tuple containing the tokens and the corresponding NER tags for idx.
        """
        return self.tokens[idx], self.ner_tags[idx]

