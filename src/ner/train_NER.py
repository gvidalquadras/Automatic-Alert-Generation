import torch
import os
import fasttext
import torch
import numpy as np
import pandas as pd
import ast
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from src.ner.ner import NERDataset, NERModel
from src.ner.load_data_NER import load_data
from src.utils import set_seed

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

        def convert_to_list(x: str) -> List[str]:
            x = x.replace(" ", ",")
            return ast.literal_eval(x)

        tokens: List[List[str]] = data['tokens'].apply(convert_to_list).tolist()
        ner_tags: List[List[str]] = data['ner_tags'].apply(convert_to_list).tolist()

        return tokens, ner_tags 
    except FileNotFoundError:
        print(f"{file_path} not found. Please check the file path.")
        return [], []

def collate_fn(batch: List[Tuple[List[str], List[str]]]) -> Tuple[List[List[str]], torch.Tensor, torch.Tensor]:
    """
    Collate function for NER DataLoader.

    Pads the tag sequences and computes the lengths of token sequences.
    Tokens are returned as-is (for use with embedding lookup outside this function).

    Args:
        batch (List[Tuple[List[str], List[int]]]): List of (tokens, ner_tags) pairs.

    Returns:
        Tuple:
            - tokens (List[List[str]]): List of token sequences.
            - ner_tags_padded (Tensor): Padded tensor of tag indices [batch_size, max_seq_len].
            - lengths (Tensor): Tensor of original sequence lengths [batch_size].
    """
    tokens, ner_tags = zip(*batch)

    ner_tags_padded = pad_sequence(
        [torch.tensor(tags, dtype=torch.long) for tags in ner_tags],
        batch_first=True,
        padding_value=0
    )
    
    lengths = torch.tensor([len(seq) for seq in tokens], dtype=torch.long)

    return list(tokens), ner_tags_padded, lengths


def train_model(model: nn.Module,
                dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: str):
    """
    Trains the NER model for one epoch.

    Performs forward, loss computation, backpropagation, optimizer step,
    and computes macro F1 score on training predictions.

    Args:
        model (nn.Module): The NER model.
        dataloader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        device (str): device.

    Returns:
        Tuple[float, float]:
            - Average training loss.
            - Macro F1 score across the epoch.
    """
    
    running_loss = 0.0
    all_pred_labels = []
    all_tags = []
    
    model.train()
    for tokens, tags, lengths in dataloader:
        
        tags = tags.to(device)
        lengths = lengths.to(device)
        

        # Forward: logits of dimensions [batch, max_seq_len, num_classes]
        logits = model(tokens, lengths)

        logits_flat = logits.view(-1, logits.size(-1))
        pred_labels = logits_flat.argmax(dim=-1) 

        tags_flat = tags.view(-1)
      
        all_pred_labels.extend(pred_labels.cpu().numpy())
        all_tags.extend(tags_flat.cpu().numpy())
        loss = criterion(logits_flat, tags_flat)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        f1 = f1_score(all_tags, all_pred_labels, average='macro',zero_division=1, labels=np.arange(9))

    return (running_loss / len(dataloader)), f1

def evaluate_model(model: nn.Module,
                   dataloader: DataLoader,
                   criterion: nn.Module,
                   device: str):
    
    """
    Evaluates the NER model on validation or test data.

    Computes average loss and macro F1 score over all batches.

    Args:
        model (nn.Module): The NER model.
        dataloader (DataLoader): Validation or test data loader.
        criterion (nn.Module): Loss function.
        device (str): 'cpu' or 'cuda'.

    Returns:
        Tuple[float, float]:
            - Average evaluation loss.
            - Macro F1 score.
    """
    model.eval()
    val_loss = 0.0
    all_pred_labels = []
    all_tags = []

    with torch.no_grad():
        for tokens, tags, lengths in dataloader:

            tags = tags.to(device)
            lengths = lengths.to(device)
            
            logits = model(tokens, lengths)
            logits_flat = logits.view(-1, logits.size(-1))
            tags_flat = tags.view(-1)
            
            pred_labels = logits_flat.argmax(dim=-1)  # [batch*max_seq_len]

            # Imprimimos las predicciones y las etiquetas reales
            #pred_labels = pred_labels.view(tags.size())  # Volvemos a la forma original [batch, seq_len]
            all_pred_labels.extend(pred_labels.cpu().numpy())
            all_tags.extend(tags_flat.cpu().numpy())

            loss = criterion(logits_flat, tags_flat)
            val_loss += loss.item()
    f1 = f1_score(all_tags, all_pred_labels, average='macro', labels=np.arange(9))

    return val_loss / len(dataloader), f1


def main():
    """
    This function is the main program for training.
    """
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading FastText...")
    embedding_model = fasttext.load_model("cc.en.300.bin")
    
    print("Loading Conll2003...")
    load_data()
    train_tokens, train_ner_tags = load_ner_data("data/conll2003_train.csv")
    validation_tokens, validation_ner_tags = load_ner_data("data/conll2003_validation.csv")  

    train_dataset = NERDataset(train_tokens, train_ner_tags)
    validation_dataset = NERDataset(validation_tokens, validation_ner_tags)
    
    hidden_dim = 256
    num_classes = 9  
    lr = 0.001
    weight_decay = 1e-4
    epochs = 10
    batch_size = 16
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    model = NERModel(embedding_model, hidden_dim, num_classes).to(device)
    
    # Definimos el optimizador y la función de pérdida
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  

    print("Starting training...")
    for epoch in range(epochs):
        train_loss, f1_train = train_model(model, train_loader, optimizer, criterion, device)
        val_loss, f1_val = evaluate_model(model, val_loader, criterion, device)  
        
        print(f"Epoch {epoch+1}/{epochs}| Train Loss: {train_loss:.4f}, F1: {f1_train:.4f}| Val Loss: {val_loss:.4f}, F1: {f1_val:.4f}")
        

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/ner_model.pth") 
    print("NER model saved in: ", "models/ner_model.pth")
    
if __name__ == "__main__":
    main()