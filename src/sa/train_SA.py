import torch
from torch import Tensor, nn
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import os
from typing import List, Tuple, Any
import fasttext
from src.sa.SA import SADataset, SentimentClassifier

from src.sa.load_data_SA import load_data 
from src.utils import set_seed


def collate_fn(batch: List[Tuple[Tensor, int]]) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Custom collate function for the sentiment analysis DataLoader.
    Extracts the text tensor and labels and pads the text sequences to equal length.

    Args:
        batch (List[Tuple[Tensor, int]]): List of (embedded_text, label) pairs.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            - padded_texts: Padded text tensor of shape (batch_size, max_length).
            - lengths: Tensor of original sequence lengths, shape (batch_size,).
            - labels: Tensor of labels, shape (batch_size, 1).
    """
    
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in texts])
    padded_texts = pad_sequence(texts, batch_first=True)
    labels = torch.tensor(labels).unsqueeze(1)
    return padded_texts, lengths, labels

def load_sentiment140_csv(path:str)-> Tuple[List[str], List[int]]:
    """
    Loads the Sentiment140 dataset from a CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        Tuple[List[str], List[int]]:
            - A list of text samples.
            - A list of corresponding sentiment labels.
    """
    df = pd.read_csv(path)
    return df["text"].tolist(), df["label"].tolist()


def train(
    model: Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Module,
    device: torch.device
    ) -> Tuple[float, float]:
    """
    Trains the sentiment analysis model for one full epoch.

    Performs forward pass, computes the loss, backpropagation, and updates the model's parameters.

    Args:
        model (nn.Module): The PyTorch model.
        loader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the training

    Returns:
        Tuple[float, float]:
            - Average loss over the epoch.
            - Average accuracy over the epoch.
    """
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x, lengths)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = (torch.sigmoid(output) > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


def evaluate(
    model: Module,
    loader: DataLoader,
    criterion: Module,
    device: torch.device
    ) -> Tuple[float, float]:
    """
    Evaluates the sentiment analysis model on validation or test data.

    Computes the loss and accuracy without updating model parameters.

    Args:
        model (nn.Module): The PyTorch model.
        loader (DataLoader): DataLoader for the evaluation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the evaluation on ('cpu' or 'cuda').

    Returns:
        Tuple[float, float]:
            - Average loss over the evaluation set.
            - Average accuracy over the evaluation set.
    """
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, lengths, y in loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            output = model(x, lengths)
            loss = criterion(output, y)
            total_loss += loss.item()
            preds = (torch.sigmoid(output) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss / len(loader), correct / total

def main():
    """
    This function is the main program for training.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading FastText...")
    ft_model = fasttext.load_model("cc.en.300.bin")

    print("Loading Sentiment140...")
    load_data()
    train_texts, train_labels = load_sentiment140_csv("data/sentiment140_train.csv")
    val_texts, val_labels = load_sentiment140_csv("data/sentiment140_validation.csv")
    test_texts, test_labels = load_sentiment140_csv("data/sentiment140_test.csv")

    train_data = SADataset(train_texts, train_labels, ft_model)
    val_data = SADataset(val_texts, val_labels, ft_model)
    test_data = SADataset(test_texts, test_labels, ft_model)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = SentimentClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    print("Starting training...")
    epochs = 5
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    print("\nEvaluating test data...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    # Saving the model
    os.makedirs("models", exist_ok=True)
    model_path = "models/SA_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n SA model saved in: {model_path}")

    
if __name__ == "__main__":
    main()
    