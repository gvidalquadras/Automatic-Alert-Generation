import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import fasttext
from SA import SADataset, SentimentClassifier
import os

def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in texts])
    padded_texts = pad_sequence(texts, batch_first=True)
    labels = torch.tensor(labels).unsqueeze(1)
    return padded_texts, lengths, labels

def load_sentiment140_csv(path):
    df = pd.read_csv(path)
    return df["text"].tolist(), df["label"].tolist()

def train(model, loader, optimizer, criterion, device):
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

def evaluate(model, loader, criterion, device):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading FastText...")
    ft_model = fasttext.load_model("cc.en.300.bin")

    print("Loading Sentiment140...")
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

    # Save the model
    os.makedirs("models", exist_ok=True)
    model_path = "models/SA_model.pt"
    torch.save(model, model_path)
    print(f"\n Modelo completo guardado en: {model_path}")

    
if __name__ == "__main__":
    main()
    