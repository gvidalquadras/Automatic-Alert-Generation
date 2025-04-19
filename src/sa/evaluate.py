import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import fasttext
import pandas as pd
import os

from SA import SentimentClassifier, SADataset
from train_SA import collate_fn, load_sentiment140_csv, evaluate
from utils import set_seed

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ft_model = fasttext.load_model("cc.en.300.bin")

    test_texts, test_labels = load_sentiment140_csv("data/sentiment140_test.csv")
    test_dataset = SADataset(test_texts, test_labels, ft_model)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    print("Cargando modelo entrenado...")
    model = torch.load("models/SA_model.pt", map_location=device,weights_only=False)
    model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    print("Evaluando en test...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
