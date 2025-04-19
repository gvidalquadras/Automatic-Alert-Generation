import torch
import os
import fasttext
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from ner_dataset import load_ner_data, collate_fn, NERDataset
from ner_doblelstm import NERModel

def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for tokens, tags, lengths in dataloader:
        tags, lengths = tags.to(device), lengths.to(device)

        logits = model(tokens, lengths)
        logits_flat = logits.view(-1, logits.size(-1))
        tags_flat = tags.view(-1)

        loss = criterion(logits_flat, tags_flat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = logits_flat.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(tags_flat.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=1, labels=np.arange(9))
    return total_loss / len(dataloader), f1

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for tokens, tags, lengths in dataloader:
            tags, lengths = tags.to(device), lengths.to(device)
            logits = model(tokens, lengths)

            logits_flat = logits.view(-1, logits.size(-1))
            tags_flat = tags.view(-1)
            loss = criterion(logits_flat, tags_flat)
            total_loss += loss.item()

            preds = logits_flat.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(tags_flat.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=1, labels=np.arange(9))
    return total_loss / len(dataloader), f1

def main():
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    embedding_model = fasttext.load_model("cc.en.300.bin")

    train_tokens, train_tags = load_ner_data("data/conll2003_train.csv")
    val_tokens, val_tags = load_ner_data("data/conll2003_validation.csv")

    train_dataset = NERDataset(train_tokens, train_tags)
    val_dataset = NERDataset(val_tokens, val_tags)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    model = NERModel(embedding_model, hidden_dim=256, num_classes=9).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(10):
        train_loss, train_f1 = train_model(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1 = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/10 - Train Loss: {train_loss:.4f}, F1: {train_f1:.4f} | Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/ner_model2.pth")
    print("Modelo guardado en 'models/ner_model2.pth'")

if __name__ == "__main__":
    main()
