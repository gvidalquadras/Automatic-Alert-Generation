import torch.optim as optim
from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader
import torch.nn as nn 
import fasttext 
from ner_dataset import load_ner_data, NERDataset, collate_fn
from ner import BiRNN_CRF_FastText
from load_data import load_data
import os 

def train_torch_model(model: torch.nn.Module, train_dataloader: DataLoader,
                      val_dataloader: DataLoader, optimizer: optim.Optimizer, epochs: int,
                      print_every: int, patience: int,
                      device: str = 'cpu') -> Tuple[Dict[int, float], Dict[int, float]]:
    train_accuracies: Dict[int, float] = {}
    val_accuracies: Dict[int, float] = {}
    best_loss: float = float('inf')
    epochs_no_improve: int = 0

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for embeds, ner_tags_padded, lengths in train_dataloader:
            embeds, ner_tags_padded = embeds.to(device), ner_tags_padded.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            emissions, lengths  = model(embeds)
            loss = model.loss(emissions, ner_tags_padded, lengths)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for embeds, ner_tags_padded, lengths in val_dataloader:
                embeds, ner_tags_padded = embeds.to(device), ner_tags_padded.to(device)
                lengths = lengths.to(device)

                emissions, lengths  = model(embeds)
                loss = model.loss(emissions, ner_tags_padded, lengths)
                val_loss += loss.item()

        if epoch % print_every == 0 or epoch == epochs - 1:
            train_accuracy = calculate_accuracy(model, train_dataloader, device)
            val_accuracy = calculate_accuracy(model, val_dataloader, device)

            train_accuracies[epoch] = train_accuracy
            val_accuracies[epoch] = val_accuracy

            print(f"Epoch {epoch}/{epochs}, Train Loss: {total_loss / len(train_dataloader):.4f}, "
                  f"Validation Loss: {val_loss / len(val_dataloader):.4f}, "
                  f"Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch: {epoch}")
            break

    return train_accuracies, val_accuracies

def calculate_accuracy(model: torch.nn.Module, dataloader: DataLoader, device: str) -> float:
    correct = 0
    total = 0
    model.to(device)
    model.eval()

    with torch.no_grad():
        for embeds, ner_tags_padded, lengths in dataloader:
            embeds, ner_tags_padded = embeds.to(device), ner_tags_padded.to(device)
            lengths = lengths.to(device)
            emissions, lengths = model(embeds)


            predictions = model.predict(emissions, lengths)
            print(predictions)
            print(ner_tags_padded)

            for i in range(len(ner_tags_padded)):
                for j in range(len(ner_tags_padded[i])):
                    if ner_tags_padded[i][j] == predictions[i][j]:
                        correct += 1
                    total += 1

    accuracy = correct / total
    return accuracy

def main():
    train_file_path = 'conll2003_train.csv'
    validation_file_path = 'conll2003_validation.csv'
    fasttext_model = fasttext.load_model("cc.en.300.bin")
    
    '''    if not os.path.isfile(train_file_path):    
        print('LOADING DATA')
        load_data()
        '''

    train_tokens, train_ner_tags = load_ner_data(train_file_path)
    validation_tokens, validation_ner_tags = load_ner_data(validation_file_path)

    train_dataset = NERDataset(train_tokens, train_ner_tags)
    validation_dataset = NERDataset(validation_tokens, validation_ner_tags)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: collate_fn(x, fasttext_model))
    validation_loader = DataLoader(validation_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: collate_fn(x, fasttext_model))

    tagset_size = 9
    embedding_dim = 300
    hidden_dim = 256

    model = BiRNN_CRF_FastText(fasttext_model=fasttext_model, tagset_size=tagset_size, hidden_dim=hidden_dim)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    print_every = 1
    patience = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_accuracies, val_accuracies = train_torch_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=validation_loader,
        optimizer=optimizer,
        epochs=epochs,
        print_every=print_every,
        patience=patience,
        device=device
    )

if __name__ == "__main__": 
    main()
