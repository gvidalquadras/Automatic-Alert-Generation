import torch
import os
import fasttext
from typing import List, Tuple  
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from ner_dataset import load_ner_data, collate_fn, NERDataset
import time
from sklearn.metrics import f1_score
import numpy as np

class MyModel(torch.nn.Module):
    def __init__(self, embedding_model, hidden_dim, num_classes):
        super(MyModel, self).__init__()
        self.embedding_model = embedding_model  # FastText por ejemplo
        self.hidden_dim = hidden_dim

        self.lstm = torch.nn.LSTM(input_size=300, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.linear = torch.nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, tokens: List[List[str]], lengths: torch.Tensor):
        # Convertimos listas de palabras en embeddings usando FastText
        embedded = []
        for sent in tokens:
            vectors = [self.embedding_model.get_word_vector(word) for word in sent]
            embedded.append(torch.tensor(vectors, dtype=torch.float32, device=lengths.device))

        padded = pad_sequence(embedded, batch_first=True, padding_value=0.0)

        packed = torch.nn.utils.rnn.pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(packed)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        logits = self.linear(outputs)
        return logits
    

def train_model(model: nn.Module,
                dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: str):
    model.train()
    running_loss = 0.0
    all_pred_labels = []
    all_tags = []

    for tokens, tags, lengths in dataloader:
        
        # En este ejemplo tokens es una lista de listas de strings y tags un tensor con shape [batch, max_seq_len]
        tags = tags.to(device)
        lengths = lengths.to(device)
        
        optimizer.zero_grad()

        # Forward: obtenemos logits de la forma [batch, max_seq_len, num_classes]
        logits = model(tokens, lengths)
        
        # Para CrossEntropyLoss, necesitamos reacomodar los tensores a 2D.
        # Aplanamos los logits a shape [batch*max_seq_len, num_classes]
        logits_flat = logits.view(-1, logits.size(-1))
        pred_labels = logits_flat.argmax(dim=-1)  # [batch*max_seq_len]
        # Aplanamos las etiquetas a shape [batch*max_seq_len]
        tags_flat = tags.view(-1)
      
        all_pred_labels.extend(pred_labels.cpu().numpy())
        all_tags.extend(tags_flat.cpu().numpy())
        loss = criterion(logits_flat, tags_flat)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        f1 = f1_score(all_tags, all_pred_labels, average='macro',zero_division=1, labels=np.arange(9))

    return running_loss / len(dataloader), f1

# Función de validación (similar al entrenamiento, pero sin gradientes)
def evaluate_model(model: nn.Module,
                   dataloader: DataLoader,
                   criterion: nn.Module,
                   device: str):
    model.eval()
    running_loss = 0.0
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
            running_loss += loss.item()
    f1 = f1_score(all_tags, all_pred_labels, average='macro', labels=np.arange(9))

    return running_loss / len(dataloader), f1

# Función principal para entrenar y evaluar
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #Cargar modelo FastText (asegúrate de tener el archivo correcto)
    embedding_model = fasttext.load_model("cc.en.300.bin")
    
    # Cargar datos de entrenamiento y validación
    train_tokens, train_ner_tags = load_ner_data("data/conll2003_train.csv")
    validation_tokens, validation_ner_tags = load_ner_data("data/conll2003_validation.csv")  

    train_dataset = NERDataset(train_tokens, train_ner_tags)
    validation_dataset = NERDataset(validation_tokens, validation_ner_tags)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(validation_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    hidden_dim = 256
    num_classes = 9  # según tu mapeo de etiquetas
    model = MyModel(embedding_model, hidden_dim, num_classes).to(device)
    
    # Definimos el optimizador y la función de pérdida
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # ignoramos el padding
    
    epochs = 10
    for epoch in range(epochs):
        train_loss, f1_train = train_model(model, train_loader, optimizer, criterion, device)
        val_loss, f1_val = evaluate_model(model, val_loader, criterion, device)  # Por ejemplo, usando el mismo train_loader para validación
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train F1: {f1_train:.4f}, Val F1: {f1_val:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/ner_model.pth")  # Guardamos el modelo entrenado
    
if __name__ == "__main__":
    main()
