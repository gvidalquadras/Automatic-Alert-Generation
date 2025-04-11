from sklearn.metrics import f1_score
import numpy as np
import torch
from torch.utils.data import DataLoader
from ner_dataset import NERDataset, load_ner_data
from typing import List, Tuple
from ner_doblelstm import evaluate_model, collate_fn, MyModel
import fasttext
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_dim = 256
    num_classes = 9 
    embedding_model = fasttext.load_model("cc.en.300.bin")
    # Cargar el modelo entrenado (suponiendo que lo hayas guardado previamente)
    model = MyModel(embedding_model, hidden_dim, num_classes).to(device)  # Inicializa el modelo antes de cargar los pesos
    model.load_state_dict(torch.load("models/ner_model.pth"))

    # Cargar el conjunto de datos de prueba (suponiendo que ya lo hayas cargado)
    test_tokens, test_ner_tags = load_ner_data("conll2003_test.csv")
    test_dataset = NERDataset(test_tokens, test_ner_tags)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Definir el criterio de pérdida
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Evaluar el modelo en el conjunto de test
    test_loss = evaluate_model(model, test_loader, criterion, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Test Loss: {test_loss}")
    

if __name__ == "__main__":
    main()