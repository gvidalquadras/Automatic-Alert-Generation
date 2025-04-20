from sklearn.metrics import f1_score
import numpy as np
import torch
from torch.utils.data import DataLoader
import fasttext

from src.ner.ner import NERDataset, NERModel
from src.ner.train_NER import evaluate_model,collate_fn, load_ner_data
from src.utils import set_seed

def main():
    """"
    This is the main function for evaluating test data.
    """
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_dim = 256
    num_classes = 9 
    embedding_model = fasttext.load_model("cc.en.300.bin")
    # Cargar el modelo entrenado (suponiendo que lo hayas guardado previamente)
    model = NERModel(embedding_model, hidden_dim, num_classes).to(device)  # Inicializa el modelo antes de cargar los pesos
    model.load_state_dict(torch.load("models/ner_model.pth"))

    # Cargar el conjunto de datos de prueba (suponiendo que ya lo hayas cargado)
    test_tokens, test_ner_tags = load_ner_data("data/conll2003_test.csv")
    test_dataset = NERDataset(test_tokens, test_ner_tags)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Definir el criterio de pérdida
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Evaluar el modelo en el conjunto de test
    test_loss, f1 = evaluate_model(model, test_loader, criterion, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Test Loss: {test_loss}, F1: {f1}")
    

if __name__ == "__main__":
    main()