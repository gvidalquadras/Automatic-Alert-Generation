import torch
import fasttext
import numpy as np
from src.ner.ner import NERModel  
from torch.nn.utils.rnn import pad_sequence
from typing import List

# Diccionario inverso: del índice a la etiqueta NER
idx2label = {
    0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"
}

# Función para predecir las etiquetas NER de una frase
def predict_ner(model, tokens: List[str], device, embedding_model, idx2label):
    model.eval()
    with torch.no_grad():
        vectors = [embedding_model.get_word_vector(tok) for tok in tokens]
        input_tensor = torch.tensor(np.array(vectors), dtype=torch.float32).unsqueeze(0).to(device)
        lengths = torch.tensor([len(tokens)], dtype=torch.long).to(device)
        logits = model([tokens], lengths)
        preds = logits.argmax(dim=-1).squeeze(0).tolist()
        return list(zip(tokens, [idx2label[i] for i in preds]))

if __name__ == "__main__":
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    print("📦 Cargando modelo FastText...")
    ft_model = fasttext.load_model("cc.en.300.bin")

    print("🧠 Cargando modelo NER...")
    model = NERModel(embedding_model=ft_model, hidden_dim=256, num_classes=9).to(device)
    model.load_state_dict(torch.load("models/ner_model.pth", map_location=device))

    print("\n Modelo cargado. Escribe frases para analizar. Escribe 'exit' para salir.\n")


    sentence = "Barack Obama visited Paris and met with Microsoft officials.".strip()
    tokens = sentence.split()
    predictions = predict_ner(model, tokens, device, ft_model, idx2label)
    print("🔍 Etiquetas NER:")
    for word, tag in predictions:
        print(f"{word:15} → {tag}")
    print()
