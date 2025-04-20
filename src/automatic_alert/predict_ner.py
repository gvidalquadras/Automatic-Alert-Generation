import torch
import fasttext
import numpy as np
from ner import NERModel  # Asegúrate de que esta ruta sea correcta
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict, Any


idx2label = {
    0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"
}


def predict_ner(model: torch.nn.Module,
                tokens: List[str],
                device: torch.device,
                embedding_model: Any,
                idx2label: Dict[int, str]) -> List[Tuple[str, str]]:
    """
    Performs NER prediction on a single tokenized sentence.

    Args:
        model (torch.nn.Module): Trained NER model.
        tokens (List[str]): List of tokens (a sentence).
        device (torch.device): Device to run the model on ('cpu' or 'cuda').
        embedding_model (Any): Embedding model (e.g., FastText) with `get_word_vector()`.
        idx2label (Dict[int, str]): Mapping from label indices to NER tag strings.

    Returns:
        List[Tuple[str, str]]: List of (token, predicted_tag) pairs.
    """
    model.eval()
    with torch.no_grad():
        vectors = [embedding_model.get_word_vector(tok) for tok in tokens]
        input_tensor = torch.tensor(np.array(vectors), dtype=torch.float32).unsqueeze(0).to(device)
        lengths = torch.tensor([len(tokens)], dtype=torch.long).to(device)
        logits = model([tokens], lengths)
        preds = logits.argmax(dim=-1).squeeze(0).tolist()
        return list(zip(tokens, [idx2label[i] for i in preds]))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading modelo FastText...")
    ft_model = fasttext.load_model("cc.en.300.bin")

    print("Loading NER model...")
    model = NERModel(embedding_model=ft_model, hidden_dim=256, num_classes=9).to(device)
    model.load_state_dict(torch.load("models/ner_model.pth", map_location=device))

    sentence = "Barack Obama visited Paris and met with Microsoft officials.".strip()
    tokens = sentence.split()
    predictions = predict_ner(model, tokens, device, ft_model, idx2label)
    print("Etiquetas NER:")
    for word, tag in predictions:
        print(f"{word:15} → {tag}")
    print()