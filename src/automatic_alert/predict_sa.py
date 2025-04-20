import torch
import fasttext
from src.sa.SA import SentimentClassifier
import numpy as np

# Cargar FastText
ft_model = fasttext.load_model("cc.en.300.bin")

# Elegir dispositivo (igual que antes)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Cargar el modelo completo
model = torch.load("models/SA_model.pt", map_location=device, weights_only=False)

model.eval()
text = "I love this movie, it's amazing!"
label, prob = model.predict(text, ft_model, device)

print(f"Predicción: {'Positivo' if label == 1 else 'Negativo'}")
print(f"Probabilidad: {prob:.4f}")

def predict_sa(text: str, model: SentimentClassifier, embedding_model: fasttext.FastText._FastText, device: torch.device) -> str:
    """
    Predict the sentiment of a given text using the sentiment analysis model.

    Args:
        text (str): The input text to analyze.
        model (SentimentClassifier): The sentiment analysis model.
        embedding_model (fasttext.FastText._FastText): The FastText embedding model.
        device (torch.device): The device to run the model on.

    Returns:
        str: The predicted sentiment label (1 for positive, 0 for negative).
        float: The probability of the predicted sentiment.
    """
    label, prob = model.predict(text, ft_model, device)
    return label, prob
