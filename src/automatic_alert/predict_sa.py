import torch
import fasttext
from src.sa.SA import SentimentClassifier
import numpy as np

# Elegir dispositivo
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


# Función de predicción
def predict_sa(text: str, model: SentimentClassifier, embedding_model: fasttext.FastText._FastText, device: torch.device):
    """
    Predict the sentiment of a given text using the sentiment analysis model.

    Args:
        text (str): The input text to analyze.
        model (SentimentClassifier): The sentiment analysis model.
        embedding_model (fasttext.FastText._FastText): The FastText embedding model.
        device (torch.device): The device to run the model on.

    Returns:
        int: The predicted sentiment label (1 for positive, 0 for negative).
        float: The probability of the predicted sentiment.
    """
    label, prob = model.predict(text, embedding_model, device)
    return label, prob