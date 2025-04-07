from datasets import load_dataset
from transformers import pipeline
import pandas as pd
import numpy as np

# Cargar el dataset
dataset = load_dataset("conll2003")

# Cargar modelo de análisis de sentimiento
 # Procesa en lotes

def add_sentiment_and_save(dataset_split, filename):
    """Añade una columna de sentimiento al dataset y lo guarda en CSV."""
    sentiment_analyzer = pipeline("sentiment-analysis", batch_size=32) 
    df = dataset_split.to_pandas()  # Convertir a DataFrame

    # Reconstruir oraciones agrupando por 'id'
    unique_ids = df["id"].unique()
    sentence_texts = df.groupby("id")["tokens"].apply(lambda x: " ".join(map(str, x))).tolist()  # Convertimos a string


    # Obtener sentimientos de las oraciones en lotes
    try:
        sentiments = sentiment_analyzer(sentence_texts)
        sentiment_labels = {unique_ids[i]: s["label"] for i, s in enumerate(sentiments)}
    except Exception as e:
        print(f"Error en el análisis de sentimiento: {e}")
        sentiment_labels = {uid: "UNKNOWN" for uid in unique_ids}  # Valor por defecto en caso de error

    # Agregar sentimiento al DataFrame sin perder tokens
    df["sentiment"] = df["id"].map(sentiment_labels)
    df["sentiment"] = df["sentiment"].apply(lambda x: int(x=="POSITIVE" ))

    # Guardar CSV manteniendo la estructura original
    df.to_csv(filename, index=False)
    print(f"Guardado: {filename}")



# Procesar y guardar cada split del dataset
add_sentiment_and_save(dataset["train"], "conll2003_train.csv")
add_sentiment_and_save(dataset["validation"], "conll2003_validation.csv")
add_sentiment_and_save(dataset["test"], "conll2003_test.csv")

print("¡Todos los archivos fueron guardados con la columna de sentimiento! 🚀")