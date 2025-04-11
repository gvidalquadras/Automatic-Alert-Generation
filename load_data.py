from datasets import load_dataset
from transformers import pipeline
import pandas as pd
import numpy as np
import os

# Cargar el dataset
dataset = load_dataset("conll2003")

# Función para balancear el dataset al 40% positivos y 60% negativos
def balance_sentiment_proportion(df, positive_col="sentiment", target_pos_ratio=0.4):
    positives = df[df[positive_col] == 1]
    negatives = df[df[positive_col] == 0]

    num_pos = len(positives)
    num_neg_required = int((num_pos / target_pos_ratio) * (1 - target_pos_ratio))

    if num_neg_required > len(negatives):
        print("Advertencia: No hay suficientes negativos para balancear al 40/60. Se usarán todos los disponibles.")
        negatives_sampled = negatives
    else:
        negatives_sampled = negatives.sample(n=num_neg_required, random_state=42)

    balanced_df = pd.concat([positives, negatives_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Cálculo de proporciones
    total = len(balanced_df)
    num_pos = (balanced_df[positive_col] == 1).sum()
    num_neg = (balanced_df[positive_col] == 0).sum()
    pos_pct = round(100 * num_pos / total, 2)
    neg_pct = round(100 * num_neg / total, 2)

    print(f"Dataset balanceado: {num_pos} positivos ({pos_pct}%) y {num_neg} negativos ({neg_pct}%).")
    return balanced_df

# Análisis de sentimiento, balanceo y guardado del CSV final
def add_sentiment_and_save(dataset_split, filename):
    """Añade sentimiento, balancea y guarda el CSV final con nombre dado."""
    sentiment_analyzer = pipeline("sentiment-analysis", batch_size=32) 
    df = dataset_split.to_pandas()  # Convertir a DataFrame

    # Reconstruir oraciones
    unique_ids = df["id"].unique()
    sentence_texts = df.groupby("id")["tokens"].apply(lambda x: " ".join(map(str, x))).tolist()  # Convertimos a string

    try:
        sentiments = sentiment_analyzer(sentence_texts)
        sentiment_labels = {unique_ids[i]: s["label"] for i, s in enumerate(sentiments)}
    except Exception as e:
        print(f"Error en el análisis de sentimiento: {e}")
        sentiment_labels = {uid: "UNKNOWN" for uid in unique_ids}

    df["sentiment"] = df["id"].map(sentiment_labels)
    df["sentiment"] = df["sentiment"].apply(lambda x: int(x == "POSITIVE"))

    # Balancear y guardar
    df_balanced = balance_sentiment_proportion(df)
    
    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Guardar archivo
    df_balanced.to_csv(filename, index=False)
    print(f"Guardado: {filename}")

# Procesar todo el dataset
def load_and_process_all():
    add_sentiment_and_save(dataset["train"], "data/conll2003_train.csv")
    add_sentiment_and_save(dataset["validation"], "data/conll2003_validation.csv")
    add_sentiment_and_save(dataset["test"], "data/conll2003_test.csv")
    print("Todos los archivos fueron procesados y guardados correctamente!")

# Ejecutar
load_and_process_all()