import os
import zipfile
import pandas as pd
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split

# ===============================
# 1. Descargar y descomprimir
# ===============================
def download_and_extract_sentiment140(save_dir="data/"):
    os.makedirs(save_dir, exist_ok=True)
    url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
    print("Descargando Sentiment140...")
    response = requests.get(url)
    with zipfile.ZipFile(BytesIO(response.content)) as z:
        z.extractall(save_dir)
    print("Descarga y extracción completa.")

# ===============================
# 2. Procesar y balancear (solo binario)
# ===============================
def load_and_prepare(path="data/training.1600000.processed.noemoticon.csv"):
    colnames = ["target", "ids", "date", "flag", "user", "text"]
    df = pd.read_csv(path, encoding="latin-1", names=colnames)

    df = df[df["target"].isin([0, 4])].copy()
    df["label"] = df["target"].map({0: 0, 4: 1})  # map to binary labels
    df = df[["text", "label"]].dropna()

    # Balancear
    min_size = df["label"].value_counts().min()
    balanced_df = df.groupby("label").apply(lambda x: x.sample(min_size, random_state=42)).reset_index(drop=True)

    print("Dataset balanceado (binario):")
    print(balanced_df["label"].value_counts())
    return balanced_df

# ===============================
# 3. Dividir y guardar
# ===============================
def split_and_save(df, save_dir="data/"):
    train_val, test = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)
    train, val = train_test_split(train_val, test_size=0.1111, stratify=train_val["label"], random_state=42)
    # 0.1111 de 90% ≈ 10% → 80/10/10

    print("Loading Sentiment140 CSVs:")
    train.to_csv(os.path.join(save_dir, "sentiment140_train.csv"), index=False)
    val.to_csv(os.path.join(save_dir, "sentiment140_validation.csv"), index=False)
    test.to_csv(os.path.join(save_dir, "sentiment140_test.csv"), index=False)

    print(f"   Train: {len(train)} ejemplos")
    print(f"   Val:   {len(val)} ejemplos")
    print(f"   Test:  {len(test)} ejemplos")

def main():
    download_and_extract_sentiment140()
    df = load_and_prepare()
    split_and_save(df)
if __name__ == "__main__":
    main()
