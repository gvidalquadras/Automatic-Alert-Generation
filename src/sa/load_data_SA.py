import os
import zipfile
import pandas as pd
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split


def download_and_extract_sentiment140(save_dir: str = "data/") -> None:
    """
    Downloads and extracts the Sentiment140 dataset ZIP file into a given directory.

    Args:
        save_dir (str): Directory where the data should be saved. Defaults to "data/".
    """
    os.makedirs(save_dir, exist_ok=True)
    url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
    print("Downloading Sentiment140...")
    response = requests.get(url)
    with zipfile.ZipFile(BytesIO(response.content)) as z:
        z.extractall(save_dir)

def load_and_prepare(path:str="data/training.1600000.processed.noemoticon.csv")-> pd.DataFrame:
    """
    Loads the original Sentiment140 CSV file, filters for binary sentiment,
    maps the labels to {0,1}, and balances the dataset.

    Args:
        path (str): Path to the downloaded CSV file.

    Returns:
        pd.DataFrame: Balanced DataFrame with columns ["text", "label"].
    """
    colnames = ["target", "ids", "date", "flag", "user", "text"]
    df = pd.read_csv(path, encoding="latin-1", names=colnames)

    df = df[df["target"].isin([0, 4])].copy()
    df["label"] = df["target"].map({0: 0, 4: 1})  # map to binary labels
    df = df[["text", "label"]].dropna()

    # Balancear
    min_size = df["label"].value_counts().min()
    balanced_df = df.groupby("label").apply(lambda x: x.sample(min_size, random_state=42)).reset_index(drop=True)

    print("Dataset balanced:")
    print(balanced_df["label"].value_counts())
    return balanced_df


def split_and_save(df: pd.DataFrame, save_dir: str="data/")-> None:
    """
    Splits the dataset into train, validation, and test sets (80/10/10) and saves them as CSVs.

    Args:
        df (pd.DataFrame): The balanced dataset to split.
        save_dir (str): Directory where the CSV files will be saved.
    """
    # 80/10/10 split
    train_val, test = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)
    train, val = train_test_split(train_val, test_size=0.1111, stratify=train_val["label"], random_state=42)
    

    print("Loading Sentiment140 CSVs:")
    train.to_csv(os.path.join(save_dir, "sentiment140_train.csv"), index=False)
    val.to_csv(os.path.join(save_dir, "sentiment140_validation.csv"), index=False)
    test.to_csv(os.path.join(save_dir, "sentiment140_test.csv"), index=False)

    print(f"   Train: {len(train)} examples")
    print(f"   Val:   {len(val)} examples")
    print(f"   Test:  {len(test)} examples")

def load_data():
    download_and_extract_sentiment140()
    df = load_and_prepare()
    split_and_save(df)
    
if __name__ == "__main__":
    load_data()
