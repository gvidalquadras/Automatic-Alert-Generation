from datasets import load_dataset
import pandas as pd
import os


def save_dataset_split(split, filename):
    """
    Saves a single split of the dataset (train/validation/test) to a CSV file.

    Args:
        split (Dataset): HuggingFace dataset split to convert and save.
        filename (str): Destination file path to save the CSV.
    """
    df = split.to_pandas()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")


def load_data():
    """
    Loads the CoNLL-2003 dataset using HuggingFace Datasets
    and saves the train, validation, and test splits as CSV files.

    Generated files:
        - data/conll2003_train.csv
        - data/conll2003_validation.csv
        - data/conll2003_test.csv
    """
    # Saves the train, validation, and test splits of the CoNLL-2003 dataset to CSV files.
    dataset = load_dataset("conll2003")
    save_dataset_split(dataset["train"], "data/conll2003_train.csv")
    save_dataset_split(dataset["validation"], "data/conll2003_validation.csv")
    save_dataset_split(dataset["test"], "data/conll2003_test.csv")

    print("All Conll2003 datasets saved successfully.")

if __name__ == "__main__":
    load_data()