from datasets import load_dataset
import pandas as pd
import os

# Cargar el dataset
dataset = load_dataset("conll2003")

# Guardar cada split como CSV
def save_dataset_split(split, filename):
    df = split.to_pandas()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Guardado: {filename}")


def load_data():
    # Guardar los tres splits
    save_dataset_split(dataset["train"], "data/conll2003_train.csv")
    save_dataset_split(dataset["validation"], "data/conll2003_validation.csv")
    save_dataset_split(dataset["test"], "data/conll2003_test.csv")

    print("Todos los archivos fueron cargados y guardados correctamente.")

if __name__ == "__main__":
    load_data()