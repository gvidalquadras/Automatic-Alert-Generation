import torch
from torch.utils.data import Dataset, DataLoader
from gensim.models.fasttext import FastText
import fasttext.util
import pandas as pd 
import numpy as np 

# Cargar el modelo preentrenado de FastText
# Suponiendo que ya tienes un modelo preentrenado de FastText llamado "fasttext_model.bin"
# Puedes cargarlo con gensim (si tienes el modelo en formato binario de gensim)
fasttext.util.download_model('en', if_exists='ignore') 
fasttext_model = fasttext.load_model('cc.en.300.bin') # Cargar embeddings preentrenados (si lo tienes en ese formato)

# Crear un dataset personalizado para PyTorch
class NERDataset(Dataset):
    def __init__(self, sentences, labels, fasttext_model):
        self.sentences = sentences  # Lista de oraciones (como listas de palabras)
        self.labels = labels  # Lista de etiquetas (numéricas)
        self.word_to_ix = word_to_ix
        self.fasttext_model = fasttext_model  # Modelo de FastText preentrenado

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        
        # Convertir las palabras a embeddings usando el modelo FastText preentrenado
        sentence_embeddings = [self.get_embedding(word) for word in sentence]
        
        # Convertir las listas de embeddings y etiquetas numéricas a tensores de PyTorch
        return torch.tensor(sentence_embeddings), torch.tensor(label)

    def get_embedding(self, word):
        """Obtiene el embedding de una palabra usando FastText"""
        try:
            # Obtener el embedding de la palabra usando el modelo FastText preentrenado
            return self.fasttext_model.wv[word]
        except KeyError:
            # Si la palabra no está en el vocabulario, usar un vector de ceros
            return np.zeros(self.fasttext_model.vector_size)

# 1️⃣ Agrupar tokens y etiquetas por 'id'
df = pd.read_csv("conll2003_train.csv")
sentences = df.groupby("id")["tokens"].apply(list).tolist()  # Lista de oraciones
labels = df.groupby("id")["ner_tags"].apply(list).tolist()  # Lista de etiquetas numéricas (NER)

# 2️⃣ Crear el dataset y el DataLoader
dataset = NERDataset(sentences, labels, fasttext_model)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print("✅ Datos preparados con embeddings preentrenados y DataLoader creado.")
