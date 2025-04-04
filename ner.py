from gensim.models import FastText
import torch
import torch.nn as nn
import numpy as np
from torchcrf import CRF  # Librería para CRF en PyTorch

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, pre_trained_embeddings=None, embedding_dim=100, hidden_dim=256):
        super().__init__()
        # Si tienes embeddings preentrenados, los cargamos
        if pre_trained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(pre_trained_embeddings, dtype=torch.float32))
        else:
            # Si no hay embeddings preentrenados, inicializamos de manera aleatoria
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, len(tag_to_ix))
        self.crf = CRF(len(tag_to_ix))

    def forward(self, x):
        embeds = self.embedding(x)  # (seq_len, batch, emb_dim)
        lstm_out, _ = self.lstm(embeds)  # (seq_len, batch, hidden_dim)
        emissions = self.hidden2tag(lstm_out)  # (seq_len, batch, num_tags)
        return emissions

def load_fasttext_embeddings_from_gensim(model_path, vocab, embedding_dim=100):
    """Carga embeddings de FastText desde un modelo de Gensim y los mapea a un vocabulario"""
    fasttext_model = FastText.load(model_path)  # Cargar el modelo FastText preentrenado de Gensim

    word_vectors = fasttext_model.wv  # Obtener los vectores de palabras del modelo

    embeddings = np.zeros((len(vocab), embedding_dim))  # Tamaño de vocabulario x dimensión de los vectores
    for word, idx in vocab.items():
        if word in word_vectors:
            embeddings[idx] = word_vectors[word]  # Asignar el vector de la palabra
        else:
            # Inicialización aleatoria si no se encuentra la palabra en los embeddings
            embeddings[idx] = np.random.uniform(-0.25, 0.25, embedding_dim)
    return embeddings

# Ejemplo de vocabulario e índices
vocab = {'the': 0, 'cat': 1, 'sat': 2, 'on': 3, 'mat': 4}  # Vocabulario de ejemplo
tag_to_ix = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}# Etiquetas de ejemplo

# Cargar embeddings de FastText desde el modelo de Gensim
# Asegúrate de que el archivo 'fasttext_model' esté en el mismo directorio o especifica la ruta correcta
fasttext_embeddings = load_fasttext_embeddings_from_gensim('fasttext_model.model', vocab, embedding_dim=100)

# Crear el modelo BiLSTM-CRF con embeddings preentrenados
model = BiLSTM_CRF(vocab_size=len(vocab), tag_to_ix=tag_to_ix, pre_trained_embeddings=fasttext_embeddings)

# Ejemplo de uso
input_tensor = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)  # Ejemplo de entrada (índices de palabras)
output = model(input_tensor)  # Salida del modelo
print(output)
