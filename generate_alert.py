from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd
import ast
import re

# Cargar modelo y tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Cargar CSV
df = pd.read_csv("conll2003_train.csv", header=None,
                 names=["id", "tokens", "pos_tags", "chunk_tags", "ner_tags", "sentiment"])

# Función para corregir la columna 'tokens' y asegurar los espacios
def parse_tokens_column(x):
    if isinstance(x, str):
        try:
            # Intenta interpretar strings como listas reales de tokens
            return ast.literal_eval(x)
        except:
            # Si falla, intenta limpiar manualmente
            x = x.strip("[]")
            x = x.replace("'", "").replace('"', "")
            return x.split()
    return x

# Aplica la función a la columna 'tokens'
df["tokens"] = df["tokens"].apply(parse_tokens_column)

# Función para construir el input para el modelo
def construir_input(tokens, ner_tags, sentiment):
    # Asegurarse de que las palabras estén correctamente separadas
    texto = " ".join(tokens)  # Aseguramos que haya un espacio entre cada token
    
    # Crear el mensaje del prompt con el sentimiento
    alerta_prompt = f"Generate an alert message for this text with {'POSITIVE' if sentiment == 1 else 'NEGATIVE'} sentiment : {texto}"
    
    return alerta_prompt

# Crear la columna 'prompt' para enviar al modelo
df["prompt"] = df.apply(lambda row: construir_input(row["tokens"], row["ner_tags"], row["sentiment"]), axis=1)

# Función para generar alertas
def generar_alerta(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=50)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Prompt:", prompt)
    print("Alerta generada:", decoded)
    print("-" * 50)
    # Devolver solo la alerta generada sin el prompt
    return decoded
# Generar alertas automáticamente
df["alerta_generada"] = df["prompt"].apply(generar_alerta)

# Mostrar solo las alertas generadas
print(df[["id", "alerta_generada"]].head(10))
