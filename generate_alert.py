import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ✅ Paso 1: Cargar tokenizer y modelo DeepSeek
model_name = "deepseek-ai/deepseek-llm-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # usa GPU si hay
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# ✅ Paso 2: Leer CSV
df = pd.read_csv("conll2003_train.csv")

# ✅ Paso 3: Limpiar tokens correctamente
def limpiar_tokens(tokens):
    # tokens es una lista de strings tipo "'EU'" o "[ 'EU', 'rejects', ... ]"
    if isinstance(tokens, str):
        tokens = tokens.replace("[", "").replace("]", "").replace("'", "").split(",")
        tokens = [t.strip() for t in tokens if t.strip()]
    return " ".join(tokens)

# Agrupar por oración (por ID)
sentences = df.groupby("id")["tokens"].apply(lambda x: limpiar_tokens(" ".join(x))).tolist()

# ✅ Paso 4: Crear prompts tipo instruct
prompts = [f"<|user|>\nGenerate an alert for this text:\n{sentence}\n<|assistant|>" for sentence in sentences]

# ✅ Paso 5: Generar alertas
def generar_alertas_batch(prompts, batch_size=4):
    alertas = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=80)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for prompt, alerta in zip(batch, decoded):
            print("📝 Prompt:", prompt.split("\n")[1])
            print("⚠️  Alerta generada:", alerta.replace(prompt.split("\n")[0], "").strip())
            print("-" * 50)

        alertas.extend(decoded)
    return alertas

# Ejecutar
alertas = generar_alertas_batch(prompts)

# ✅ Guardar resultado
result_df = pd.DataFrame({
    "sentence": sentences,
    "alerta": alertas
})
result_df.to_csv("alertas_deepseek.csv", index=False)
