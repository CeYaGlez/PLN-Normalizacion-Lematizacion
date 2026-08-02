import os
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
import spacy
from nltk.stem import SnowballStemmer
import pandas as pd

# 1. Cargar modelo
nlp = spacy.load("es_core_news_sm")

# 2. Leer tu libro
with open("libro.txt", "r", encoding="utf-8-sig") as f:
    texto = f.read()

doc = nlp(texto)
print(f"Texto cargado. Longitud: {len(texto)} caracteres, {len(doc)} tokens.")

# 3. Filtrar stop words y puntuación
tokens_relevantes = [token.text for token in doc if not token.is_stop and not token.is_punct and token.text.strip()]

# 4. Lematizar
tokens_normalizados = []
cambios_interesantes = []
for token in doc:
    if not token.is_stop and not token.is_punct and token.text.strip():
        lema = token.lemma_.lower()
        tokens_normalizados.append(lema)
        if token.text.lower() != lema:
            cambios_interesantes.append(f"{token.text} ➡ {lema}")

print("Ejemplos de lematización:", cambios_interesantes[:10])
print("Primeros tokens normalizados:", tokens_normalizados[:10])

# 5. (Opcional) Comparativa Stemming vs Lematización
stemmer = SnowballStemmer("spanish")
data_comparativa = []
for token in doc:
    if not token.is_punct and not token.is_space:
        raiz_stem = stemmer.stem(token.text)
        lema = token.lemma_
        data_comparativa.append({
            "Original": token.text,
            "Stemming": raiz_stem,
            "Lematización": lema,
            "¿Coinciden?": raiz_stem == lema
        })

df = pd.DataFrame(data_comparativa)
print(df.head(15).to_string(index=False))