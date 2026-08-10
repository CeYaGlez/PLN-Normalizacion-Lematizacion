import os
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
import spacy
from nltk.stem import SnowballStemmer
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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

# ─────────────────────────────────────────────
# MÓDULO 4 — Representación Vectorial
# ─────────────────────────────────────────────

# 6. Construir corpus lematizado por oración
corpus_lematizado = []

for oracion in doc.sents:
    lemas_oracion = [
        token.lemma_.lower()
        for token in oracion
        if not token.is_punct and not token.is_space and not token.is_stop
    ]
    if lemas_oracion:
        corpus_lematizado.append(" ".join(lemas_oracion))

print(f"\nTotal de oraciones procesadas: {len(corpus_lematizado)}")
print("Ejemplo de oración lematizada:", corpus_lematizado[0])

# 7. Bag of Words
bow_vectorizer = CountVectorizer(max_features=100)
X_bow = bow_vectorizer.fit_transform(max_features=100)
print(f"\nMatriz BoW: {X_bow.shape[0]} oraciones × {X_bow.shape[1]} palabras")

# 8. TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus_lematizado)
print(f"Matriz TF-IDF: {X_tfidf.shape[0]} oraciones × {X_tfidf.shape[1]} palabras")

# 9. Visualización 3D
def graficar_palabras_3d(ax, matriz, vocabulario, titulo, color_puntos):
    matriz_palabras = matriz.T
    pca = PCA(n_components=3)
    coords = pca.fit_transform(matriz_palabras.toarray())

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    ax.scatter(x, y, z, c=color_puntos, s=80, edgecolors='k', alpha=0.8, depthshade=True)

    for i, palabra in enumerate(vocabulario):
        ax.text(x[i], y[i], z[i] + 0.1, palabra, fontsize=9)

    ax.set_title(titulo)
    ax.set_xlabel('Comp. Principal 1')
    ax.set_ylabel('Comp. Principal 2')
    ax.set_zlabel('Comp. Principal 3')

    ax.plot([0,0], [0,0], [z.min(), z.max()], c='grey', ls='--', lw=0.5, alpha=0.3)
    ax.plot([x.min(), x.max()], [0,0], [0,0], c='grey', ls='--', lw=0.5, alpha=0.3)
    ax.plot([0,0], [y.min(), y.max()], [0,0], c='grey', ls='--', lw=0.5, alpha=0.3)


fig = plt.figure(figsize=(18, 8))

ax1 = fig.add_subplot(121, projection='3d')
vocab_bow = bow_vectorizer.get_feature_names_out()
graficar_palabras_3d(ax1, X_bow, vocab_bow, "Espacio BoW 3D (Conteos)", "orange")

ax2 = fig.add_subplot(122, projection='3d')
vocab_tfidf = tfidf_vectorizer.get_feature_names_out()
graficar_palabras_3d(ax2, X_tfidf, vocab_tfidf, "Espacio TF-IDF 3D (Importancia)", "teal")

plt.tight_layout()
plt.show()