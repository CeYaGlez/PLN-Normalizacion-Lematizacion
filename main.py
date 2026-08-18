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

# 2. Leer el libro
with open("libro.txt", "r", encoding="utf-8-sig") as f:
    texto = f.read()

doc = nlp(texto)
print(f"Texto cargado. Longitud: {len(texto)} caracteres, {len(doc)} tokens.")

# 3. Filtrar stop words y puntuación
tokens_relevantes = [
    token.text for token in doc
    if not token.is_stop and not token.is_punct and token.text.strip()
]

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

# 5. Comparativa Stemming vs Lematización
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
X_bow = bow_vectorizer.fit_transform(corpus_lematizado)
print(f"\nMatriz BoW: {X_bow.shape[0]} oraciones × {X_bow.shape[1]} palabras")

# 8. TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=100)
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


# ─────────────────────────────────────────────
# MÓDULO 5 — Semántica Distribucional (Word2Vec)
# ─────────────────────────────────────────────
from gensim.models import Word2Vec
import multiprocessing

print("\n--- 5. Entrenando Word2Vec (Semántica Distribucional) ---")

# 1. Preparar los datos: Gensim necesita una lista de listas de palabras (oraciones tokenizadas)
sentences = [oracion.split() for oracion in corpus_lematizado if len(oracion.split()) > 1]
print(f"Total de oraciones para entrenar: {len(sentences)}")

# 2. Entrenar el modelo Word2Vec
# Usamos Skip-gram (sg=1) porque, según Eisenstein, captura mejor las relaciones semánticas finas
model_w2v = Word2Vec(
    sentences,
    vector_size=50,       # 50 dimensiones (mayor que 10 para capturar más contexto del libro completo)
    window=5,             # Tamaño de la ventana de contexto
    min_count=2,          # Ignoramos palabras que aparecen solo 1 vez (ruido)
    workers=multiprocessing.cpu_count(),
    sg=1,                 # 1 para Skip-gram, 0 para CBOW
    seed=42
)

# 3. Exploración semántica (Similitud)
# Buscamos palabras cercanas a conceptos clave del libro de Goggins
palabras_clave = ["dolor", "mente", "correr", "alma"]
for palabra in palabras_clave:
    if palabra in model_w2v.wv:
        print(f"\nPalabras cercanas semánticamente a '{palabra}':")
        similares = model_w2v.wv.most_similar(palabra, topn=3)
        for sim_palabra, score in similares:
            print(f"  - {sim_palabra} (Similitud: {score:.4f})")
    else:
        print(f"\nLa palabra '{palabra}' no está en el vocabulario (min_count muy alto).")

# 4. Visualización 3D del Espacio Semántico (Embeddings)
# Obtenemos el vocabulario y sus vectores
vocabulario_w2v = list(model_w2v.wv.index_to_key)
vectores_w2v = model_w2v.wv[vocabulario_w2v]

# Reducimos las 50 dimensiones a 3 usando PCA
pca_w2v = PCA(n_components=3)
coords_w2v = pca_w2v.fit_transform(vectores_w2v)

# Crear DataFrame para graficar
df_w2v = pd.DataFrame(coords_w2v, columns=['x', 'y', 'z'])
df_w2v['palabra'] = vocabulario_w2v

# Gráfico 3D de Word2Vec
fig_w2v = plt.figure(figsize=(12, 8))
ax_w2v = fig_w2v.add_subplot(111, projection='3d')

# Scatter plot
ax_w2v.scatter(df_w2v['x'], df_w2v['y'], df_w2v['z'], c='purple', s=80, edgecolors='white', alpha=0.8, depthshade=True)

# Etiquetas (Limitamos a las primeras 80 palabras para no saturar la imagen)
for i, row in df_w2v.head(80).iterrows():
    ax_w2v.text(row['x'], row['y'], row['z'] + 0.1, row['palabra'], fontsize=8, color='black')

ax_w2v.set_title('Espacio Semántico (Word2Vec - Embeddings) - Goggins', fontsize=14, fontweight='bold')
ax_w2v.set_xlabel('Dimensión Latente 1')
ax_w2v.set_ylabel('Dimensión Latente 2')
ax_w2v.set_zlabel('Dimensión Latente 3')

plt.tight_layout()

# GUARDAR LA SEGUNDA IMAGEN PARA EL REPOSITORIO
plt.savefig("embeddings_3d_goggins.png", dpi=150, bbox_inches='tight')
print("\nImagen guardada como: embeddings_3d_goggins.png")
plt.show()