# Procesamiento de Lenguaje Natural — PLN

Proyecto de PLN que procesa un libro en texto plano usando spaCy y NLTK.
Cubre desde la normalización básica hasta representación vectorial y visualización semántica en 3D.

## Requisitos

- Python 3.13
- Ver `requirements.txt`

## Instalación

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm

## Uso

python main.py

## ¿Qué hace?

1. Carga el texto de `libro.txt`
2. Tokeniza con spaCy
3. Filtra stop words y puntuación
4. Lematiza cada token
5. Compara resultados de lematización vs stemming (NLTK)
6. Construye un corpus lematizado por oración
7. Genera representaciones vectoriales con Bag of Words y TF-IDF
8. Reduce dimensionalidad con PCA y visualiza el espacio vectorial en 3D

## Estructura del proyecto

.
├── main.py        # Script principal
├── libro.txt      # Texto a procesar
└── requirements.txt

## Dependencias principales

| Librería | Uso |
|---|---|
| spaCy | Tokenización, lematización, stop words |
| NLTK | Stemming comparativo |
| scikit-learn | Vectorización (BoW, TF-IDF) y reducción de dimensionalidad (PCA) |
| matplotlib | Visualización 3D del espacio vectorial |
| pandas | Tabla comparativa stemming vs lematización |
