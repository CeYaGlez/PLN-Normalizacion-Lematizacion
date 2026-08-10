# Procesamiento de Lenguaje Natural — PLN

Proyecto de PLN que procesa un libro en texto plano usando spaCy y NLTK.
Cubre desde la normalización básica hasta representación vectorial y visualización semántica en 3D.

## Requisitos

- Python 3.13
- Ver `requirements.txt`

## Instalación

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

## Uso

```bash
python main.py
```

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

```
.
├── main.py          # Script principal
├── libro.txt        # Texto a procesar
├── assets/          # Imágenes y recursos
└── requirements.txt
```

## Visualización del espacio vectorial

Representación 3D de las 100 palabras más frecuentes del corpus (*Can't Hurt Me* — David Goggins).
A la izquierda el espacio **BoW** (por conteos), a la derecha **TF-IDF** (por importancia relativa).

![Visualización 3D BoW vs TF-IDF](assets/Captura3D.png)

**Bag of Words (BoW)** representa cada oración como un vector de conteos de palabras,
sin considerar el orden ni la relevancia de cada término.

**TF-IDF** pondera cada palabra según qué tan frecuente es en una oración específica
pero qué tan rara es en el resto del corpus — destacando términos distintivos
y penalizando los genéricos.

## Conclusiones

En el espacio **BoW**, "yo" y "él" dominan y distorsionan toda la representación por puro
conteo — tiene sentido en una autobiografía escrita en primera persona, pero no aportan
significado temático real.

En el espacio **TF-IDF**, los outliers más alejados del centro son "kilómetro", "correr",
"carrera", "poder" y "vida" — las palabras más *distintivas* del corpus. Sin entender
una sola palabra, el algoritmo identificó estadísticamente los temas centrales del libro:
resistencia física, distancia recorrida y fortaleza mental.

**BoW te dice quién habla. TF-IDF te dice de qué habla.**

## Dependencias principales

| Librería | Uso |
|---|---|
| spaCy | Tokenización, lematización, stop words |
| NLTK | Stemming comparativo |
| scikit-learn | Vectorización (BoW, TF-IDF) y reducción de dimensionalidad (PCA) |
| matplotlib | Visualización 3D del espacio vectorial |
| pandas | Tabla comparativa stemming vs lematización |

