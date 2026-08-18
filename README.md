# Procesamiento de Lenguaje Natural — PLN

> ¿Puede una computadora entender de qué habla un libro sin que nadie se lo explique?
> Este proyecto lo intenta.

Se tomó *Can't Hurt Me* de David Goggins en texto plano y se procesó con técnicas de PLN — desde limpiar el texto hasta entrenar un modelo que aprende el **significado** de las palabras por cómo se usan.

---

## ¿Qué pasa aquí, en términos simples?

Imagina que le das el libro a alguien que no habla español. No entiende nada, pero puede contar cuántas veces aparece cada palabra, cuáles aparecen juntas siempre, cuáles son raras. Con solo esas estadísticas, puede adivinar los temas del libro.

Eso es exactamente lo que hace este proyecto — y funciona.

---

## Flujo del proyecto

```
libro.txt
    │
    ▼
Limpieza del texto          → quitar signos, stop words ("el", "la", "de"...)
    │
    ▼
Lematización                → "corriendo" y "corrió" se convierten en "correr"
    │
    ▼
Vectorización               → cada palabra se convierte en números que la computadora puede comparar
    │         │
   BoW      TF-IDF          → dos formas distintas de "pesar" las palabras
    │
    ▼
Word2Vec                    → el modelo aprende el significado por contexto
    │
    ▼
Visualización 3D            → el espacio matemático proyectado para poder verlo
```

---

## Resultados

### Bag of Words vs TF-IDF

Las mismas palabras, dos formas de representarlas.

![Visualización 3D BoW vs TF-IDF](assets/Captura3D.png)

**BoW (izquierda)** cuenta cuántas veces aparece cada palabra. Simple y directo, pero "yo" y "él" dominan todo el espacio — tiene sentido en una autobiografía, pero no dicen nada útil sobre los *temas* del libro.

**TF-IDF (derecha)** premia las palabras que son frecuentes *en una oración* pero raras *en el resto del libro* — las que realmente la distinguen. Los outliers más alejados del centro son "kilómetro", "correr", "carrera", "poder" y "vida". Sin entender una sola palabra, el algoritmo identificó los temas centrales: resistencia física y fortaleza mental.

> **BoW te dice quién habla. TF-IDF te dice de qué habla.**

---

### Word2Vec — cuando las palabras aprenden su propio significado

BoW y TF-IDF trabajan con conteos. Word2Vec hace algo más interesante: **aprende el contexto** en el que cada palabra aparece y le asigna una posición en un espacio matemático. Palabras que se usan en situaciones parecidas quedan cerca entre sí.

El modelo se entrenó con **Skip-gram** — en lugar de predecir una palabra a partir de sus vecinas, predice las vecinas a partir de la palabra central. Captura mejor las relaciones semánticas finas, especialmente en corpus de tamaño moderado.

![Espacio Semántico Word2Vec - Embeddings 3D](assets/embeddings_3d_goggins.png)

Lo que muestra la gráfica:

- **La nube densa (derecha):** palabras de uso genérico que aparecen en contextos muy variados. El modelo no las diferencia con fuerza porque son intercambiables.
- **La cola y los outliers:** "kilómetro", "correr", "entrenamiento", "infernal", "semana" — aparecen siempre en el mismo tipo de oraciones. El modelo aprendió que son únicas.
- **"bud"** está completamente aislado: es el apodo del padre de Goggins. Aparece en un contexto tan específico que ninguna otra palabra se le acerca.
- **"seal"** (SEAL Teams) también queda separado — el entrenamiento militar de élite tiene su propio universo semántico dentro del libro.

> **TF-IDF te dice qué palabras importan. Word2Vec te dice qué palabras significan lo mismo.**

---

## Instalación

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

**Requisitos:** Python 3.13

## Uso

```bash
python main.py
```

## Estructura del proyecto

```
.
├── main.py               # Script principal
├── libro.txt             # Texto a procesar
├── assets/
│   ├── Captura3D.png              # Visualización BoW vs TF-IDF
│   └── embeddings_3d_goggins.png  # Visualización Word2Vec
└── requirements.txt
```

## Dependencias principales

| Librería | Uso |
|---|---|
| spaCy | Tokenización, lematización, stop words |
| NLTK | Stemming comparativo |
| scikit-learn | Vectorización (BoW, TF-IDF) y reducción de dimensionalidad (PCA) |
| gensim | Entrenamiento del modelo Word2Vec |
| matplotlib | Visualización 3D del espacio vectorial |
| pandas | Tabla comparativa stemming vs lematización |