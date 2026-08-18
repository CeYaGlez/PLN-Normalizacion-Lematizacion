# Procesamiento de Lenguaje Natural — PLN

> ¿Puede una computadora entender de qué habla un libro sin que nadie se lo explique?
> Este proyecto lo intenta.

Se tomó *Can't Hurt Me* de David Goggins en texto plano y se analizó con PLN (Procesamiento de Lenguaje Natural), el área de la inteligencia artificial que estudia cómo las computadoras pueden leer, interpretar y trabajar con texto humano.

---

## ¿Qué pasa aquí, en términos simples?

Imagina que le das el libro a alguien que no habla español. No entiende nada, pero puede contar cuántas veces aparece cada palabra, cuáles aparecen juntas siempre, cuáles son raras. Con solo esas estadísticas, puede adivinar los temas del libro.

Eso es exactamente lo que hace este proyecto. Y funciona.

---

## Flujo del proyecto

```
libro.txt
    │
    ▼
Limpieza del texto          → quitar signos y stop words ("el", "la", "de"...)
    │
    ▼
Lematización                → "corriendo" y "corrió" se convierten en "correr"
    │
    ▼
Vectorización               → cada palabra se convierte en números comparables
    │         │
   BoW      TF-IDF          → dos formas distintas de pesar las palabras
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

Antes de analizar el texto, se eliminan las *stop words*, palabras tan comunes ("el", "la", "y") que no aportan significado. Luego se aplica *lematización*: reducir cada palabra a su forma base para que "corriendo", "corrió" y "correrá" cuenten como la misma unidad.

Con el texto limpio, se generan dos tipos de representación vectorial, es decir, se convierte cada palabra en un vector de números para que la computadora pueda operar con ella matemáticamente.

![Visualización 3D BoW vs TF-IDF](assets/Captura3D.png)

**Bag of Words (BoW, izquierda)** construye un vector contando cuántas veces aparece cada palabra en cada oración. Es el modelo más simple posible. El problema se ve al instante: "yo" y "él" dominan y distorsionan todo el espacio por puro volumen de apariciones. Tienen sentido en una autobiografía, pero no dicen nada sobre los *temas* del libro.

**TF-IDF (derecha)** corrige eso. Premia las palabras que son frecuentes en una oración específica pero raras en el resto del corpus, las que realmente la distinguen, y penaliza las genéricas. El resultado es radicalmente distinto: los outliers más alejados del centro son "kilómetro", "correr", "carrera", "poder" y "vida". Sin entender una sola palabra, el algoritmo identificó estadísticamente los temas centrales del libro. Incluso "él", que en BoW era ruido, se vuelve distintivo en TF-IDF porque aparece concentrado en pasajes muy específicos.

> **BoW te dice quién habla. TF-IDF te dice de qué habla.**

---

### Word2Vec — cuando las palabras aprenden su propio significado

BoW y TF-IDF trabajan con conteos. Word2Vec es una red neuronal que va más lejos: en lugar de contar palabras, **aprende el contexto** en el que aparecen. Cada palabra queda representada como un vector de 50 números (un *embedding*) posicionado en un espacio matemático donde las palabras con contextos similares quedan cerca entre sí. Sin que nadie le haya dicho qué significa ninguna.

El modelo se entrenó con la arquitectura **Skip-gram**: dada una palabra, predice qué otras palabras suelen aparecer a su alrededor. Se eligió sobre su alternativa (CBOW) porque captura mejor las relaciones semánticas finas en corpus de tamaño moderado. Como el espacio tiene 50 dimensiones, se usó PCA (reducción de dimensionalidad) para proyectarlo a 3 y poder visualizarlo.

![Espacio Semántico Word2Vec - Embeddings 3D](assets/embeddings_3d_goggins.png)

La nube densa de la derecha concentra el vocabulario genérico. Lo interesante está en los outliers: `"kilómetro"` es el punto más aislado de todo el espacio porque siempre aparece en el mismo tipo de oración. Debajo de la nube, `"correr"`, `"carrera"`, `"hora"`, `"minuto"` y `"ciento"` quedaron agrupados sin instrucciones, porque comparten universo semántico. Más a la izquierda, `"entrenamiento"`, `"infernal"` y `"semana"` forman su propio cluster: las semanas de entrenamiento brutal tienen su propia firma lingüística dentro del libro.

`"bud"`, el apodo del padre de Goggins, queda completamente aislado en el extremo izquierdo. Aparece en pasajes tan específicos y emocionalmente distintos que ninguna otra palabra del libro se le acerca. `"seal"` (SEAL Teams) también se separa por la misma razón.

> **TF-IDF te dice qué palabras importan. Word2Vec te dice qué palabras significan lo mismo.**

---

## Limitaciones

Word2Vec se entrenó únicamente con este libro, así que sus embeddings reflejan las relaciones del texto de Goggins, no del idioma en general. Una palabra puede quedar cerca de otra simplemente porque ambas aparecen en situaciones similares dentro de esta obra.

Que una palabra aparezca aislada en la gráfica tampoco significa automáticamente que sea un tema central del libro. Su posición depende de la frecuencia, el tamaño del corpus, los hiperparámetros del modelo y la propia reducción dimensional. Las visualizaciones son evidencia de patrones estadísticos, no una interpretación definitiva del texto.

---

## Próximos pasos

El proyecto puede extenderse bastante. Algunas direcciones interesantes serían comparar Skip-gram vs CBOW, experimentar con distintos tamaños de embedding y ventana de contexto, entrenar el modelo con varios libros del mismo autor o del mismo género, y aplicar clustering para detectar grupos de palabras automáticamente.

La pregunta natural que sigue es: ¿los patrones que aparecen en este libro se mantienen cuando el corpus crece?

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