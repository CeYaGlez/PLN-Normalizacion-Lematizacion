# Normalización y Lematización

Proyecto de PLN que procesa un libro en texto plano usando spaCy y NLTK,
aplicando tokenización, filtrado de stop words, lematización y una
comparativa contra stemming.

## Requisitos

- Python 3.13
- Ver `requirements.txt`

## Instalación

\`\`\`bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm
\`\`\`

## Uso

\`\`\`bash
python main.py
\`\`\`

## ¿Qué hace?

1. Carga el texto de `libro.txt`
2. Tokeniza con spaCy
3. Filtra stop words y puntuación
4. Lematiza cada token
5. Compara resultados de lematización vs stemming (NLTK)