# 📌 Text Mining & Topic Modeling auf Consumer Complaints

Dieses Projekt dient der semantischen Analyse und dem Themen-Extrahieren aus einem Kaggle-Datensatz mit Verbraucherbeschwerden („Consumer Complaints“).
Es kombiniert Vektorisierungstechniken (Bag-of-Words, TF-IDF) mit Themenmodellierungsmethoden (Latente Semantische Analyse (LSA) und Latent Dirichlet Allocation (LDA)).

---

## 🚀 Features

- Automatischer Download und Versionierung des Consumer Complaint Datensatzes von Kaggle.

- Preprocessing Pipeline:
    - Tokenisierung
    - Stopword-Entfernung
    - Lemmatisierung

- Vektorisierung von Texten mit:
    - Bag-of-Words (BoW)
    - TF-IDF

- Vergleich von BoW und TF-IDF:
    - Häufigkeiten vs. gewichtete Relevanz
    - Euklidische Distanz zwischen Dokumenten

- Themenanalyse mit:
    - Latente Semantische Analyse (LSA) → Dimensionsreduktion mit SVD
    - Latent Dirichlet Allocation (LDA) → probabilistisches Modell mit Themenwahrscheinlichkeiten

- Ausgabe von Top-Wörtern pro Thema inkl. prozentualer Verteilung.

---

## 📂 Projektstruktur

```bash
example.env
run.py  
main/
│── dataset_loader.py     # Download von Kaggle-Datensätzen
│── preprocessor.py       # Textbereinigung, Tokenisierung, Lemmatisierung
│── topic_models.py       # LSA- und LDA-Analyzer
│── utils.py              # Hilfsfunktionen (Distanzen, Vergleich, Laden)
│── vectorizers.py        # BoW- und TF-IDF-Vectorizer
```

---

## ⚙️ Installation & Setup

### 1. Repository klonen  
```bash
git clone https://github.com/TimRhr/NLP-DataAnalysis.git
cd NLP-DataAnalysis
```

### 2. Virtuelle Umgebung erstellen & aktivieren
```bash
python -m venv .venv
```
```bash
source .venv/bin/activate   # Linux / macOS
```
```bash
.venv\Scripts\activate      # Windows
```

### 3. Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

### .env Datei konfigurieren (Beispiel siehe example.env):
```ini
# Kaggle API credentials
KAGGLE_USERNAME=dein_name
KAGGLE_KEY=dein_key

# Preprocessing
MAX_FEATURES=100
MAX_TOKENS=50
MAX_ROWS=0
COLUMN_NAME=narrative

# Topic Modeling
N_LSA_TOPICS=5
N_LDA_TOPICS=5
N_WORDS_PER_TOPIC=8
COMPARE_DOC_IDX=5
```

## ▶️ Nutzung
Das Hauptskript starten:

```bash
python run.py
```

Ablauf:
1. Dataset herunterladen (falls nicht vorhanden).
2. Preprocessing durchführen oder gecachte Version laden.
3. BoW und TF-IDF Vektorisierungen erstellen.
4. Vergleich der Features zwischen BoW und TF-IDF für ein Dokument (COMPARE_DOC_IDX).
5. Euklidische Distanzen zwischen Dokumenten berechnen.
6. Semantische Analyse mit LSA und LDA zur Extraktion der Themen im Korpus.

## 🧠 Interpretation

- BoW vs TF-IDF:
    - BoW zeigt absolute Worthäufigkeiten, TF-IDF hebt kontextuell relevante Wörter hervor.
    - Ein Wort mit hoher BoW, aber niedrigem TF-IDF → häufig, aber wenig aussagekräftig.
    - Ein Wort mit niedriger BoW, aber hohem TF-IDF → selten, aber charakteristisch für das Dokument.

- Dokumentenähnlichkeit:
    - Niedrige Distanzen → Dokumente inhaltlich ähnlich.
    - Hohe Distanzen → unterschiedliche Themen oder Fokus.

- Themenanalyse:
    - LSA: Zeigt Muster und Zusammenhänge zwischen Wörtern, eher latente Strukturen. Negative Werte reflektieren Gegensätze oder Abwesenheit im Thema.
    - LDA: Liefert klare, probabilistische Themen. Werte geben die relative Bedeutung jedes Wortes im Thema an.

- Coherence Scores: Helfen bei der Beurteilung, ob die extrahierten Themen sinnvoll und inhaltlich konsistent sind.
