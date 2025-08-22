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
6. Semantische Analyse mit LSA und LDA durchführen.

Beispielausgabe:

            Word  BoW    TF-IDF
payment  payment    6  0.449771
issue      issue    5  0.463652
report    report    4  0.195095
...

BoW Euclidean distances (first 5 docs):
           Doc0       Doc1       Doc2       Doc3       Doc4
Doc0   0.000000  16.278821  16.792856  16.340135  15.842980
Doc1  16.278821   0.000000  15.779734  15.491933  14.966630
...

TF-IDF Euclidean distances (first 5 docs):
          Doc0      Doc1      Doc2      Doc3      Doc4
Doc0  0.000000  1.279001  1.301569  1.252626  1.242231
Doc1  1.279001  0.000000  1.161487  1.228888  1.216835
...

--- LSA Topics ---
Topic 1 (account): account (6.2%), credit (5.6%), report (5.3%), ...
...

--- LDA Topics ---
Topic 1 (card): card (5.6%), call (5.3%), account (4.6%), ...
...

## 🧠 Interpretation

- LSA findet latente Strukturen, kann aber negative Werte in den Gewichtungen erzeugen, was die Interpretation erschwert.

- LDA liefert klarere, probabilistische Themen mit gut interpretierbaren Verteilungen.

- Beide Methoden identifizieren ähnliche Kernthemen:
    - account, credit, consumer, payment, debt
