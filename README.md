# Text Vectorization & Similarity Analysis  

Dieses Projekt demonstriert die Analyse von Textdaten anhand des Datensatzes **Consumer Complaints**.  
Ziel ist es, Texte durch unterschiedliche Methoden in numerische Repräsentationen zu überführen und die Dokumentähnlichkeit mithilfe von euklidischen Distanzen zu vergleichen und die Hauptthemen zu extrahieren.

---

## ⚙️ Installation & Nutzung  

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

### 4. Analysen starten
run.py ausführen, um:
- die Vektorisierung durchzuführen,
- Distanzen zu berechnen,
- Ergebnisse zu vergleichen.