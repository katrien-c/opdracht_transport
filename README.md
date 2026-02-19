# Learning driver preferences

Een Python project voor het analyseren van transport route data, inclusief taken, routes, afstanden en bewegingen over tijd.

## 📋 Inhoudsopgave

- [Overzicht](#overzicht)
- [Team](#team)
- [Installatie](#installatie)
- [Gebruik](#gebruik)
- [Project Structuur](#project-structuur)
- [Features](#features)

## 🎯 Overzicht

Dit project analyseert transport data uit JSON bestanden (requests en responses) en genereert visualisaties en statistieken over:
- Totale taken per dag
- Routes per dag
- Gemiddelde taken per route
- Kilometers per route
- Bewegingen per route

## � Team

Dit project is ontwikkeld door:
- **Christian Belinga**
- **Katrien Cogghe**
- **Dirk De Mulder**
- **Pascal Delannoye**

## �🚀 Installatie

### Vereisten

- Python 3.8 of hoger
- pip (Python package manager)

### Stap 1: Clone of Download het Project

Download het project naar je lokale machine of navigeer naar de project directory:

```bash
cd pad/naar/opdracht_transport
```

### Stap 2: Installeer Python Dependencies

#### Methode 1: Via requirements.txt (Aanbevolen)

Open een terminal (PowerShell of Command Prompt) in de project directory:

```bash
# Installeer alle dependencies in één keer
pip install -r requirements.txt
```

#### Methode 2: Met Virtual Environment (Best Practice)

```bash
# Maak een virtual environment
python -m venv venv

# Activeer de virtual environment
# Voor Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Voor Windows Command Prompt:
.\venv\Scripts\activate.bat

# Installeer dependencies vanuit requirements.txt
pip install -r requirements.txt
```

#### Methode 3: Handmatige installatie

```bash
pip install pandas matplotlib numpy ipython ipywidgets
```

### Stap 3: Installeer Jupyter (optioneel, voor notebooks)

Als je de Jupyter notebooks wilt gebruiken:

```bash
pip install jupyter notebook
```

Voor JupyterLab:

```bash
pip install jupyterlab
```

### Stap 4: Verifieer de Installatie

Test of alle packages correct zijn geïnstalleerd:

```bash
python -c "import pandas, matplotlib, numpy, IPython, ipywidgets; print('✓ Alle packages succesvol geïnstalleerd!')"
```

## 📖 Gebruik

### Optie 1: Python Script

Voer het main script uit vanuit de scripts directory:

```bash
cd scripts
python main.py
```

Dit zal:
1. Data inladen uit `data/requests/` en `data/responses/`
2. Dataframes creëren en transformeren
3. Verschillende plots genereren
4. Een interactieve route selector tonen in de console

### Optie 2: Jupyter Notebook (Aanbevolen)

Start Jupyter en open het main notebook:

```bash
# Start Jupyter Notebook
jupyter notebook

# Of JupyterLab
jupyter lab
```

Navigeer naar `scripts/main.ipynb` en voer de cellen uit.

**Voordelen van het notebook:**
- Interactieve visualisaties
- Dropdown widget voor route selectie
- Stap-voor-stap uitvoering
- Duidelijke documentatie per sectie

### Optie 3: VS Code met Jupyter Extension

1. Open VS Code in de project directory
2. Installeer de "Jupyter" extension van Microsoft
3. Open `scripts/main.ipynb`
4. Klik op "Run All" of voer cellen individueel uit

## 📁 Project Structuur

```
opdracht_transport/
├── data/
│   ├── requests/          # Input JSON bestanden (transport requests)
│   ├── responses/         # Output JSON bestanden (transport responses)
│   ├── output/            # Gegenereerde output bestanden
│   └── ruw/              # Ruwe data (parquet bestanden)
├── scripts/
│   ├── main.py           # Hoofd Python script
│   ├── main.ipynb        # Hoofd Jupyter notebook (AANBEVOLEN)
│   ├── config.py         # Project configuratie en paths
│   ├── create_dataframes.py  # Data inlees en transformatie functies
│   ├── plots.py          # Visualisatie functies
│   ├── styling.py        # Plot styling configuratie
│   └── utils/            # Utility functies (file handling, helpers)
├── notebooks/            # Aanvullende analyse notebooks
└── README.md            # Deze handleiding
```

## 🎨 Features

### Data Processing

- Automatisch inlezen van JSON bestanden uit requests/responses directories
- Transformatie van geneste data structuren
- Berekening van KPIs (totale taken, afstanden, bewegingen)
- Filtering op tijd (< 18:00 uur)

### Visualisaties

1. **Totale Taken per Dag**: Input vs Output vergelijking
2. **Routes per Dag**: Aantal actieve routes
3. **Gemiddelde Taken per Route**: Per dag analyse
4. **Kilometers per Route**: Gemiddelde afstand per route per dag
5. **Bewegingen per Route**: Aantal stops/bewegingen analyse
6. **Route-specifieke Plots**: Gedetailleerde analyse per route

### Interactiviteit

- **Console selectie** (main.py): Kies een route uit een genummerde lijst
- **Dropdown widget** (main.ipynb): Interactieve route selectie met directe visualisatie

## 🔧 Troubleshooting

### Module Import Errors

Als je `ModuleNotFoundError` krijgt:

```bash
# Zorg dat je in de juiste directory bent
cd scripts

# Of voeg het scripts pad toe aan PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;pad\naar\opdracht_transport\scripts
```

### Data Niet Gevonden

Controleer of de data directories bestaan en JSON bestanden bevatten:
- `data/requests/0521_301-YYYYMMDD/` (met JSON bestanden)
- `data/responses/` (met corresponderende response bestanden)

### Jupyter Widget Werkt Niet

Activeer ipywidgets voor Jupyter:

```bash
jupyter nbextension enable --py widgetsnbextension
```

Voor JupyterLab:

```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

## 📞 Support

Voor vragen of problemen, controleer:
1. Of alle dependencies correct zijn geïnstalleerd
2. Of de data directories de juiste structuur hebben
3. Of Python 3.8+ wordt gebruikt

## 📝 Licentie

Zie het LICENSE bestand voor details.
