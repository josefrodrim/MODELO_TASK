# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Risk Data Scientist evaluation case. The task is to build two regression models (GLM + a ML model of choice) to predict `TARGET` using features X1–X12 from `data_modelo.csv`. Deliverables are a presentation covering preprocessing, modeling, evaluation, interpretation, deployment strategy, and monitoring.

Full requirements are in `Caso Practico Risk Data Scientist.docx`.

## Common Commands

```bash
# Launch Jupyter for analysis
jupyter notebook
# or
jupyter lab

# Run a Python script
python3 script.py

# Install dependencies
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn xgboost lightgbm

# Read the CSV (use latin-1 encoding — file is not UTF-8)
python3 -c "import pandas as pd; df = pd.read_csv('data_modelo.csv', encoding='latin-1'); print(df.shape)"
```

## Dataset: `data_modelo.csv`

- **50,001 rows** (50,000 observations), **15 columns**
- **Encoding:** `latin-1` (not UTF-8 — always pass `encoding='latin-1'`)
- **Train/test split** via `BASE` column: `TRAIN` = 33,334 rows, `TEST` = 16,667 rows
- **Target:** `TARGET` — continuous numeric, range 1,000–40,000, mean ~8,425 (regression task)
- **ID column:** `ID` — not a feature

### Feature Summary

| Column | Type | Missing / Notes |
|--------|------|-----------------|
| X1 | numeric | 786 empty |
| X2 | numeric | 4,993 empty |
| X3 | numeric | 19,922 sentinel values (`-9999998`) |
| X4 | numeric | 15,007 sentinel values (`-9999998`) |
| X5 | numeric | 16,398 empty |
| X6 | numeric | 1 empty |
| X7 | numeric | 551 empty |
| X8 | numeric (low cardinality) | none |
| X9 | **categorical** (Peruvian districts) | 912 empty |
| X10 | numeric | none |
| X11 | numeric | 1,720 empty |
| X12 | numeric | 1,720 empty |

**Key data issue:** `-9999998` is a sentinel for missing in X3 and X4 — must be replaced with `NaN` before any modeling. X11 and X12 are always missing together.

## Required Deliverables (from the case document)

a. GLM model + one ML model (justify ML choice)  
b. Evaluate model assumptions and flag violations  
c. Preprocessing: missing imputation, outlier treatment, transformations — justify all decisions  
d. Performance metrics (AUC, KS, or others) with train vs. validation comparison  
e. Variable interpretation with business/risk coherence  
f. Model selection recommendation with trade-off discussion (performance vs. interpretability, deployment risks)  
g. Stability monitoring strategy (PSI, performance drift metrics)  
h. Conceptual production pipeline (training, scoring, monitoring)
