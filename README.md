# Modelo de Riesgo Crediticio — Scotiabank Perú

Pipeline completo de ciencia de datos para la estimación de score de riesgo crediticio:
EDA → Preprocesamiento → Modelado (GLM + LightGBM) → Evaluación → API REST → Docker.

---

## Estructura del proyecto

```
├── notebooks/                  # Análisis reproducible paso a paso
│   ├── 01_eda.ipynb            # Análisis exploratorio de datos
│   ├── 02_preprocessing.ipynb  # Preprocesamiento y pipeline
│   ├── 03_glm_model.ipynb      # Modelo GLM (OLS log-normal)
│   ├── 04_ml_model_lgbm.ipynb  # Modelo LightGBM + Optuna + SHAP
│   └── 05_evaluacion_comparacion.ipynb  # Comparación, PSI, recomendación
│
├── src/                        # Código de producción (importable)
│   ├── config.py               # Rutas y constantes centralizadas
│   ├── data/preprocessing.py   # RiskPreprocessor (sklearn-compatible)
│   ├── models/glm_model.py     # GLMRiskModel (statsmodels)
│   ├── models/ml_model.py      # LGBMRiskModel (LightGBM + Optuna)
│   ├── evaluation/metrics.py   # RMSE, MAE, R², Gini, KS, PSI
│   └── api/                    # FastAPI: endpoints de predicción
│
├── data/
│   ├── raw/data_modelo.csv     # Dataset original (50K obs, latin-1)
│   └── processed/              # Parquet generados por notebook 02
│
├── models/                     # Artefactos serializados (joblib)
├── reports/figures/            # Gráficos generados por los notebooks
├── tests/                      # Tests unitarios (pytest)
├── Dockerfile                  # Multi-stage build (builder + runtime)
└── docker-compose.yml          # API + Jupyter (perfil dev)
```

## Instalación

```bash
# Clonar e instalar dependencias
git clone <repo-url>
cd MODELO_TASK
pip install -r requirements.txt
```

## Ejecutar los notebooks (flujo recomendado)

```bash
jupyter notebook notebooks/
```

**Orden de ejecución:**

1. `01_eda.ipynb` — EDA y generación de figuras en `reports/figures/`
2. `02_preprocessing.ipynb` — Genera `data/processed/` y `models/preprocessor.pkl`
3. `03_glm_model.ipynb` — Genera `models/glm_model.pkl`
4. `04_ml_model_lgbm.ipynb` — Genera `models/lgbm_model.pkl` (requiere ~10 min en CPU para Optuna)
5. `05_evaluacion_comparacion.ipynb` — Comparación final y recomendación

## API de Predicción

### Ejecutar localmente

```bash
# Asegurarse de haber corrido los notebooks primero (modelos en models/)
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# O con Make
make api
```

### Ejecutar con Docker

```bash
make docker-build
make docker-run

# Solo API
docker-compose up api

# API + Jupyter Lab
docker-compose --profile dev up
# → Jupyter: http://localhost:8888 (token: scotia2024)
```

### Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Estado del servicio |
| `GET` | `/model/info` | Metadata de modelos cargados |
| `POST` | `/predict` | Predicción individual |
| `POST` | `/predict/batch` | Predicción en lote (≤ 10K) |

Documentación interactiva (Swagger): `http://localhost:8000/docs`

### Ejemplo de predicción

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "X1": 207137,
    "X2": 6427,
    "X3": null,
    "X5": 3222,
    "X6": 1,
    "X7": 633,
    "X8": 1,
    "X9": "SANTIAGO DE SURCO",
    "X10": 30170,
    "X11": 146,
    "X12": 47,
    "model": "lgbm"
  }'
```

## Tests

```bash
# Correr todos los tests
make test

# Test específico
pytest tests/test_preprocessing.py -v
pytest tests/test_metrics.py -v
```

## Dataset

- **Archivo:** `data/raw/data_modelo.csv`
- **Encoding:** `latin-1` (requerido — el archivo contiene caracteres especiales)
- **Filas:** 50,001 observaciones
- **Split:** columna `BASE` → `TRAIN` (33,334) / `TEST` (16,667)
- **Target:** `TARGET` — continuo 1,000–40,000 PEN
- **Nota crítica:** X3 y X4 usan `-9999998` como valor centinela de ausencia

## Decisiones técnicas clave

| Aspecto | Decisión | Justificación |
|---------|----------|---------------|
| GLM | OLS sobre log(TARGET) | TARGET positivo y skewed → log-normal |
| ML | LightGBM | Mejor performance en tabulares; SHAP nativo; baja latencia |
| Tuning | Optuna TPE | 10x más eficiente que GridSearch en espacios grandes |
| Outliers | Capping IQR×3 | Preserva variabilidad legítima en variables de ingresos |
| Missing | Mediana + flags | Robusta a skew; preserva señal de ausencia |
| X9 | OHE top-25 + OTROS | Sin leakage; maneja distribución de Pareto |
| API | FastAPI + Pydantic v2 | Async nativo; validación automática; OpenAPI docs |
| Contenedor | Docker multi-stage | Imagen liviana (~500MB vs ~2GB en single-stage) |

## Monitoreo en producción

El notebook 05 detalla la estrategia completa. En resumen:

- **PSI mensual** sobre scores y features de entrada
- **Umbrales:** PSI < 0.10 | 0.10–0.25 | > 0.25 reentrenar
- **Métricas de performance** trimestrales sobre datos etiquetados (rezago 90 días)
- **Trigger automático** de reentrenamiento si PSI > 0.25 por 2 meses consecutivos

---

*Desarrollado como caso práctico de evaluación — Risk Data Scientist, Scotiabank Perú.*
