# Antesala — prototipo

Primer código funcional del proyecto, alineado al documento técnico
(`Antesala_Documento_Tecnico.docx`) y al desafío Bluba de NeuroHack 2026.

## Qué incluye esta primera versión

- `data/generate_synthetic_data.py` — genera bitácoras sintéticas de varios
  niños, con ausencia de datos simulada (MCAR/MAR/MNAR, Sección 4.4).
- `core/bayesian.py` — pooling bayesiano jerárquico (shrinkage), Sección 3.4.
  Validado contra el ejemplo numérico del documento (Niño A / Niño B).
- `core/risk_model.py` — score heurístico ponderado + cálculo de confianza
  (Sección 6.1 "nivel de respaldo" y Sección 6.2).
- `core/recommendations.py` — mapeo driver → recomendación accionable
  (Sección 7).
- `app.py` — interfaz Streamlit: registro del día, riesgo, confianza,
  pregunta sugerida y recomendación, todo integrado.

## Qué falta (ver Carta Gantt, Sección 9 del documento)

- Entrenar el Random Forest / LightGBM (nivel principal, Sección 6.1) —
  por ahora corre solo el score heurístico de respaldo.
- Selector de "la pregunta del día" con reducción de varianza real
  (Sección 6.3) — la versión actual usa un proxy heurístico simple
  (elige la variable faltante de mayor peso clínico).
- Backend FastAPI (Sección 8.2) — hoy la interfaz llama directo a `core/`.

## Cómo correrlo

```bash
pip install -r requirements.txt
python data/generate_synthetic_data.py --out data/bitacoras.csv
streamlit run app.py
```

Abre el enlace que muestra la terminal (por defecto `http://localhost:8501`).
