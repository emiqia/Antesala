# Antesala — prototipo

Código funcional del proyecto, alineado al documento técnico
(`docs/Antesala_Documento_Tecnico.docx`) y al desafío Bluba de NeuroHack 2026.
El esquema de campos y categorías está alineado a los datos reales
anonimizados de Bluba (`docs/Datos base/*.csv`) y a las capturas de la app
móvil (`docs/5. Presentación BLUBA.pdf`), no a nombres inventados.

## Qué incluye

- `data/generate_synthetic_data.py` — genera bitácoras sintéticas de varios
  niños (más un puñado de niños de **arranque en frío**, con solo 2-5 días de
  historial, para poder demostrar el escenario "niño nuevo" de la Sección 3.6),
  con ausencia de datos simulada (MCAR/MAR/MNAR, Sección 4.4). La probabilidad
  de crisis de mañana depende explícitamente de la severidad observada hoy
  (sin esto, el modelo no puede aprender ningún patrón predictivo real).
- `core/bayesian.py` — pooling bayesiano jerárquico (shrinkage), Sección 3.4.
  Validado contra el ejemplo numérico del documento (Niño A / Niño B) — ver
  `tests/test_bayesian.py`.
- `core/features.py` — ingeniería de variables (Sección 4.3): ventanas
  móviles, línea base bayesiana por variable, antigüedad de registro, etc.
  Sin fuga temporal y con paridad exacta entre entrenamiento e inferencia —
  ver `tests/test_features.py` y `scripts/validate_features.py`.
- `core/train_model.py` — entrena el Random Forest (nivel principal, Sección
  6.1) sobre `crisis_24h`, con split por niño (evalúa generalización a niños
  no vistos) y guarda el modelo + `mu` poblacional + feature importance en
  `models/antesala_rf.joblib`. También calcula `confidence_weights`: los
  pesos wᵢ del cálculo de confianza (Sección 6.2) tomados del feature
  importance real del modelo — no de los pesos clínicos manuales, que el
  documento reserva para el score heurístico (Sección 6.1).
- `core/risk_model.py` — motor de riesgo completo: usa el Random Forest
  cuando está disponible y cae al score heurístico ponderado (nivel de
  respaldo, Sección 6.1) si no. Cálculo de confianza (Sección 6.2).
- `core/question_selector.py` — selector de "la pregunta del día" por
  **reducción esperada de varianza del ensamble** (Sección 6.3, nivel
  principal): simula 2-3 valores probables por variable faltante (historial
  propio del niño, o poblacional si es nuevo) y elige la que más reduce la
  dispersión entre árboles del Random Forest. Cae al proxy heurístico (mayor
  peso clínico) si el modelo no está disponible.
- `core/recommendations.py` — mapeo driver → recomendación accionable
  (Sección 7).
- `app.py` — interfaz Streamlit con tres pestañas:
  - **Hoy**: el flujo principal — UNA sola pregunta (la del día), nada más.
  - **Registro completo**: bitácora extendida opcional con las 14 variables
    de la Sección 4.1.
  - **Riesgo**: panel visual (gauges, completitud, variables más predictivas,
    tendencia histórica) que responde a los 6 requisitos explícitos de las
    bases del desafío.
- `tests/` — suite de pytest (pooling bayesiano contra el ejemplo del
  documento, ingeniería de variables sin fuga, motor de riesgo, selector de
  pregunta por varianza).

## Auditoría contra las bases oficiales (`docs/Bases Hackatón FICA...pdf`)

Una revisión completa contra el documento técnico y las bases oficiales de
la hackatón encontró y corrigió tres problemas reales:

1. **Imputación silenciosa de facto en `n_eventos_desregulacion`**: como en
   los datos sintéticos ese campo nunca faltaba, el Random Forest no tenía
   forma de aprender un indicador de ausencia para él, y trataba "confirmado
   que no hubo episodio" igual que "no sabemos" — violando la Sección 4.4.
   Corregido: el registro de un episodio ahora se trata como una acción
   compuesta que puede faltar completa (como en la app real).
2. **Antigüedad de registro solo en 2 de 14 variables** — la Sección 4.4
   pide esa pieza de información "por cada variable". Ahora se calcula para
   las 14.
3. **Los pesos de confianza usaban los pesos clínicos del score heurístico**
   en vez del feature importance del modelo, que es lo que pide
   explícitamente la Sección 6.2. Separado en dos fuentes independientes.

El ROC AUC bajó de ~0.95 a ~0.73 al corregir el punto 1 — la cifra anterior
reflejaba en parte que el modelo explotaba un atajo de los datos, no
capacidad predictiva real.

## Qué falta

**Código (ver Carta Gantt, Sección 9 del documento):**
- Backend FastAPI (Sección 8.2) — hoy la interfaz llama directo a `core/`.
  Declarado como extensión, no como bloqueante del MVP.
- Modelo bayesiano completo con PyMC (Sección 3.7, Nivel 2) — el shrinkage
  manual (Nivel 1) ya está implementado y validado; PyMC es explícitamente
  "no indispensable para el MVP" según el documento.
- LightGBM, Whisper, Docker (Sección 8.4-8.5) — stretch goals declarados.

**Entregables no-código (bases oficiales, Sección 5.2):** para la entrega del
26 de agosto 23:59 (Google Forms) hace falta, además del código: una
**presentación** con capturas de pantalla, diagramas arquitectónicos y
enlaces (repo, video demostrativo de máx. 2 min o link al prototipo). Nada
de esto existe todavía en el repositorio.

## Cómo correrlo

```bash
pip install -r requirements.txt
python data/generate_synthetic_data.py --out data/bitacoras.csv
python core/train_model.py
streamlit run app.py
```

Abre el enlace que muestra la terminal (por defecto `http://localhost:8501`).

## Cómo correr los tests

```bash
pytest
```

Algunos tests requieren que ya exista `models/antesala_rf.joblib` (se saltan
automáticamente si no está entrenado todavía).

## Cómo validar la ingeniería de variables manualmente

```bash
python scripts/validate_features.py
```

Imprime, paso a paso: cobertura de la matriz de variables, un caso de
prueba día a día, verificación de ausencia de fuga temporal, consistencia
con `core/bayesian.py`, y paridad entrenamiento/inferencia.
