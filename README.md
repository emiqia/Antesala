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
- `core/bayesian.py` — **partial pooling / shrinkage empirical-Bayes**,
  Sección 3.4. No es un modelo jerárquico bayesiano completamente estimado
  (no hay MCMC ni posteriores), y el módulo lo dice explícitamente. Lo que
  sí es: para una proporción, θ = (k·μ + n·ȳ)/(k + n) es **exactamente** la
  media posterior de un Beta(k·μ, k·(1−μ)) con verosimilitud binomial, y
  como μ se estima de la propia población, es empirical Bayes por
  definición. `estimate_prior_strength()` además **estima k de los datos**
  por momentos en vez de fijarlo a mano. Validado contra el ejemplo numérico
  del documento (Niño A / Niño B) — ver `tests/test_bayesian.py`.
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
  respaldo, Sección 6.1) si no. Aplica la calibración isotónica antes de
  mostrar la probabilidad. Calcula el **índice de suficiencia de
  información** (Sección 6.2) — que ya no se llama "confianza", porque no es
  la probabilidad de que la predicción sea correcta.
- `core/uncertainty.py` — **incertidumbre predictiva**, el tercer número,
  separado del riesgo y de la suficiencia. Dispersión entre los árboles del
  ensamble, normalizada por la dispersión máxima posible para esa media (sin
  normalizar, toda predicción extrema parecería estable solo por ser
  extrema). Declarado como *proxy computacional de inestabilidad*, no como
  incertidumbre calibrada.
- `core/evaluation.py` — panel de métricas y las dos particiones (por niño y
  por tiempo). No reporta accuracy a propósito.
- `core/question_selector.py` — "la pregunta del día" como **adquisición
  activa de información** (Sección 6.3). Simula 2-3 valores probables por
  variable faltante (historial propio del niño, o poblacional si es nuevo) y
  mide cuánto se estrecha la dispersión entre árboles con cada respuesta
  posible. La elección final **no** es la ganancia bruta sino la utilidad
  neta `U = ganancia − λ·carga`, donde la carga se descompone en factores
  auditables (cuántos campos hay que llenar, si lo responde la familia o hay
  que esperar al colegio, si obliga a volver sobre un episodio difícil).
  Puede además decidir **no preguntar nada**: si ninguna variable faltante
  tiene ganancia positiva, preguntar solo agrega carga. Cae al proxy
  heurístico (mayor peso clínico) si el modelo no está disponible.
- `core/recommendations.py` — mapeo driver → recomendación accionable
  (Sección 7).
- `core/narrative.py` — explicación narrativa en el formato exacto que dan
  las bases como ejemplo ("Durante los últimos tres días Juan ha presentado
  … Se detecta un riesgo alto … Considere …"): observación → nivel de riesgo
  → estrategia preventiva. Es una tabla de reglas determinista, no un
  generador de texto: cada frase es trazable al dato que la originó, y cada
  señal declara si viene de la familia, del colegio o del equipo profesional.
  Con confianza baja no afirma un nivel de riesgo como hecho, sino que emite
  la alerta de información insuficiente que piden las bases.
- `app.py` — interfaz Streamlit con tres vistas, separadas por AUDIENCIA
  porque las bases piden explicaciones "para las familias y profesionales",
  que no son la misma persona ni necesitan la misma pantalla:
  - **Hoy · familia**: replica la app móvil de Bluba dentro de un marco de
    teléfono. UNA sola pregunta al día y la explicación en lenguaje cotidiano.
    Al lado, un panel "detrás de la pantalla" que muestra las señales
    detectadas con su evidencia y el **ranking completo** de preguntas
    candidatas por reducción esperada de varianza — se ve que la elegida gana
    de verdad, y que algunas variables incluso aumentarían la incertidumbre.
  - **Panel del equipo**: triage de toda la cohorte ordenada por riesgo. Las
    alertas con confianza baja se muestran **suprimidas en gris** en vez de
    presentarse como certeza.
  - **Bitácora completa**: las 14 variables de la Sección 4.1, opcional.
- `scripts/benchmark.py` — **comparadores** (Sección 18): tasa global, tasa
  individual, partial pooling solo, logística sin personalizar, RF sin
  personalizar y Antesala completo, más ablaciones por bloque de variables y
  un barrido de k. Responde con números si la complejidad adicional se paga.
- `scripts/sensibilidad_ausencia.py` — genera el mismo mundo bajo MCAR, MAR,
  MNAR y mixto, y entrena el modelo en cada uno. Convierte el supuesto sobre
  datos faltantes en algo medible en vez de afirmado.
- `tests/` — suite de pytest, 76 tests (partial pooling contra el ejemplo del
  documento, ingeniería de variables sin fuga, motor de riesgo, selector de
  pregunta con carga, separación riesgo/suficiencia/incertidumbre, métricas y
  particiones sin fuga).

## Auditoría contra las bases oficiales (`docs/Bases Hackatón FICA...pdf`)

Una revisión completa contra el documento técnico y las bases oficiales de
la hackatón encontró y corrigió cuatro problemas reales:

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
4. **El día completamente sin registrar no existía en los datos**: la ausencia
   se sorteaba campo por campo, así que un día con las 14 variables en blanco
   tenía probabilidad casi nula (el máximo observado eran 10). Pero ese es
   justo el estado que la interfaz consulta al abrir cada día, antes de que
   nadie registre nada — el modelo estaba extrapolando, no prediciendo.
   Corregido: no registrar es ahora un evento del día completo, como en la
   práctica (la familia no abrió la app).

El ROC AUC bajó de ~0.95 a ~0.73 al corregir el punto 1 — la cifra anterior
reflejaba en parte que el modelo explotaba un atajo de los datos, no
capacidad predictiva real. Tras corregir el punto 4 quedó en **0.754**
(niños no vistos en el entrenamiento).

## Revisión metodológica (agosto 2026)

Una segunda revisión externa señaló cinco puntos. Estos son los cambios de
código que se hicieron y los resultados que arrojaron:

1. **"Confianza" no era una confianza.** `completitud × factor_historial` mide
   cuánta información hay, no la probabilidad de acertar. Ahora son tres
   números separados en el motor y en la pantalla: riesgo, **suficiencia de
   información**, e **incertidumbre predictiva** (`core/uncertainty.py`).
   Sobre la cohorte, suficiencia y estabilidad correlacionan −0.76: son
   medidas distintas, no un renombre.
2. **El shrinkage no era "Bayes completo".** Correcto, y `core/bayesian.py`
   ahora lo declara así. Pero además: para una proporción la fórmula *es* la
   media posterior Beta-Binomial, y k se estima de los datos. Da **k ≈ 20.2**
   (el documento usaba 5). Se probaron los dos: AUROC 0.754 con k=5 y 0.756
   con k=20. **El valor de k no es el punto frágil de la formulación.**
3. **El evento objetivo ya tenía definición operacional en el código**
   (`crisis_hoy` = `nivel_regulacion_general_dia == "Desregulación Frecuente"`
   **o** un evento de intensidad ≥ 8), solo faltaba declararla. Sigue
   pendiente acordarla con Bluba.
4. **Los datos sintéticos no validan capacidad predictiva.** Declarado en el
   modelo (`bundle["alcance"]`), en la interfaz y en cada script de
   evaluación.
5. **La pregunta del día ahora descuenta la carga** y puede decidir no
   preguntar (ver `core/question_selector.py`).

### Lo que se encontró al medirlo

- **La afirmación MNAR era circular y costosa.** El generador asumía que se
  registra menos en los días difíciles, así que el modelo lo aprendía y el
  resultado parecía confirmarlo. Con `--missingness` se puede medir: bajo
  MNAR fuerte el AUROC sube a **0.925**, pero los indicadores de "nadie
  registró nada" se llevan el **36%** de la importancia del modelo — deja de
  leer al niño y pasa a leer si alguien abrió la app. Bajo MCAR, donde el
  silencio no informa, queda en **0.718**. Esa es la cifra honesta: el piso
  que no depende de un supuesto sin verificar.
- **La personalización no se paga en el promedio.** Quitarle a Antesala el
  bloque de línea base individual cambia el AUROC de 0.754 a 0.753. Pero el
  promedio está dominado por días normales de niños con 150 días de
  historial, donde el registro de hoy ya lo dice casi todo. En el segmento
  donde la personalización tiene que servir — **el día sin ningún registro**,
  que es el estado en que la app abre cada mañana — sí paga: AUPRC 0.622 →
  **0.681**.
- **1.6 falsas alertas por niño y por semana** al 80% de sensibilidad. Es un
  número incómodo y por eso se reporta: el umbral de alerta es una decisión
  clínica sobre cuánta carga tolera una familia, no una constante del modelo.

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

Para ver la evidencia detrás de las cifras de arriba:

```bash
python scripts/benchmark.py              # comparadores, ablaciones y barrido de k
python scripts/sensibilidad_ausencia.py  # MCAR / MAR / MNAR / mixto
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
