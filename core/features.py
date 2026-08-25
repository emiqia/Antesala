"""
Ingenieria de variables (feature engineering) - Seccion 4.3 del documento tecnico.

Trabaja sobre el ESQUEMA REAL de campos de Bluba (ver
data/generate_synthetic_data.py y docs/Datos base/*.csv), no sobre nombres
inventados: bitacora diaria del tutor (calidad_sueno, modo_despertar,
adherencia_medicacion, estado_gastrointestinal, nivel_regulacion_general_dia)
mas el agregado diario de eventos de desregulacion (n_eventos_desregulacion,
intensidad_max_desregulacion, intensidad_sum_desregulacion, tipo_evento_principal,
resultado_estrategia_principal).

Construye las variables derivadas de la Seccion 4.3:
  - Promedio movil de sueno (3 y 7 dias) -- usando una codificacion ordinal
    de calidad_sueno (Bluba no registra horas de sueno, solo su calidad).
  - Desviacion respecto a la linea base individual theta_i (Seccion 3, via
    core/bayesian.py) para sueno (ordinal) y para eventos de desregulacion.
  - Dias desde la ultima crisis registrada.
  - Conteo de eventos de desregulacion en ventanas de 1, 3 y 7 dias.
  - Indicador binario de transicion reciente (3 dias).
  - Antiguedad del ultimo registro por variable.
  - Antiguedad del registro del nino (dias de historial -> cold start, Seccion 3.6).
  - Dia de la semana / dia escolar vs. no escolar.

Los indicadores de dato faltante (Seccion 4.4) NO se generan aqui: los produce
el SimpleImputer(add_indicator=True) del pipeline de entrenamiento
(core/train_model.py), como indica el stack de la Seccion 8.1.

Punto critico: NO hay fuga temporal (data leakage). Toda variable derivada de
un dia d usa unicamente informacion disponible al final del dia d:
  - Las ventanas moviles miran hacia atras (incluyen d, nunca d+1).
  - La linea base theta_i de un dia d se estima con el historial ESTRICTAMENTE
    anterior a d (dias < d), de modo que la "desviacion de hoy" compara el dato
    de hoy contra lo que ya se sabia del nino antes de hoy.
La variable objetivo crisis_24h (Seccion 4.2) ya viene desplazada +1 dia en el
generador de datos, y nunca se usa como insumo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bayesian import shrinkage_weight, K_DEFAULT

# --- Codificacion ordinal de calidad_sueno (mejor -> peor) --------------------
# Bluba no registra horas de sueno, solo su calidad categorica. Para poder
# calcular promedios moviles y una linea base bayesiana numerica (Seccion 4.3),
# se codifica como ordinal: Reparador=2, Interrumpido=1, Dificultad=0.
_SLEEP_ORDINAL = {"Reparador": 2.0, "Interrumpido": 1.0, "Dificultad de Conciliacion": 0.0}


def _sleep_ordinal(series: pd.Series) -> pd.Series:
    return series.map(_SLEEP_ORDINAL)


# --- Variables originales de Bluba que entran directo al modelo (Seccion 4.1) -
# Numericas conocidas al final del dia d (agregado diario de eventos).
RAW_NUMERIC = ["n_eventos_desregulacion", "intensidad_max_desregulacion", "intensidad_sum_desregulacion"]

# Categoricas (el pipeline las imputa y codifica; aqui se dejan como texto/NaN).
# Las primeras 5 + los campos de evento vienen de la bitacora real de Bluba
# (docs/Datos base/*.csv). Las siguientes 8 completan el listado de variables
# de la Seccion 4.1 del documento tecnico que no estaban en la muestra de
# datos reales disponible (esa muestra solo cubria la pantalla "Mi Dia" del
# tutor); se modelan segun la descripcion de las bases, y estado_alerta
# reutiliza las categorias reales observadas en las sesiones profesionales
# (docs/Datos base/3_sesiones_profesionales.csv).
RAW_CATEGORICAL = [
    "calidad_sueno",
    "modo_despertar",
    "adherencia_medicacion",
    "estado_gastrointestinal",
    "nivel_regulacion_general_dia",
    "tipo_evento_principal",
    "resultado_estrategia_principal",
    "fuente_registro",
    "nivel_apoyo_requerido",
    "cambios_alimentacion",
    "cambios_rutina",
    "comportamiento_observado",
    "estado_alerta",
    "participacion_actividades",
    "interacciones_sociales",
    "alimentacion_recreos",
    # Observaciones terapeuticas/escolares (bases oficiales: la informacion
    # viene de "la bitacora diaria Y observaciones terapeuticas/escolares",
    # no solo de la bitacora del tutor). Esporadicas, no diarias -- por eso
    # son mayormente NaN salvo los dias que hubo sesion. NO son "preguntables"
    # a la familia (ver core/question_selector.py::ASKABLE_FIELDS): son datos
    # del lado profesional de la plataforma, no del check-in diario del tutor.
    "nivel_alerta_sesion",
    "profesion_sesion",
]

# Variables con linea base bayesiana theta_i (Seccion 3.6): sueno (ordinal) y
# carga de eventos de desregulacion del dia.
BASELINE_VARS = ["sueno_ord", "n_eventos_desregulacion"]

# Clave del mu poblacional de la tasa de crisis (linea base de riesgo, Seccion 3.4).
CRISIS_RATE_KEY = "crisis_rate"

# Campos para los que se calcula antiguedad de registro (Seccion 4.4: una de
# las CUATRO piezas de informacion que se conservan por variable -- valor
# imputado, indicador de ausencia, antiguedad, fuente. Se excluyen
# tipo_evento_principal/resultado_estrategia_principal, que dependen de la
# MISMA accion de registro que n_eventos_desregulacion, profesion_sesion que
# depende de la MISMA accion que nivel_alerta_sesion, y fuente_registro, que
# nunca es nula). nivel_alerta_sesion no es "preguntable" a la familia (ver
# ASKABLE_FIELDS en core/question_selector.py) pero su antiguedad ("hace
# cuantos dias fue la ultima sesion con dato de alerta") es igual de valida
# como senal para el modelo -- reutiliza el mismo mecanismo generico.
ANTIQUITY_FIELDS = [
    "calidad_sueno", "modo_despertar", "adherencia_medicacion",
    "estado_gastrointestinal", "nivel_regulacion_general_dia",
    "nivel_apoyo_requerido", "cambios_alimentacion", "cambios_rutina",
    "comportamiento_observado", "estado_alerta", "participacion_actividades",
    "interacciones_sociales", "alimentacion_recreos",
    "n_eventos_desregulacion", "nivel_alerta_sesion",
]

# --- Variables derivadas que produce este modulo (Seccion 4.3) ---
DERIVED_NUMERIC = [
    "sueno_ord",                 # codificacion ordinal de calidad_sueno
    "crisis_hoy_num",            # estado de crisis de hoy (0/1), conocido al dia d
    "theta_crisis_rate",         # linea base bayesiana de riesgo del nino (Seccion 3.4)
    "dias_historial",            # antiguedad del registro del nino (cold start)
    "sueno_ma3", "sueno_ma7",    # promedio movil de sueno (ordinal)
    "theta_sueno_ord", "desviacion_sueno_ord",
    "theta_n_eventos_desregulacion", "desviacion_n_eventos_desregulacion",
    "desreg_sum3", "desreg_sum7",   # conteo de eventos de desregulacion en ventanas 3/7d
    "dias_desde_ultima_crisis",
    "transicion_reciente_3d",
    "cambio_rutina_reciente_3d",  # indicador binario de cambio de rutina reciente (Seccion 4.3)
] + [f"antiguedad_{f}" for f in ANTIQUITY_FIELDS] + [
    "dia_semana",
    "es_dia_escolar",
]

# Columnas de entrada al modelo (orden estable). Las categoricas van al final.
FEATURE_NUMERIC = RAW_NUMERIC + DERIVED_NUMERIC
FEATURE_CATEGORICAL = RAW_CATEGORICAL
FEATURE_COLUMNS = FEATURE_NUMERIC + FEATURE_CATEGORICAL

TARGET = "crisis_24h"
ID_COLUMNS = ["child_id", "date"]

# Sentinela para "nunca ocurrio / nunca se registro" en variables de antiguedad.
# Se usa un numero alto y finito (mejor que NaN para los arboles: es un valor
# ordenable que significa "hace mucho / nunca").
_NEVER = 999


def population_means(logs: pd.DataFrame, variables: list[str] = BASELINE_VARS) -> dict[str, float]:
    """mu de cada variable: promedio poblacional sobre todos los ninos-dias
    (misma definicion que core/bayesian.py).

    mu es un ARTEFACTO DEL ENTRENAMIENTO: se calcula una vez sobre los datos de
    entrenamiento y debe persistirse junto al modelo, para que la inferencia use
    exactamente el mismo mu (y no uno recalculado sobre un subconjunto distinto
    de datos). build_features_for_today acepta este mu ya calculado."""
    df = logs.copy()
    if "sueno_ord" not in df.columns and "calidad_sueno" in df.columns:
        df["sueno_ord"] = _sleep_ordinal(df["calidad_sueno"])
    mu = {}
    for var in variables:
        col = pd.to_numeric(df[var], errors="coerce").dropna()
        mu[var] = float(col.mean()) if len(col) else 0.0
    return mu


# Alias interno para compatibilidad con llamadas previas.
_population_means = population_means


def population_baselines(logs: pd.DataFrame) -> dict[str, float]:
    """mu poblacional de todas las variables con linea base bayesiana, incluida
    la TASA DE CRISIS (Seccion 3.4): proporcion de dias con crisis sobre todos
    los ninos-dias. Es el artefacto que se persiste junto al modelo."""
    mu = population_means(logs, BASELINE_VARS)
    crisis = _coerce_bool(logs["crisis_hoy"]).dropna() if "crisis_hoy" in logs else pd.Series(dtype=float)
    mu[CRISIS_RATE_KEY] = float(crisis.mean()) if len(crisis) else 0.0
    return mu


def _expanding_theta(values: np.ndarray, mu: float, k: int) -> np.ndarray:
    """theta_i por dia, estimado con el historial ESTRICTAMENTE anterior a cada
    dia (sin fuga). Reproduce la formula de shrinkage de la Seccion 3.4:

        theta_d = w * ybar_{<d} + (1 - w) * mu ,   w = n_{<d} / (n_{<d} + k)

    donde ybar_{<d} y n_{<d} usan solo los dias previos con dato observado.
    En el primer dia (n=0), w=0 y theta = mu.
    """
    val = pd.Series(values, dtype="float64")
    notnull = val.notna().astype(int)
    n_past = notnull.cumsum().shift(1).fillna(0).to_numpy()
    sum_past = val.fillna(0.0).cumsum().shift(1).fillna(0.0).to_numpy()

    with np.errstate(invalid="ignore", divide="ignore"):
        ybar_past = np.where(n_past > 0, sum_past / np.where(n_past == 0, 1, n_past), mu)
    w = n_past / (n_past + k)
    return w * ybar_past + (1.0 - w) * mu


def _days_since_last_true(mask: np.ndarray) -> np.ndarray:
    """Dias desde el ultimo True (incluyendo el dia actual: si hoy es True -> 0).
    Si aun no ocurrio ningun True, devuelve el sentinela _NEVER."""
    out = np.empty(len(mask), dtype=float)
    last = -1
    for i, flag in enumerate(mask):
        if flag:
            last = i
        out[i] = (i - last) if last >= 0 else _NEVER
    return out


def _antiquity(notnull: np.ndarray) -> np.ndarray:
    """Antiguedad del ultimo registro de una variable: dias desde la ultima
    observacion no nula (incluyendo hoy: si hoy hay dato -> 0). Si nunca se
    registro, devuelve el sentinela _NEVER."""
    out = np.empty(len(notnull), dtype=float)
    last = -1
    for i, flag in enumerate(notnull):
        if flag:
            last = i
        out[i] = (i - last) if last >= 0 else _NEVER
    return out


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Una columna booleana (crisis_hoy) llega como bool/str/NaN; se normaliza
    a 0/1/NaN."""
    def to_num(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, str):
            return 1.0 if v.strip().lower() == "true" else 0.0
        return 1.0 if bool(v) else 0.0
    return series.map(to_num)


def engineer_child(
    child_logs: pd.DataFrame,
    mu: dict[str, float],
    k: int = K_DEFAULT,
) -> pd.DataFrame:
    """Calcula todas las variables derivadas para UN nino, sobre sus filas
    ordenadas cronologicamente. Devuelve el DataFrame con las columnas
    originales + las derivadas. Es la unica implementacion de la logica temporal;
    tanto el entrenamiento (build_features) como la inferencia
    (build_features_for_today) la reutilizan, garantizando paridad."""
    g = child_logs.sort_values("date").reset_index(drop=True).copy()
    n = len(g)

    # Aseguramos numericas base como float (por si vienen como object).
    for var in RAW_NUMERIC:
        g[var] = pd.to_numeric(g[var], errors="coerce")

    g["sueno_ord"] = _sleep_ordinal(g["calidad_sueno"]) if "calidad_sueno" in g else np.nan
    g["crisis_hoy_num"] = _coerce_bool(g["crisis_hoy"]) if "crisis_hoy" in g else np.nan

    # Linea base bayesiana de riesgo del nino (Seccion 3.4): pooling de su tasa
    # de crisis propia hacia la tasa poblacional. Es el "porcentaje base" del
    # ejemplo Nino A / Nino B del documento. Expanding (solo dias < d, sin fuga).
    mu_crisis = mu.get(CRISIS_RATE_KEY, 0.0)
    g["theta_crisis_rate"] = _expanding_theta(
        g["crisis_hoy_num"].to_numpy(dtype="float64"), mu_crisis, k)

    # Antiguedad del registro del nino (dias de historial previos): cold start.
    g["dias_historial"] = np.arange(n, dtype=float)

    # Promedios moviles de sueno (miran hacia atras, incluyen hoy).
    g["sueno_ma3"] = g["sueno_ord"].rolling(3, min_periods=1).mean()
    g["sueno_ma7"] = g["sueno_ord"].rolling(7, min_periods=1).mean()

    # Linea base bayesiana theta_i y desviacion de hoy respecto a ella.
    for var in BASELINE_VARS:
        theta = _expanding_theta(g[var].to_numpy(dtype="float64"), mu[var], k)
        g[f"theta_{var}"] = theta
        g[f"desviacion_{var}"] = g[var].to_numpy(dtype="float64") - theta

    # Conteo de eventos de desregulacion en ventanas de 3 y 7 dias (la ventana
    # de 1 dia es la propia variable n_eventos_desregulacion).
    g["desreg_sum3"] = g["n_eventos_desregulacion"].rolling(3, min_periods=1).sum()
    g["desreg_sum7"] = g["n_eventos_desregulacion"].rolling(7, min_periods=1).sum()

    # Dias desde la ultima crisis registrada (usa crisis_hoy, presente/pasado).
    crisis_mask = (g["crisis_hoy_num"] == 1.0).to_numpy()
    g["dias_desde_ultima_crisis"] = _days_since_last_true(crisis_mask)

    # Indicador de transicion reciente (algun evento tipo "Transicion de
    # Actividad" en los ultimos 3 dias).
    transicion = (g["tipo_evento_principal"] == "Transicion de Actividad").astype(int)
    g["transicion_reciente_3d"] = transicion.rolling(3, min_periods=1).max()

    # Indicador de cambio de rutina reciente (Seccion 4.3): algun "Si" en 3 dias.
    if "cambios_rutina" in g.columns:
        rutina_si = (g["cambios_rutina"] == "Si").astype(int)
        g["cambio_rutina_reciente_3d"] = rutina_si.rolling(3, min_periods=1).max()
    else:
        g["cambio_rutina_reciente_3d"] = 0.0

    # Antiguedad del ultimo registro, POR VARIABLE (Seccion 4.4: una de las
    # cuatro piezas de informacion que se conservan explicitamente en vez de
    # imputar en silencio -- valor imputado, indicador de ausencia,
    # antiguedad y fuente).
    for field in ANTIQUITY_FIELDS:
        g[f"antiguedad_{field}"] = _antiquity(g[field].notna().to_numpy())

    # Dia de la semana / dia escolar.
    dow = pd.to_datetime(g["date"]).dt.dayofweek
    g["dia_semana"] = dow.astype(float)
    g["es_dia_escolar"] = (dow < 5).astype(float)

    return g


def build_features(
    logs: pd.DataFrame,
    k: int = K_DEFAULT,
    mu: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Construye la matriz de variables para TODO el dataset (entrenamiento).

    Devuelve un DataFrame con ID_COLUMNS + FEATURE_COLUMNS + TARGET, una fila
    por nino-dia. El mu poblacional se calcula una vez sobre todos los datos y
    se reutiliza para cada nino (consistente con la inferencia). Si se pasa `mu`,
    se usa ese (util para reproducir exactamente el mu de entrenamiento)."""
    if mu is None:
        mu = population_baselines(logs)
    parts = [engineer_child(g, mu, k) for _, g in logs.groupby("child_id", sort=False)]
    feat = pd.concat(parts, ignore_index=True)

    keep = ID_COLUMNS + FEATURE_COLUMNS + ([TARGET] if TARGET in feat.columns else [])
    return feat[keep]


def build_features_for_today(
    history_logs: pd.DataFrame,
    child_id: str,
    today: dict,
    k: int = K_DEFAULT,
    today_date: pd.Timestamp | None = None,
    mu: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Construye la fila de variables del registro de HOY para un nino, usando
    su historial. Garantiza paridad con el entrenamiento: agrega `today` como
    una fila nueva al final del historial del nino y corre exactamente la misma
    ingenieria (engineer_child), devolviendo solo la ultima fila.

    - history_logs: bitacoras crudas del nino (y opcionalmente de otros).
    - today: dict con las columnas registradas hoy (parciales o completas).
    - mu: referencia poblacional del ENTRENAMIENTO. Si es None se calcula de
      history_logs; para paridad exacta con el modelo, pasar el mu persistido.
    """
    if mu is None:
        mu = population_baselines(history_logs)
    child_hist = history_logs[history_logs["child_id"] == child_id].copy()

    if today_date is None:
        if len(child_hist):
            today_date = pd.to_datetime(child_hist["date"]).max() + pd.Timedelta(days=1)
        else:
            today_date = pd.Timestamp.today().normalize()

    new_row = {col: np.nan for col in history_logs.columns}
    new_row.update({k2: v for k2, v in today.items() if k2 in history_logs.columns})
    new_row["child_id"] = child_id
    new_row["date"] = today_date

    child_frame = pd.concat([child_hist, pd.DataFrame([new_row])], ignore_index=True)
    engineered = engineer_child(child_frame, mu, k)

    last = engineered.iloc[[-1]].reset_index(drop=True)
    keep = ID_COLUMNS + FEATURE_COLUMNS
    keep = [c for c in keep if c in last.columns]
    return last[keep]
