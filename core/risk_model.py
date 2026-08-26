"""
Motor de riesgo (nivel de respaldo) e INDICE DE SUFICIENCIA DE INFORMACION.
Seccion 6.1 (score heuristico ponderado) y Seccion 6.2 del documento tecnico.

Nomenclatura corregida tras la revision metodologica de agosto 2026: lo que
antes se llamaba "confianza" NO es la probabilidad de que la prediccion sea
correcta. Es cuanta de la informacion relevante esta disponible hoy
(completitud x historial). Un registro completo y mucho historial no implican
que el modelo este seguro. Por eso el sistema reporta TRES numeros separados:

    riesgo        p(episodio de desregulacion en 24 h)   -- este modulo
    suficiencia   cuanta informacion respalda esa cifra  -- este modulo
    incertidumbre que tan estable es la prediccion       -- core/uncertainty.py

En la interfaz la suficiencia se sigue condensando en alta/moderada/baja
porque es lo legible para una familia, pero el nombre y la definicion son los
de un indice de suficiencia, no los de una confianza estadistica.

Este es el "Nivel 1 - garantizado" del roadmap (Seccion 3.7): no requiere
entrenar ningun modelo, corre en milisegundos, y es completamente explicable.
El Random Forest / LightGBM (Nivel principal) se conecta despues sin cambiar
esta interfaz.
"""

from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np

from .bayesian import compute_all_baselines, history_factor, K_DEFAULT

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "antesala_rf.joblib"
_MODEL_CACHE: dict | None = None


def load_model() -> dict | None:
    """Carga (y cachea) el bundle del Random Forest entrenado. Si no existe el
    archivo o falta alguna dependencia, devuelve None y el motor cae al score
    heuristico (nivel de respaldo, Seccion 6.1)."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    try:
        import joblib
        if not MODEL_PATH.exists():
            return None
        _MODEL_CACHE = joblib.load(MODEL_PATH)
        return _MODEL_CACHE
    except Exception:
        return None

# Variables numericas usadas para la linea base bayesiana (Seccion 4.1 y 4.3).
# sueno_ord es la codificacion ordinal de calidad_sueno (Bluba no registra
# horas de sueno, solo su calidad); se calcula al vuelo antes de llamar a
# compute_all_baselines (ver score_heuristic).
NUMERIC_VARIABLES = ["sueno_ord", "n_eventos_desregulacion"]

# Pesos clinicos asignados por el equipo (Seccion 6.1) -- ejemplo de partida,
# ajustable una vez se calcule el feature importance del modelo de ML.
VARIABLE_WEIGHTS = {
    "sueno_ord": 3.0,
    "calidad_sueno": 3.0,   # alias: el driver/pregunta se reporta con el nombre del campo real
    "n_eventos_desregulacion": 2.5,
    "nivel_regulacion_general_dia": 3.0,
    "comportamiento_observado": 2.0,
    "estado_alerta": 1.8,
    "cambios_rutina": 1.8,
    "modo_despertar": 1.5,
    "estado_gastrointestinal": 1.5,
    "nivel_apoyo_requerido": 1.3,
    "interacciones_sociales": 1.2,
    "adherencia_medicacion": 1.0,
    "participacion_actividades": 1.0,
    "cambios_alimentacion": 0.8,
    "alimentacion_recreos": 0.8,
}

CATEGORICAL_RISK_VALUES = {
    "modo_despertar": {"Irritable/Llorando": 1.0, "Cansado/Con Sueno": 0.5, "Tranquilo/Alegre": 0.0},
    "estado_gastrointestinal": {"Diarrea": 1.0, "Estrenimiento": 0.4, "Normal": 0.0},
    "adherencia_medicacion": {"No": 1.0, "Si": 0.0, "No Aplica": 0.0},
    "nivel_regulacion_general_dia": {"Desregulacion Frecuente": 1.0, "Estable con Apoyo": 0.4, "Excelente": 0.0},
    "nivel_apoyo_requerido": {"Alto": 1.0, "Medio": 0.5, "Bajo": 0.0},
    "cambios_alimentacion": {"Selectividad aumentada": 1.0, "Menor apetito": 0.6, "Sin cambios": 0.0},
    "cambios_rutina": {"Si": 1.0, "No": 0.0},
    "comportamiento_observado": {"Desregulado": 1.0, "Inquieto": 0.5, "Estable": 0.0},
    # estado_alerta no es monotono: tanto hiper como hipo son senales de riesgo.
    "estado_alerta": {"Alto (Sobreexcitado)": 1.0, "Bajo (Letargico)": 0.8, "Optimo (Regulado)": 0.0},
    "participacion_actividades": {"No participa": 1.0, "Parcial": 0.5, "Completa": 0.0},
    "interacciones_sociales": {"Evitativa": 1.0, "Baja": 0.6, "Normal": 0.0},
    "alimentacion_recreos": {"Rechaza": 1.0, "Reducida": 0.5, "Normal": 0.0},
}

# Codificacion ordinal de calidad_sueno (debe coincidir con core/features.py).
_SLEEP_ORDINAL = {"Reparador": 2.0, "Interrumpido": 1.0, "Dificultad de Conciliacion": 0.0}


@dataclass
class RiskResult:
    child_id: str
    risk: float                     # 0-1: p(desregulacion en 24 h)
    sufficiency: float               # 0-1: indice de suficiencia de informacion
    sufficiency_level: str           # "baja" | "moderada" | "alta"
    drivers: list[str] = field(default_factory=list)
    missing_relevant: list[str] = field(default_factory=list)
    n_history_days: int = 0
    model_used: str = "heuristico"   # "random_forest" | "heuristico"
    base_rate_bayes: float | None = None  # linea base bayesiana de riesgo (theta_crisis_rate)
    suggested_question: str | None = None  # "la pregunta del dia" (Seccion 6.3)
    # "reduccion_varianza" | "heuristico" | "sin_pregunta_util".
    # "sin_pregunta_util" = hay datos faltantes, pero ninguno mejora la
    # estimacion de hoy, asi que el sistema decide NO preguntar nada.
    question_method: str | None = None
    # Reduccion esperada de varianza de CADA variable candidata, no solo de la
    # ganadora: el selector ya las calcula todas para elegir el argmax, asi que
    # exponerlas no cuesta nada y permite mostrar el ranking completo en la
    # interfaz (sin recalcularlo por segunda vez).
    question_scores: dict[str, float] = field(default_factory=dict)
    # Utilidad neta de cada pregunta candidata: ganancia informativa MENOS
    # carga de registro (Seccion 6.3 revisada). Es lo que se ordena de verdad
    # para elegir la pregunta; question_scores guarda solo la ganancia bruta.
    question_utilities: dict[str, float] = field(default_factory=dict)
    # Incertidumbre predictiva (core/uncertainty.py). Deliberadamente NO se
    # fusiona con `sufficiency`: son dos preguntas distintas.
    uncertainty: dict = field(default_factory=dict)

    # --- Alias retrocompatibles -------------------------------------------
    # El nombre "confidence" es justamente el que la revision metodologica
    # pidio no usar, pero hay codigo y tests que ya lo referencian. Se
    # mantiene como propiedad de solo lectura.
    @property
    def confidence(self) -> float:
        return self.sufficiency

    @property
    def confidence_level(self) -> str:
        return self.sufficiency_level


def _sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def score_heuristic(
    logs: pd.DataFrame,
    child_id: str,
    today: dict,
    k: int = K_DEFAULT,
    confidence_weights: dict[str, float] | None = None,
) -> RiskResult:
    """Calcula el score heuristico ponderado (Seccion 6.1, nivel de respaldo)
    comparando el registro de `today` contra la linea base ajustada (theta_i)
    de cada nino, y el indice de suficiencia de informacion (Seccion 6.2).

    Dos fuentes de peso DISTINTAS, tal como especifica el documento:
      - El SCORE de riesgo usa VARIABLE_WEIGHTS: pesos clinicos fijos,
        "asignados por el equipo" (Seccion 6.1).
      - La SUFICIENCIA usa `confidence_weights`, si se entrega: "la importancia
        relativa de la variable, obtenida del feature importance del modelo"
        (Seccion 6.2) -- viene del Random Forest (ver
        core/train_model.py::compute_confidence_weights). Si no hay modelo
        entrenado, cae a VARIABLE_WEIGHTS para esa variable.

    `today` es un dict con las mismas columnas que `logs` (mas "sueno_ord" si
    se quiere pasar directo), representando el registro (parcial o completo)
    del dia de hoy para ese nino.
    """
    logs2 = logs.copy()
    if "sueno_ord" not in logs2.columns and "calidad_sueno" in logs2.columns:
        logs2["sueno_ord"] = logs2["calidad_sueno"].map(_SLEEP_ORDINAL)
    today = dict(today)
    if "sueno_ord" not in today and today.get("calidad_sueno") is not None:
        today["sueno_ord"] = _SLEEP_ORDINAL.get(today["calidad_sueno"])

    baselines = compute_all_baselines(logs2, child_id, NUMERIC_VARIABLES, k)
    n_history = len(logs2.loc[logs2["child_id"] == child_id])

    def _conf_weight(name: str) -> float:
        if confidence_weights:
            w = confidence_weights.get(name)
            if w is not None and w > 0:
                return w
        return VARIABLE_WEIGHTS.get(name, 1.0)

    weighted_sum = 0.0
    total_weight = 0.0        # para el SCORE (pesos clinicos, Seccion 6.1)
    total_weight_conf = 0.0   # para la SUFICIENCIA (feature importance del RF, Sec. 6.2)
    present_weight_conf = 0.0
    drivers: list[tuple[str, float]] = []
    missing_relevant: list[str] = []

    # --- Variables numericas: desviacion respecto a theta_i ---
    for var in NUMERIC_VARIABLES:
        name = "calidad_sueno" if var == "sueno_ord" else var
        w = VARIABLE_WEIGHTS.get(var, 1.0)
        w_conf = _conf_weight(name)
        total_weight += w
        total_weight_conf += w_conf
        value = today.get(var)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            missing_relevant.append(name)
            continue
        present_weight_conf += w_conf
        theta = baselines[var].theta
        # normalizamos la desviacion: para sueno, menos calidad = mas riesgo;
        # para eventos de desregulacion, mas eventos = mas riesgo.
        if var == "sueno_ord":
            deviation = max(0.0, theta - value) / 1.0   # 1 nivel bajo la base = deviation 1.0
        else:
            deviation = min(1.0, value / 3.0)             # 3+ eventos = deviation 1.0
        weighted_sum += w * min(1.0, deviation)
        if deviation > 0.4:
            drivers.append((name, deviation))

    # --- Variables categoricas: mapa fijo a un valor de riesgo 0-1 ---
    for var, risk_map in CATEGORICAL_RISK_VALUES.items():
        w = VARIABLE_WEIGHTS.get(var, 1.0)
        w_conf = _conf_weight(var)
        total_weight += w
        total_weight_conf += w_conf
        value = today.get(var)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            missing_relevant.append(var)
            continue
        present_weight_conf += w_conf
        deviation = risk_map.get(value, 0.0)
        weighted_sum += w * deviation
        if deviation >= 0.8:
            drivers.append((var, deviation))

    raw_score = weighted_sum / total_weight if total_weight else 0.0
    risk = float(_sigmoid((raw_score - 0.35) * 6))  # centra y agudiza la curva

    # --- Indice de suficiencia de informacion (Seccion 6.2) ---
    # suficiencia = completitud x factor_historial. Es un indice de cuanta
    # informacion hay, NO una probabilidad de acierto: dos registros con la
    # misma suficiencia pueden tener incertidumbre predictiva muy distinta
    # (eso se mide aparte, en core/uncertainty.py).
    completeness = present_weight_conf / total_weight_conf if total_weight_conf else 0.0
    hist_factor = history_factor(n_history, k)
    sufficiency = completeness * hist_factor

    if sufficiency < 0.4:
        level = "baja"
    elif sufficiency < 0.7:
        level = "moderada"
    else:
        level = "alta"

    drivers.sort(key=lambda d: d[1], reverse=True)
    top_drivers = [d[0] for d in drivers[:3]]

    return RiskResult(
        child_id=child_id,
        risk=round(risk, 3),
        sufficiency=round(sufficiency, 3),
        sufficiency_level=level,
        drivers=top_drivers,
        missing_relevant=missing_relevant,
        n_history_days=n_history,
    )


def predict_risk(
    logs: pd.DataFrame,
    child_id: str,
    today: dict,
    k: int = K_DEFAULT,
    today_date: pd.Timestamp | None = None,
    compute_question: bool = True,
) -> RiskResult:
    """Motor de riesgo completo (Seccion 6.1). Usa el Random Forest entrenado
    (nivel principal) para el numero de riesgo; si no hay modelo, cae al score
    heuristico (nivel de respaldo). En ambos casos reutiliza la maquinaria
    interpretable del heuristico para la suficiencia (Seccion 6.2), los drivers y
    las variables faltantes, y expone la linea base bayesiana del nino.

    `today_date` permite recalcular el riesgo de un dia PASADO (para graficar
    una tendencia historica): `logs` debe entonces contener solo el historial
    ANTERIOR a esa fecha, para no filtrar informacion futura.

    `compute_question=False` omite el calculo de "la pregunta del dia"
    (Seccion 6.3): tiene un costo no trivial (evalua el Random Forest sobre
    varios escenarios simulados) y no tiene sentido para dias PASADOS -- usar
    False al recalcular una tendencia historica (ver app.py::historical_risk_trend).

    Esta es la funcion que consume la interfaz (app.py)."""
    model = load_model()

    # El heuristico aporta suficiencia, drivers, faltantes e historial (siempre
    # disponible). La suficiencia usa los pesos de completitud del RF (Seccion
    # 6.2) cuando hay modelo entrenado; si no, cae a los pesos clinicos.
    confidence_weights = model.get("confidence_weights") if model is not None else None
    result = score_heuristic(logs, child_id, today, k, confidence_weights=confidence_weights)

    feature_row = None
    if model is not None:
        try:
            from .features import build_features_for_today
            row = build_features_for_today(logs, child_id, today, k=k, mu=model["mu"], today_date=today_date)
            cols = model["feature_numeric"] + model["feature_categorical"]
            proba = float(model["pipeline"].predict_proba(row[cols])[0, 1])
            # Calibracion isotonica (Seccion 9.3 revisada): el RF crudo esta
            # sistematicamente descalibrado -- sus probabilidades se agolpan
            # lejos de 0 y 1 por promediar arboles. El calibrador se ajusta
            # fuera de muestra en core/train_model.py; si no existe (modelo
            # viejo), se usa la probabilidad cruda.
            calibrator = model.get("calibrator")
            if calibrator is not None:
                proba = float(calibrator.predict([proba])[0])
            result.risk = round(proba, 3)
            result.model_used = "random_forest"
            feature_row = row
            if "theta_crisis_rate" in row.columns:
                result.base_rate_bayes = round(float(row["theta_crisis_rate"].iloc[0]), 3)
        except Exception:
            # Ante cualquier problema, se conserva el riesgo heuristico ya calculado.
            result.model_used = "heuristico"

    # --- Incertidumbre predictiva (tercer numero, separado de la suficiencia) ---
    if feature_row is not None:
        from .uncertainty import evaluate as _evaluar_incertidumbre
        result.uncertainty = _evaluar_incertidumbre(model, feature_row)

    # Si no se pudo obtener del modelo, calculamos igual la linea base bayesiana.
    if result.base_rate_bayes is None:
        try:
            from .features import build_features_for_today, population_baselines
            mu = population_baselines(logs)
            row = build_features_for_today(logs, child_id, today, k=k, mu=mu, today_date=today_date)
            result.base_rate_bayes = round(float(row["theta_crisis_rate"].iloc[0]), 3)
        except Exception:
            result.base_rate_bayes = None

    # --- "La pregunta del dia" (Seccion 6.3) ---
    # Nivel principal: reduccion esperada de varianza del ensamble (requiere
    # el Random Forest). Si no esta disponible o falla, cae al proxy
    # heuristico (variable faltante de mayor peso clinico, siempre disponible).
    if compute_question and result.missing_relevant:
        chosen = None
        no_vale_la_pena = False
        if model is not None:
            try:
                from .question_selector import (
                    expected_variance_reductions, net_utilities,
                    vale_la_pena_preguntar)
                reductions = expected_variance_reductions(
                    logs, child_id, today, result.missing_relevant, model, k)
                if reductions:
                    result.question_scores = reductions
                    # La pregunta NO se elige por ganancia bruta, sino por
                    # utilidad neta = ganancia - lambda * carga de registro
                    # (Seccion 6.3 revisada): la variable mas informativa no
                    # siempre es la que vale la pena pedirle a la familia.
                    result.question_utilities = net_utilities(reductions)
                    if vale_la_pena_preguntar(reductions):
                        chosen = max(result.question_utilities,
                                     key=result.question_utilities.get)
                    else:
                        # Ningun dato faltante mejora la estimacion de hoy. No
                        # se pregunta nada: no preguntar tambien es una salida
                        # valida del selector, y es la coherente con el objetivo
                        # de reducir la carga de registro.
                        no_vale_la_pena = True
            except Exception:
                chosen = None
        if chosen:
            result.suggested_question = chosen
            result.question_method = "reduccion_varianza"
        elif no_vale_la_pena:
            result.suggested_question = None
            result.question_method = "sin_pregunta_util"
        else:
            result.suggested_question = suggest_question(result)
            result.question_method = "heuristico" if result.suggested_question else None

    return result


def suggest_question(result: RiskResult) -> str | None:
    """Version heuristica (v1) del selector de 'la pregunta del dia'
    (Seccion 6.3). Aqui se elige la variable faltante de mayor peso clinico;
    la version completa (reduccion de varianza del ensamble) se conecta en
    el Dia 3 del cronograma (Seccion 9), una vez entrenado el Random Forest."""
    if not result.missing_relevant:
        return None
    ranked = sorted(result.missing_relevant, key=lambda v: VARIABLE_WEIGHTS.get(v, 0), reverse=True)
    return ranked[0]
