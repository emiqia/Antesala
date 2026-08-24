"""
Motor de riesgo (nivel de respaldo) y calculo de confianza.
Seccion 6.1 (score heuristico ponderado) y Seccion 6.2 (nivel de confianza)
del documento tecnico.

Este es el "Nivel 1 - garantizado" del roadmap (Seccion 3.7): no requiere
entrenar ningun modelo, corre en milisegundos, y es completamente explicable.
El Random Forest / LightGBM (Nivel principal) se conecta despues sin cambiar
esta interfaz.
"""

from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from .bayesian import compute_all_baselines, history_confidence_factor, K_DEFAULT

# Variables numericas usadas para la linea base bayesiana (Seccion 4.1 y 4.3)
NUMERIC_VARIABLES = ["horas_sueno", "regulaciones_desregulaciones"]

# Pesos clinicos asignados por el equipo (Seccion 6.1) -- ejemplo de partida,
# ajustable una vez se calcule el feature importance del modelo de ML.
VARIABLE_WEIGHTS = {
    "horas_sueno": 3.0,
    "cambios_rutina": 2.5,
    "regulaciones_desregulaciones": 2.5,
    "estado_basal_despertar": 1.5,
    "salud_gastrointestinal": 1.5,
    "nivel_apoyo_requerido": 1.0,
    "estado_alerta": 1.0,
    "comportamiento_observado": 1.0,
}

CATEGORICAL_RISK_VALUES = {
    "cambios_rutina": {"si": 1.0, "no": 0.0},
    "estado_basal_despertar": {"irritable": 1.0, "neutro": 0.4, "tranquilo": 0.0},
    "salud_gastrointestinal": {"malestar": 1.0, "normal": 0.0},
    "nivel_apoyo_requerido": {"alto": 1.0, "medio": 0.5, "bajo": 0.0},
    "estado_alerta": {"hiperalerta": 1.0, "hipoalerta": 0.8, "normal": 0.0},
    "comportamiento_observado": {"desregulado": 1.0, "estable": 0.0},
}


@dataclass
class RiskResult:
    child_id: str
    risk: float                     # 0-1
    confidence: float                # 0-1
    confidence_level: str            # "baja" | "moderada" | "alta"
    drivers: list[str] = field(default_factory=list)
    missing_relevant: list[str] = field(default_factory=list)
    n_history_days: int = 0


def _sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def score_heuristic(logs: pd.DataFrame, child_id: str, today: dict, k: int = K_DEFAULT) -> RiskResult:
    """Calcula el score heuristico ponderado (Seccion 6.1, nivel de respaldo)
    comparando el registro de `today` contra la linea base ajustada (theta_i)
    de cada nino, y el nivel de confianza (Seccion 6.2).

    `today` es un dict con las mismas columnas que `logs`, representando el
    registro (parcial o completo) del dia de hoy para ese nino.
    """
    baselines = compute_all_baselines(logs, child_id, NUMERIC_VARIABLES, k)
    n_history = len(logs.loc[logs["child_id"] == child_id])

    weighted_sum = 0.0
    total_weight = 0.0
    present_weight = 0.0
    drivers: list[tuple[str, float]] = []
    missing_relevant: list[str] = []

    # --- Variables numericas: desviacion respecto a theta_i ---
    for var in NUMERIC_VARIABLES:
        w = VARIABLE_WEIGHTS.get(var, 1.0)
        total_weight += w
        value = today.get(var)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            missing_relevant.append(var)
            continue
        present_weight += w
        theta = baselines[var].theta
        # normalizamos la desviacion: para sueno, menos horas = mas riesgo;
        # para desregulaciones, mas eventos = mas riesgo.
        if var == "horas_sueno":
            deviation = max(0.0, theta - value) / 2.0   # 2h bajo la base = deviation 1.0
        else:
            deviation = min(1.0, value / 3.0)             # 3+ eventos = deviation 1.0
        weighted_sum += w * min(1.0, deviation)
        if deviation > 0.4:
            drivers.append((var, deviation))

    # --- Variables categoricas: mapa fijo a un valor de riesgo 0-1 ---
    for var, risk_map in CATEGORICAL_RISK_VALUES.items():
        w = VARIABLE_WEIGHTS.get(var, 1.0)
        total_weight += w
        value = today.get(var)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            missing_relevant.append(var)
            continue
        present_weight += w
        deviation = risk_map.get(value, 0.0)
        weighted_sum += w * deviation
        if deviation >= 0.8:
            drivers.append((var, deviation))

    raw_score = weighted_sum / total_weight if total_weight else 0.0
    risk = float(_sigmoid((raw_score - 0.35) * 6))  # centra y agudiza la curva

    # --- Confianza (Seccion 6.2) ---
    completeness = present_weight / total_weight if total_weight else 0.0
    history_factor = history_confidence_factor(n_history, k)
    confidence = completeness * history_factor

    if confidence < 0.4:
        level = "baja"
    elif confidence < 0.7:
        level = "moderada"
    else:
        level = "alta"

    drivers.sort(key=lambda d: d[1], reverse=True)
    top_drivers = [d[0] for d in drivers[:3]]

    return RiskResult(
        child_id=child_id,
        risk=round(risk, 3),
        confidence=round(confidence, 3),
        confidence_level=level,
        drivers=top_drivers,
        missing_relevant=missing_relevant,
        n_history_days=n_history,
    )


def suggest_question(result: RiskResult) -> str | None:
    """Version heuristica (v1) del selector de 'la pregunta del dia'
    (Seccion 6.3). Aqui se elige la variable faltante de mayor peso clinico;
    la version completa (reduccion de varianza del ensamble) se conecta en
    el Dia 3 del cronograma (Seccion 9), una vez entrenado el Random Forest."""
    if not result.missing_relevant:
        return None
    ranked = sorted(result.missing_relevant, key=lambda v: VARIABLE_WEIGHTS.get(v, 0), reverse=True)
    return ranked[0]
