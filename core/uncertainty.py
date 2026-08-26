"""
Incertidumbre predictiva -- separada del riesgo y de la suficiencia de datos.

Motivacion (revision metodologica de agosto 2026). El prototipo mezclaba tres
cosas distintas bajo una sola palabra, "confianza":

  1. RIESGO            p(episodio de desregulacion en las proximas 24 h).
                       Lo estima el Random Forest.  -> core/risk_model.py
  2. SUFICIENCIA       cuanta de la informacion relevante esta disponible hoy
                       (completitud x historial). NO es la probabilidad de que
                       la prediccion sea correcta.  -> core/risk_model.py
  3. INCERTIDUMBRE     que tan estable es la prediccion. Este modulo.
     PREDICTIVA

Son independientes: se puede tener un registro completo (suficiencia alta) y
aun asi una prediccion inestable, porque el dia cae en una zona del espacio
donde los arboles no se ponen de acuerdo. Presentarlos juntos como un solo
numero era comodo para la demo pero no era honesto.

QUE ES Y QUE NO ES ESTA MEDIDA
La dispersion entre los arboles del ensamble es un PROXY COMPUTACIONAL de
inestabilidad, no una incertidumbre predictiva calibrada. Mide desacuerdo
entre modelos que comparten datos y sesgo, asi que subestima
sistematicamente la incertidumbre real (todos los arboles pueden coincidir y
estar igual de equivocados). Se usa para PRIORIZAR PREGUNTAS -- un uso
relativo, donde solo importa el orden entre variables -- y se muestra como
una banda cualitativa, nunca como un intervalo de confianza formal.

Una version validada requiere calibracion (isotonica/Platt), Brier score,
reliability diagrams y, para intervalos con garantia, metodos conformales.
La calibracion isotonica ya esta implementada en core/train_model.py y se
reporta en scripts/benchmark.py; lo conformal queda en el roadmap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# La dispersion normalizada por encima de la cual la prediccion se considera
# inestable. Calibrado sobre la distribucion observada en el dataset (ver
# scripts/benchmark.py, que imprime los percentiles reales).
UMBRAL_INESTABLE = 0.55
UMBRAL_ESTABLE = 0.35


def tree_probabilities(model: dict, row: pd.DataFrame) -> np.ndarray:
    """Probabilidad de desregulacion segun CADA arbol del ensamble."""
    pipeline = model["pipeline"]
    cols = model["feature_numeric"] + model["feature_categorical"]
    X_t = pipeline.named_steps["pre"].transform(row[cols])
    rf = pipeline.named_steps["rf"]
    return np.array([tree.predict_proba(X_t)[:, 1][0] for tree in rf.estimators_])


def dispersion_normalizada(preds: np.ndarray) -> float:
    """Desacuerdo entre arboles en escala 0-1, comparable entre predicciones.

    La desviacion estandar cruda NO es comparable: para una media p, la
    dispersion maxima posible de valores en [0,1] es sqrt(p*(1-p)), que vale 0
    en p=0 o p=1 y es maxima en p=0.5. Sin normalizar, toda prediccion extrema
    pareceria "muy estable" solo por ser extrema.

    Dividiendo por esa cota se obtiene "que fraccion del desacuerdo maximo
    posible hay realmente", que si se puede comparar entre dias y entre ninos.
    """
    if preds.size == 0:
        return 0.0
    p = float(preds.mean())
    maximo = np.sqrt(p * (1.0 - p))
    if maximo <= 1e-9:
        return 0.0
    return float(np.clip(preds.std() / maximo, 0.0, 1.0))


def evaluate(model: dict | None, row: pd.DataFrame | None) -> dict:
    """Incertidumbre predictiva del dia de hoy.

    Devuelve dict con:
      disponible   False si no hay ensamble (motor heuristico de respaldo)
      media        promedio de los arboles (= la prediccion del modelo)
      std          desviacion estandar cruda entre arboles
      dispersion   std normalizada 0-1 (ver dispersion_normalizada)
      p10, p90     rango intercuantil de los arboles, para dibujar la banda
      nivel        "estable" | "moderada" | "inestable"
      etiqueta     texto corto para la interfaz
    """
    vacio = {"disponible": False, "media": None, "std": None, "dispersion": None,
             "p10": None, "p90": None, "nivel": "desconocida",
             "etiqueta": "sin ensamble"}
    if model is None or row is None:
        return vacio
    try:
        preds = tree_probabilities(model, row)
    except Exception:
        return vacio
    if preds.size == 0:
        return vacio

    disp = dispersion_normalizada(preds)
    if disp >= UMBRAL_INESTABLE:
        nivel, etiqueta = "inestable", "predicción inestable"
    elif disp >= UMBRAL_ESTABLE:
        nivel, etiqueta = "moderada", "estabilidad moderada"
    else:
        nivel, etiqueta = "estable", "predicción estable"

    return {
        "disponible": True,
        "media": float(preds.mean()),
        "std": float(preds.std()),
        "dispersion": disp,
        "p10": float(np.percentile(preds, 10)),
        "p90": float(np.percentile(preds, 90)),
        "nivel": nivel,
        "etiqueta": etiqueta,
    }
