"""
Pooling bayesiano jerarquico (shrinkage) - Seccion 3.4 del documento tecnico.

    theta_i = w_i * ybar_i + (1 - w_i) * mu
    w_i = n_i / (n_i + k)

Este modulo implementa la version simplificada (Nivel 1, Seccion 3.7):
shrinkage manual con pandas/numpy, sin MCMC. Se aplica variable por variable,
no sobre un solo numero de riesgo.
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np

K_DEFAULT = 5  # dias de historial para empezar a confiar mas en el patron individual


@dataclass
class BaselineEstimate:
    variable: str
    child_id: str
    mu: float          # promedio poblacional
    ybar: float | None  # promedio individual (None si no hay historial)
    n: int              # dias de historial disponibles para esta variable
    w: float            # peso otorgado al dato individual
    theta: float         # estimacion final ajustada


def shrinkage_weight(n: int, k: int = K_DEFAULT) -> float:
    """w_i = n_i / (n_i + k) -- Seccion 3.4."""
    return n / (n + k)


def compute_baseline(
    logs: pd.DataFrame,
    child_id: str,
    variable: str,
    k: int = K_DEFAULT,
) -> BaselineEstimate:
    """Calcula theta_i para una variable numerica de un nino especifico,
    combinando su propio historial con el promedio poblacional.

    `logs` debe tener columnas: child_id, y la variable pedida (numerica).
    """
    col = logs[variable].dropna()
    mu = float(col.mean()) if len(col) else 0.0

    own = logs.loc[logs["child_id"] == child_id, variable].dropna()
    n = len(own)
    ybar = float(own.mean()) if n > 0 else None

    w = shrinkage_weight(n, k)
    theta = (w * ybar + (1 - w) * mu) if ybar is not None else mu

    return BaselineEstimate(variable=variable, child_id=child_id, mu=mu, ybar=ybar, n=n, w=w, theta=theta)


def compute_all_baselines(
    logs: pd.DataFrame,
    child_id: str,
    numeric_variables: list[str],
    k: int = K_DEFAULT,
) -> dict[str, BaselineEstimate]:
    """Calcula theta_i para varias variables de un mismo nino de una vez.
    Esto es lo que usa el motor de riesgo (Seccion 6.1) como linea base
    contra la que se compara el registro de hoy."""
    return {var: compute_baseline(logs, child_id, var, k) for var in numeric_variables}


def history_confidence_factor(n: int, k: int = K_DEFAULT) -> float:
    """factor_historial = n / (n + k) -- reutilizado en el calculo de
    confianza (Seccion 6.2). Es matematicamente la misma funcion que el
    peso de shrinkage, pero se nombra distinto porque su rol conceptual
    en el calculo de confianza es otro."""
    return shrinkage_weight(n, k)
