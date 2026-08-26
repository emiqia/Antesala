"""
Partial pooling / shrinkage empirical-Bayes - Seccion 3.4 del documento tecnico.

    theta_i = w_i * ybar_i + (1 - w_i) * mu
    w_i = n_i / (n_i + k)

NOMENCLATURA (corregida tras la revision metodologica de agosto 2026).
Esto NO es un "modelo bayesiano jerarquico completamente estimado": no hay
MCMC, no hay distribuciones posteriores, no hay estimacion conjunta de
hiperparametros por verosimilitud marginal. Es partial pooling: la estimacion
individual se encoge hacia la referencia poblacional en proporcion a cuanta
evidencia propia existe. Reservamos "modelo jerarquico bayesiano" para la
version con PyMC (Nivel 2, Seccion 3.7).

Dicho eso, la formula NO es una heuristica inventada. Para una PROPORCION
(que es el caso de la tasa de desregulacion, la variable central del sistema)
es exactamente la media posterior de un modelo Beta-Binomial:

    prior:      theta ~ Beta(alpha, beta)   con  alpha = k*mu,  beta = k*(1-mu)
    verosim.:   y | theta ~ Binomial(n, theta)
    posterior:  theta | y ~ Beta(alpha + y, beta + n - y)

    E[theta | y] = (alpha + y) / (alpha + beta + n)
                 = (k*mu + n*ybar) / (k + n)
                 =  w*ybar + (1-w)*mu        con w = n/(n+k)

es decir: k es la CONCENTRACION del prior, medida en observaciones
equivalentes ("cuantos dias de evidencia vale nuestra creencia previa").
Como mu se estima de la propia poblacion y no se fija a priori, esto es
empirical Bayes por definicion.

Para una variable CONTINUA la misma formula es la media posterior del modelo
Normal-Normal jerarquico, con k = sigma^2 / tau^2 (ruido dentro del nino
sobre varianza entre ninos). Para variables ORDINALES (calidad de sueno,
nivel de apoyo, estado de alerta) la codificamos a una escala numerica y
aplicamos la version continua: es una aproximacion, no el modelo ordinal
correcto, y queda declarada como tal (ver Seccion 3.7 / roadmap).

SOBRE k: el documento usa k = 5 como valor de trabajo. Un valor elegido a
mano es el punto mas debil de la formulacion, asi que este modulo ademas lo
ESTIMA de los datos por el metodo de los momentos (`estimate_prior_strength`,
el estimador clasico de Kleinman para Beta-Binomial). K_DEFAULT se mantiene
porque es el valor del ejemplo numerico del documento y de los tests, pero el
k estimado es el que se reporta como justificacion empirica.
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np

K_DEFAULT = 5  # dias de historial equivalentes que "vale" el prior poblacional


@dataclass
class BaselineEstimate:
    variable: str
    child_id: str
    mu: float          # promedio poblacional (prior empirico)
    ybar: float | None  # promedio individual (None si no hay historial)
    n: int              # dias de historial disponibles para esta variable
    w: float            # peso otorgado al dato individual
    theta: float         # estimacion final encogida hacia mu


def shrinkage_weight(n: int, k: int = K_DEFAULT) -> float:
    """w_i = n_i / (n_i + k) -- Seccion 3.4."""
    return n / (n + k)


def estimate_prior_strength(
    logs: pd.DataFrame,
    variable: str,
    min_obs: int = 3,
) -> dict | None:
    """Estima k a partir de los datos, en vez de fijarlo a mano.

    Estimador de momentos para Beta-Binomial (Kleinman 1973). Idea: la
    varianza OBSERVADA entre los promedios individuales ybar_i tiene dos
    fuentes -- variacion real entre ninos, y ruido de muestreo por tener
    pocos dias. Descontando el ruido queda la varianza real entre ninos, y de
    ahi sale la concentracion del prior:

        Var(theta) = mu*(1-mu) / (k + 1)   ->   k = mu*(1-mu)/Var(theta) - 1

    Interpretacion directa: k grande = los ninos se parecen mucho entre si,
    conviene encoger fuerte hacia la poblacion. k chico = los ninos difieren
    mucho, conviene confiar antes en el dato propio.

    Devuelve None si no hay ninos suficientes o si la varianza descontada
    resulta no positiva (senal de que los datos no distinguen a los ninos, en
    cuyo caso el pooling total es lo correcto y k -> infinito).
    """
    if variable not in logs.columns:
        return None
    df = logs[["child_id", variable]].dropna()
    if df.empty:
        return None

    by_child = df.groupby("child_id")[variable].agg(["mean", "count"])
    by_child = by_child[by_child["count"] >= min_obs]
    if len(by_child) < 3:
        return None

    n_i = by_child["count"].to_numpy(dtype=float)
    y_i = by_child["mean"].to_numpy(dtype=float)

    # mu ponderado por observaciones (mas eficiente que el promedio simple).
    mu = float(np.average(y_i, weights=n_i))
    if not (0.0 < mu < 1.0):
        return None

    # Varianza observada entre promedios individuales, ponderada.
    var_observada = float(np.average((y_i - mu) ** 2, weights=n_i))
    # Ruido de muestreo esperado dentro de cada nino: mu(1-mu)/n_i.
    var_muestreo = float(np.average(mu * (1 - mu) / n_i, weights=n_i))
    var_entre = var_observada - var_muestreo

    if var_entre <= 0:
        return {"variable": variable, "mu": mu, "k": float("inf"),
                "var_entre_ninos": 0.0, "n_ninos": int(len(by_child)),
                "nota": "sin variacion real detectable entre ninos: pooling total"}

    k = mu * (1 - mu) / var_entre - 1.0
    return {"variable": variable, "mu": mu, "k": float(max(k, 0.0)),
            "var_entre_ninos": var_entre, "n_ninos": int(len(by_child)),
            "nota": "estimador de momentos Beta-Binomial"}


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


def history_factor(n: int, k: int = K_DEFAULT) -> float:
    """factor_historial = n / (n + k) -- componente del INDICE DE SUFICIENCIA
    DE INFORMACION (Seccion 6.2). Es matematicamente la misma funcion que el
    peso de shrinkage, pero su rol es otro: alli pondera cuanto creerle al
    dato propio; aqui mide cuanto historial respalda la estimacion.

    Importante (revision metodologica): esto NO es la probabilidad de que la
    prediccion sea correcta. Es cuanta informacion hay disponible. La
    incertidumbre predictiva se mide aparte, en core/uncertainty.py."""
    return shrinkage_weight(n, k)


# Alias retrocompatible: el nombre anterior sugeria que el resultado era una
# "confianza" estadistica, que es justamente lo que la revision pidio no
# afirmar. Se mantiene para no romper importaciones existentes.
history_confidence_factor = history_factor
