"""
Calibracion de probabilidades -- Seccion 9.3 del documento tecnico.

Un Random Forest promedia arboles, asi que sus probabilidades se comprimen
hacia el centro: casi nunca dice 5% ni 95%. Mientras la cifra se muestre como
porcentaje en pantalla ("riesgo 68%"), eso importa: el numero tiene que
significar lo que dice.

POR QUE PLATT Y NO ISOTONICA
La primera version usaba regresion isotonica. Funciona, pero es una funcion
ESCALONADA: sobre estos datos colapsa 2.732 probabilidades distintas en apenas
159 escalones. Eso rompe la explicabilidad local (core/explanation.py), que
necesita comparar el riesgo de hoy contra un contrafactual muy parecido -- si
los dos caen en el mismo escalon, la contribucion sale exactamente 0.000 y la
explicacion queda muda.

Se midieron las dos, fuera de muestra y con el calibrador sin ver las filas
que evalua:

    sin calibrar   ECE 0.0516   Brier 0.2115   2.732 valores distintos
    isotonica      ECE 0.0210   Brier 0.2101     159 valores distintos
    Platt          ECE 0.0157   Brier 0.2090   2.945 valores distintos

Platt gana en calibracion Y en Brier, ademas de conservar la resolucion. No es
un compromiso entre dos cosas: aqui es mejor en las dos.

Platt = regresion logistica sobre el LOG-ODDS de la probabilidad cruda. Se
ajusta sobre el log-odds y no sobre la probabilidad porque asi la correccion
es una transformacion afin en la escala natural del clasificador; sobre la
probabilidad directa, un mismo desplazamiento significaria cosas distintas
cerca de 0.5 y cerca de los extremos.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _log_odds(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


class PlattCalibrator:
    """Calibrador sigmoide. Expone .predict() para poder intercambiarse con
    IsotonicRegression sin tocar el codigo que lo consume.

    Vive en su propio modulo (y no dentro de train_model) porque joblib
    serializa las clases POR REFERENCIA: si la definicion estuviera en un
    script, cargar el modelo desde la app fallaria al no encontrarla.
    """

    def __init__(self, C: float = 1e6):
        # C muy alto = practicamente sin regularizacion. Son dos parametros
        # (pendiente e intercepto) ajustados sobre miles de puntos: regularizar
        # aqui solo sesgaria la correccion sin ganar nada.
        self.C = C
        self._lr = LogisticRegression(C=C, solver="lbfgs")

    def fit(self, proba, y) -> "PlattCalibrator":
        self._lr.fit(_log_odds(proba).reshape(-1, 1), np.asarray(y))
        return self

    def predict(self, proba) -> np.ndarray:
        return self._lr.predict_proba(_log_odds(proba).reshape(-1, 1))[:, 1]

    @property
    def pendiente(self) -> float:
        """>1 = el modelo era demasiado timido y la calibracion separa mas;
        <1 = era demasiado seguro y la calibracion lo acerca al centro."""
        return float(self._lr.coef_[0][0])

    @property
    def intercepto(self) -> float:
        return float(self._lr.intercept_[0])
