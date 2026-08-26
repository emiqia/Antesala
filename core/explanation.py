"""
Explicabilidad local -- Seccion 11 del documento tecnico.

    "La explicacion al usuario no se basara simplemente en feature importance
     global. Para cada prediccion se identificara que factores observados
     contribuyeron a alejar el estado actual de la linea base."

POR QUE NO SIRVE EL FEATURE IMPORTANCE GLOBAL
El feature importance del Random Forest dice que variables importan EN
PROMEDIO, sobre todo el dataset y todos los ninos. No dice nada sobre el dia
de hoy de ESTE nino. Puede ocurrir -- y ocurre -- que la variable mas
importante del modelo hoy este en su valor habitual y no explique nada,
mientras que una variable de importancia media este completamente fuera de
rango y sea la unica razon de la cifra en pantalla. Mostrar el ranking global
como si fuera la explicacion de la prediccion actual es, lisa y llanamente,
mostrar otra cosa.

QUE SE CALCULA AQUI
Para cada variable REGISTRADA hoy, la contribucion es la diferencia entre el
riesgo estimado con el valor real y el riesgo que habria si esa variable
estuviera en el valor habitual del propio nino:

    contribucion_j = riesgo(hoy) - riesgo(hoy con la variable j en su valor habitual)

Es una atribucion contrafactual contra la linea base individual, que es
literalmente lo que pide la Seccion 11. Positiva = ese dato empujo el riesgo
hacia arriba respecto de lo normal en este nino. Negativa = lo empujo hacia
abajo (factor protector: hoy esa variable esta MEJOR que de costumbre).

El "valor habitual" sale del historial propio del nino cuando tiene
suficientes observaciones, y de la poblacion cuando no (arranque en frio) --
el mismo criterio de partial pooling que usa el resto del sistema, aplicado
aqui a "que es normal para este nino".

LIMITES, DECLARADOS
  - Es una atribucion de UNA variable a la vez. Si dos variables solo importan
    juntas, este metodo reparte mal ese efecto. Un metodo aditivo con garantia
    de reparto (Shapley/SHAP) lo haria mejor, y queda en el roadmap; no se usa
    aqui para no agregar una dependencia pesada al MVP.
  - Las contribuciones NO suman exactamente el riesgo total, por lo mismo.
    Nunca se presentan como una descomposicion exacta.
  - Describe lo que el MODELO usa, no una causa. La Seccion 11 es explicita en
    esto: nada de "la crisis ocurrira porque durmio mal".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bayesian import K_DEFAULT
from .features import build_features_for_today
from .question_selector import ASKABLE_FIELDS, MIN_CHILD_OBSERVATIONS

# Bajo esta diferencia (en puntos de riesgo) una contribucion se considera
# ruido del ensamble y no se muestra: con 400 arboles, mover una variable
# siempre cambia el tercer decimal aunque no signifique nada.
UMBRAL_RELEVANTE = 0.01


@dataclass
class Contribution:
    campo: str
    valor: object            # lo que se registro hoy
    valor_habitual: object   # lo normal para este nino (o la poblacion)
    contribucion: float      # puntos de riesgo (positivo = empuja hacia arriba)
    es_habitual: bool        # True si hoy coincide con su valor habitual

    @property
    def direccion(self) -> str:
        if self.es_habitual or abs(self.contribucion) < UMBRAL_RELEVANTE:
            return "neutro"
        return "riesgo" if self.contribucion > 0 else "protector"


def valor_habitual(logs: pd.DataFrame, child_id: str, field: str):
    """El valor mas frecuente de `field` para este nino; poblacional si tiene
    poco historial propio (mismo criterio de partial pooling que el resto del
    sistema, Seccion 3.6)."""
    propios = logs.loc[logs["child_id"] == child_id, field].dropna()
    fuente = propios if len(propios) >= MIN_CHILD_OBSERVATIONS else logs[field].dropna()
    if len(fuente) == 0:
        return None
    if pd.api.types.is_numeric_dtype(fuente):
        return float(fuente.median())
    modo = fuente.mode()
    return modo.iloc[0] if len(modo) else None


def _sustitucion_habitual(logs: pd.DataFrame, child_id: str, field: str) -> dict | None:
    """Los cambios que hay que aplicar a `today` para poner `field` en su valor
    habitual. n_eventos_desregulacion es compuesto: arrastra intensidad, tipo y
    resultado, igual que en el selector de preguntas."""
    if field == "n_eventos_desregulacion":
        propios = logs[logs["child_id"] == child_id]
        fuente = propios if len(propios) >= MIN_CHILD_OBSERVATIONS else logs
        n_ev = pd.to_numeric(fuente["n_eventos_desregulacion"], errors="coerce").dropna()
        if len(n_ev) == 0:
            return None
        habitual = float(n_ev.median())
        if habitual <= 0:
            return {"n_eventos_desregulacion": 0,
                    "intensidad_max_desregulacion": np.nan,
                    "intensidad_sum_desregulacion": 0.0,
                    "tipo_evento_principal": None,
                    "resultado_estrategia_principal": None}
        con_evento = fuente[pd.to_numeric(
            fuente["n_eventos_desregulacion"], errors="coerce") > 0]
        inten = pd.to_numeric(
            con_evento["intensidad_max_desregulacion"], errors="coerce").dropna()
        intensidad = float(inten.median()) if len(inten) else 5.0
        return {"n_eventos_desregulacion": habitual,
                "intensidad_max_desregulacion": intensidad,
                "intensidad_sum_desregulacion": intensidad * habitual,
                "tipo_evento_principal": valor_habitual(logs, child_id, "tipo_evento_principal"),
                "resultado_estrategia_principal": valor_habitual(
                    logs, child_id, "resultado_estrategia_principal")}

    hab = valor_habitual(logs, child_id, field)
    if hab is None:
        return None
    return {field: hab}


def explicar(
    logs: pd.DataFrame,
    child_id: str,
    today: dict,
    model: dict | None,
    k: int = K_DEFAULT,
) -> list[Contribution]:
    """Contribucion de cada variable REGISTRADA hoy, ordenada por magnitud.

    Todos los escenarios contrafactuales se construyen primero y se evaluan en
    UNA sola pasada por el pipeline, igual que en question_selector: son 14
    predicciones, no 14 recorridos completos de 400 arboles.

    Devuelve [] si no hay modelo o no hay nada registrado -- en ese caso la
    interfaz cae a las senales del narrador, que no necesitan el ensamble.
    """
    if model is None:
        return []
    registrados = [f for f in ASKABLE_FIELDS
                   if f in today and today[f] is not None
                   and not (isinstance(today[f], float) and pd.isna(today[f]))]
    if not registrados:
        return []

    child_hist = logs[logs["child_id"] == child_id]
    cols = model["feature_numeric"] + model["feature_categorical"]

    filas = [build_features_for_today(child_hist, child_id, today, k=k, mu=model["mu"])]
    campos, habituales = [], []
    for field in registrados:
        cambios = _sustitucion_habitual(logs, child_id, field)
        if cambios is None:
            continue
        hipotetico = dict(today)
        hipotetico.update(cambios)
        filas.append(build_features_for_today(
            child_hist, child_id, hipotetico, k=k, mu=model["mu"]))
        campos.append(field)
        habituales.append(cambios.get(field))

    if not campos:
        return []

    todas = pd.concat(filas, ignore_index=True)
    proba = model["pipeline"].predict_proba(todas[cols])[:, 1]
    # Se calibra ANTES de restar: las contribuciones tienen que estar en la
    # misma escala que el porcentaje que se muestra en pantalla, o no se pueden
    # leer como "puntos de riesgo".
    calibrador = model.get("calibrator")
    if calibrador is not None:
        proba = calibrador.predict(proba)

    base = float(proba[0])
    salida = []
    for i, field in enumerate(campos, start=1):
        hab = habituales[i - 1]
        actual = today.get(field)
        es_habitual = _mismo_valor(actual, hab)
        salida.append(Contribution(
            campo=field, valor=actual, valor_habitual=hab,
            contribucion=base - float(proba[i]),
            es_habitual=es_habitual))

    salida.sort(key=lambda c: abs(c.contribucion), reverse=True)
    return salida


def _mismo_valor(a, b) -> bool:
    if a is None or b is None:
        return False
    try:
        return bool(np.isclose(float(a), float(b)))
    except (TypeError, ValueError):
        return str(a) == str(b)


def resumen_texto(contribs: list[Contribution], etiqueta, max_items: int = 3) -> str:
    """Frase corta para la interfaz. Solo menciona lo que se aparta de la linea
    base: decir "duerme como siempre" no explica una prediccion."""
    relevantes = [c for c in contribs if c.direccion != "neutro"][:max_items]
    if not relevantes:
        return ("Ningún dato de hoy se aparta del patrón habitual de este niño: "
                "la estimación descansa en su línea base, no en algo puntual de hoy.")
    partes = []
    for c in relevantes:
        signo = "+" if c.contribucion > 0 else "−"
        partes.append(f"{etiqueta(c.campo)} ({signo}{abs(c.contribucion):.0%})")
    return "Respecto de lo habitual en este niño: " + ", ".join(partes) + "."
