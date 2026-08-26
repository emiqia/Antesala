"""
"La pregunta del dia": ADQUISICION ACTIVA DE INFORMACION -- Seccion 6.3.

Un formulario tradicional pregunta lo mismo todos los dias. Antesala pregunta:
de lo que todavia no sabemos hoy, cual es el dato que mas ayuda a decidir,
descontando lo que cuesta pedirlo.

NOMENCLATURA (revision metodologica, agosto 2026). Antes esto se presentaba
como "Expected Value of Information". Estrictamente no lo es: el valor de
informacion se define respecto de una DECISION y su funcion de perdida, y aqui
no se optimiza una decision sino la estabilidad de una prediccion. Lo que se
implementa es adquisicion activa de informacion, aproximada por reduccion
esperada de la varianza del ensamble -- el objetivo estandar de active
learning / diseno experimental. Es novedoso sin sobreprometer.

Hasta ahora `core/risk_model.py::suggest_question` usaba un proxy: la
variable faltante de mayor peso clinico fijo. Este modulo implementa el
mecanismo real que describe la Seccion 6.3, que requiere el Random Forest
entrenado (por eso vive separado y se usa solo cuando `load_model()` no
devuelve None):

    pregunta* = argmax_j [ Var(riesgo | datos de hoy)
                            - E[ Var(riesgo | datos de hoy, variable j) ] ]

Donde Var(riesgo | ...) es la varianza de las predicciones INDIVIDUALES de
cada arbol del Random Forest (no el promedio -- el promedio es la prediccion
final; la dispersion entre arboles es la medida de incertidumbre). El
algoritmo, paso a paso (identico a la Seccion 6.3):

  1. Para cada variable NO registrada hoy, se simula rellenarla con sus 2-3
     valores mas probables, usando la distribucion historica del nino (o la
     de la poblacion si el nino es nuevo -- el mismo principio de pooling
     jerarquico de la Seccion 3 aplicado aqui a "que tanto se sabe de las
     variables de este nino en particular", no solo de su tasa de crisis).
  2. Para cada escenario simulado, se recalcula la varianza del ensamble.
  3. Se promedia esa varianza (ponderada por la probabilidad de cada
     escenario) y se compara contra la varianza actual (sin ese dato).
  4. La variable cuya reduccion esperada es mayor se elige como pregunta.

Es un "greedy de un solo paso": no se reentrena el modelo ni se simula mas
de una variable a la vez, por lo que se calcula en milisegundos-segundos,
sin infraestructura adicional (asi lo exige el plazo de la hackaton).

  5. Finalmente, la ganancia se descuenta por la CARGA DE REGISTRO (ver
     REGISTRO_COSTO mas abajo). La variable mas informativa no siempre es la
     que vale la pena preguntar: si responderla exige molestar al nino,
     revivir un episodio dificil o esperar a que conteste el colegio, una
     mejora minima de la prediccion no lo compensa. Esto conecta el mecanismo
     directamente con el objetivo declarado de reducir la carga de registro
     de las familias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bayesian import K_DEFAULT
from .features import build_features_for_today

# Variables que la interfaz puede "preguntar" -- coincide con el catalogo de
# la pestana "Hoy" de app.py (14 variables de la Seccion 4.1).
# n_eventos_desregulacion es un campo COMPUESTO: al simularlo se completan
# tambien intensidad/tipo/resultado con valores representativos del nino.
ASKABLE_FIELDS = [
    "calidad_sueno", "modo_despertar", "adherencia_medicacion",
    "estado_gastrointestinal", "nivel_regulacion_general_dia",
    "nivel_apoyo_requerido", "cambios_alimentacion", "cambios_rutina",
    "comportamiento_observado", "estado_alerta", "participacion_actividades",
    "interacciones_sociales", "alimentacion_recreos", "n_eventos_desregulacion",
]

# --- Carga de registro (Seccion 6.3 revisada) --------------------------------
# El costo NO es un numero puesto a dedo: se descompone en tres factores
# observables en la app real de Bluba, de modo que sea auditable y discutible
# con el equipo clinico.
#
#   costo = n_campos * peso_informante + carga_emocional
#
#   n_campos         cuantos campos hay que completar en la app.
#   peso_informante  1.0 si puede responderlo la familia ahi mismo; 2.2 si
#                    depende del colegio o del terapeuta, porque entonces la
#                    respuesta es asincrona y puede no llegar hoy.
#   carga_emocional  0.0 dato objetivo; 0.15 dato intimo; 0.30 exige un juicio
#                    sobre el dia completo; 0.80 exige revivir un episodio.
INFORMANTE_FAMILIA = 1.0
INFORMANTE_TERCERO = 2.2

REGISTRO_COSTO_DETALLE: dict[str, dict] = {
    # Bitacora diaria del tutor: un toque, dato objetivo.
    "calidad_sueno":                dict(campos=1, informante=INFORMANTE_FAMILIA, emocional=0.0),
    "modo_despertar":               dict(campos=1, informante=INFORMANTE_FAMILIA, emocional=0.0),
    "adherencia_medicacion":        dict(campos=1, informante=INFORMANTE_FAMILIA, emocional=0.0),
    "cambios_rutina":               dict(campos=1, informante=INFORMANTE_FAMILIA, emocional=0.0),
    "cambios_alimentacion":         dict(campos=1, informante=INFORMANTE_FAMILIA, emocional=0.0),
    "estado_gastrointestinal":      dict(campos=1, informante=INFORMANTE_FAMILIA, emocional=0.15),
    # Exigen un juicio sobre el dia completo, no un dato puntual.
    "nivel_regulacion_general_dia": dict(campos=1, informante=INFORMANTE_FAMILIA, emocional=0.30),
    "comportamiento_observado":     dict(campos=1, informante=INFORMANTE_FAMILIA, emocional=0.30),
    "estado_alerta":                dict(campos=1, informante=INFORMANTE_FAMILIA, emocional=0.30),
    "nivel_apoyo_requerido":        dict(campos=1, informante=INFORMANTE_FAMILIA, emocional=0.30),
    # Solo los sabe el colegio: la familia no puede responderlos hoy.
    "participacion_actividades":    dict(campos=1, informante=INFORMANTE_TERCERO, emocional=0.0),
    "interacciones_sociales":       dict(campos=1, informante=INFORMANTE_TERCERO, emocional=0.0),
    "alimentacion_recreos":         dict(campos=1, informante=INFORMANTE_TERCERO, emocional=0.0),
    # Registro compuesto (cantidad + intensidad + tipo + resultado) y ademas
    # obliga a volver sobre un momento dificil. Es la pregunta mas cara.
    "n_eventos_desregulacion":      dict(campos=4, informante=INFORMANTE_FAMILIA, emocional=0.80),
}


def costo_registro(field: str) -> float:
    """Carga de pedir `field`, en unidades comparables entre variables."""
    d = REGISTRO_COSTO_DETALLE.get(field)
    if d is None:
        return INFORMANTE_FAMILIA
    return d["campos"] * d["informante"] + d["emocional"]


REGISTRO_COSTO = {f: costo_registro(f) for f in ASKABLE_FIELDS}
COSTO_MAX = max(REGISTRO_COSTO.values())

# Cuanta ganancia informativa estamos dispuestos a resignar para ahorrarle a la
# familia una unidad completa de carga. Con LAMBDA_CARGA = 0.35, la pregunta
# mas cara del catalogo (el registro de un episodio) tiene que ser al menos
# ~35% mas informativa que la mas barata para que compense pedirla.
# Es el unico parametro subjetivo del mecanismo y esta deliberadamente aislado
# en una constante para poder discutirlo con el equipo clinico.
LAMBDA_CARGA = 0.35


def net_utilities(reductions: dict[str, float],
                  lambda_carga: float = LAMBDA_CARGA) -> dict[str, float]:
    """Utilidad neta de cada pregunta: ganancia informativa menos carga.

        U_j = g_j - lambda * c_j

    donde g_j es la ganancia RELATIVA a la mejor candidata del dia (0-1, para
    que lambda conserve el mismo significado aunque la escala absoluta de las
    varianzas cambie de un dia a otro) y c_j es la carga normalizada por la
    pregunta mas cara del catalogo.

    Una reduccion negativa (la variable, en promedio, AUMENTA la dispersion
    del ensamble) cuenta como ganancia cero: no se pide un dato que no ayuda.
    """
    if not reductions:
        return {}
    mejor = max(reductions.values())
    if mejor <= 0:
        # Ningun dato faltante reduce la incertidumbre: toda ganancia es cero y
        # solo queda la carga, asi que TODAS las utilidades son negativas.
        # Quien decide que hacer con eso es vale_la_pena_preguntar().
        return {f: -lambda_carga * (REGISTRO_COSTO.get(f, 1.0) / COSTO_MAX)
                for f in reductions}
    return {
        f: max(0.0, red) / mejor - lambda_carga * (REGISTRO_COSTO.get(f, 1.0) / COSTO_MAX)
        for f, red in reductions.items()
    }


def vale_la_pena_preguntar(reductions: dict[str, float]) -> bool:
    """Decide si HOY corresponde preguntar algo.

    Si ninguna variable faltante reduce la dispersion esperada del ensamble, la
    respuesta correcta no es "pregunta lo mas barato": es NO PREGUNTAR. Un
    sistema que se ha propuesto reducir la carga de registro tiene que poder
    quedarse callado los dias en que no tiene nada util que pedir; preguntar
    igual solo para llenar el formulario es exactamente el habito que Antesala
    dice estar corrigiendo.

    Ocurre de verdad: sobre las bitacoras sinteticas, en dias sin ningun
    registro hay 7 de 28 ninos para los que ninguna pregunta ayuda (con 3
    campos ya registrados baja a 2 de 28).
    """
    return bool(reductions) and max(reductions.values()) > 0


# Bajo este numero de observaciones propias, se usa la distribucion
# poblacional en vez de la del nino (arranque en frio, Seccion 3.6).
MIN_CHILD_OBSERVATIONS = 5

# Tope de escenarios simulados por variable (Seccion 6.3: "sus 2 o 3 valores
# mas probables").
MAX_CANDIDATES = 3


def _tree_predictions_batch(model: dict, rows: pd.DataFrame) -> np.ndarray:
    """Probabilidad de crisis_24h segun CADA arbol del Random Forest, para
    TODAS las filas de `rows` a la vez (una fila por escenario simulado).
    Devuelve un array (n_arboles, n_filas).

    Critico para el rendimiento: predict_proba() de un arbol ya esta
    vectorizado sobre filas, asi que evaluar N escenarios en UNA sola llamada
    por arbol (en vez de UNA llamada por arbol POR escenario) reduce el
    numero de llamadas Python de (n_arboles * n_escenarios) a solo n_arboles
    -- con 400 arboles y ~35 escenarios (14 variables x 2-3 valores) esto es
    la diferencia entre ~14.000 llamadas y 400."""
    pipeline = model["pipeline"]
    cols = model["feature_numeric"] + model["feature_categorical"]
    X_t = pipeline.named_steps["pre"].transform(rows[cols])
    rf = pipeline.named_steps["rf"]
    return np.array([tree.predict_proba(X_t)[:, 1] for tree in rf.estimators_])


def _empirical_candidates(logs: pd.DataFrame, child_id: str, field: str) -> list[tuple]:
    """2-3 valores mas probables para `field`, con su probabilidad de
    ocurrencia. Usa el historial PROPIO del nino si tiene suficientes
    observaciones; si no (nino nuevo -- cold start), usa la distribucion de
    TODA la poblacion. Mismo principio que el pooling jerarquico de la
    Seccion 3, aplicado aqui a que tan bien se conoce el patron de cada
    variable de este nino en particular."""
    child_vals = logs.loc[logs["child_id"] == child_id, field].dropna()
    source = child_vals if len(child_vals) >= MIN_CHILD_OBSERVATIONS else logs[field].dropna()
    if len(source) == 0:
        return []
    counts = source.value_counts(normalize=True).head(MAX_CANDIDATES)
    total = float(counts.sum())
    if total <= 0:
        return []
    return [(val, float(p) / total) for val, p in counts.items()]


def _event_candidates(logs: pd.DataFrame, child_id: str) -> list[tuple[dict, float]]:
    """Caso especial: n_eventos_desregulacion activa ademas intensidad, tipo
    y resultado de estrategia. Se simulan 2 escenarios -- "no hubo episodio"
    y "hubo un episodio tipico" -- con la probabilidad historica de cada uno
    (nino propio si hay suficiente historial, si no poblacion)."""
    child_rows = logs[logs["child_id"] == child_id]
    source = child_rows if len(child_rows) >= MIN_CHILD_OBSERVATIONS else logs

    n_ev = pd.to_numeric(source["n_eventos_desregulacion"], errors="coerce").dropna()
    if len(n_ev) == 0:
        return []
    p_evento = float(np.clip((n_ev > 0).mean(), 0.02, 0.98))

    has_event = source[pd.to_numeric(source["n_eventos_desregulacion"], errors="coerce") > 0]
    if len(has_event):
        inten = pd.to_numeric(has_event["intensidad_max_desregulacion"], errors="coerce").dropna()
        intensidad_tipica = float(inten.median()) if len(inten) else 5.0
        tipo_mode = has_event["tipo_evento_principal"].mode()
        tipo_tipico = tipo_mode.iloc[0] if len(tipo_mode) else "Sobrecarga Sensorial"
        res_mode = has_event["resultado_estrategia_principal"].mode()
        resultado_tipico = res_mode.iloc[0] if len(res_mode) else "Regulacion Exitosa"
    else:
        intensidad_tipica, tipo_tipico, resultado_tipico = 5.0, "Sobrecarga Sensorial", "Regulacion Exitosa"

    return [
        ({"n_eventos_desregulacion": 0, "intensidad_max_desregulacion": np.nan,
          "intensidad_sum_desregulacion": 0.0, "tipo_evento_principal": None,
          "resultado_estrategia_principal": None}, 1.0 - p_evento),
        ({"n_eventos_desregulacion": 1, "intensidad_max_desregulacion": intensidad_tipica,
          "intensidad_sum_desregulacion": intensidad_tipica, "tipo_evento_principal": tipo_tipico,
          "resultado_estrategia_principal": resultado_tipico}, p_evento),
    ]


def expected_variance_reductions(
    logs: pd.DataFrame,
    child_id: str,
    today: dict,
    missing_fields: list[str],
    model: dict,
    k: int = K_DEFAULT,
) -> dict[str, float]:
    """Devuelve, para cada variable faltante preguntable, la reduccion
    esperada de varianza del ensamble si se registrara hoy (formula de la
    Seccion 6.3). Puede devolver valores negativos (una variable puede, en
    promedio, no reducir -- incluso aumentar levemente -- la incertidumbre).

    Construye TODOS los escenarios primero (linea base + cada valor probable
    de cada variable faltante) y evalua los arboles del Random Forest UNA
    sola vez sobre todos ellos a la vez (ver _tree_predictions_batch) -- en
    vez de una pasada completa por los 400 arboles POR escenario."""
    candidates = [f for f in missing_fields if f in ASKABLE_FIELDS]
    if not candidates:
        return {}

    child_hist = logs[logs["child_id"] == child_id]  # se filtra UNA sola vez

    # --- 1. Construir todos los escenarios (linea base + por variable) ---
    scenario_rows: list[pd.DataFrame] = []
    scenario_field: list[str | None] = []   # None = linea base
    scenario_prob: list[float] = []

    base_row = build_features_for_today(child_hist, child_id, today, k=k, mu=model["mu"])
    scenario_rows.append(base_row)
    scenario_field.append(None)
    scenario_prob.append(1.0)

    field_scenario_count: dict[str, int] = {}
    for field in candidates:
        if field == "n_eventos_desregulacion":
            scenarios = _event_candidates(logs, child_id)
        else:
            scenarios = [({field: val}, p) for val, p in _empirical_candidates(logs, child_id, field)]
        field_scenario_count[field] = len(scenarios)
        for updates, prob in scenarios:
            hypo_today = dict(today)
            hypo_today.update(updates)
            row = build_features_for_today(child_hist, child_id, hypo_today, k=k, mu=model["mu"])
            scenario_rows.append(row)
            scenario_field.append(field)
            scenario_prob.append(prob)

    if len(scenario_rows) <= 1:
        return {}

    # --- 2. Evaluar TODOS los escenarios en una sola pasada por los arboles ---
    all_rows = pd.concat(scenario_rows, ignore_index=True)
    tree_preds = _tree_predictions_batch(model, all_rows)  # (n_arboles, n_escenarios)
    variances = tree_preds.var(axis=0)  # (n_escenarios,)

    base_var = float(variances[0])

    # --- 3. Promediar (ponderado) la varianza por variable y calcular reduccion ---
    reductions: dict[str, float] = {}
    idx = 1
    for field in candidates:
        n = field_scenario_count.get(field, 0)
        if n == 0:
            continue
        probs = np.array(scenario_prob[idx: idx + n])
        vars_ = variances[idx: idx + n]
        idx += n
        if probs.sum() <= 0:
            continue
        expected_var = float(np.sum(probs * vars_) / probs.sum())
        reductions[field] = base_var - expected_var
    return reductions


def select_question_variance(
    logs: pd.DataFrame,
    child_id: str,
    today: dict,
    missing_fields: list[str],
    model: dict | None,
    k: int = K_DEFAULT,
) -> str | None:
    """La pregunta del dia: argmax de la UTILIDAD NETA (ganancia informativa
    menos carga de registro), no de la ganancia bruta -- Seccion 6.3 revisada.
    Devuelve None si no hay modelo o no se pudo calcular ninguna reduccion (el
    llamador debe caer al proxy heuristico)."""
    if model is None or not missing_fields:
        return None
    reductions = expected_variance_reductions(logs, child_id, today, missing_fields, model, k)
    if not vale_la_pena_preguntar(reductions):
        return None
    utilities = net_utilities(reductions)
    return max(utilities, key=utilities.get)
