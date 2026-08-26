"""
Prueba que RIESGO, SUFICIENCIA DE INFORMACION e INCERTIDUMBRE PREDICTIVA sean
tres cosas distintas y no tres nombres del mismo numero.

Esta es la correccion central que pidio la revision metodologica de agosto
2026: el prototipo llamaba "confianza" a completitud x historial, que no es la
probabilidad de que la prediccion sea correcta. Si los tests no distinguen las
tres cantidades, el renombre seria cosmetico.
"""
import numpy as np
import pytest

from core.uncertainty import (
    UMBRAL_ESTABLE, UMBRAL_INESTABLE, dispersion_normalizada, evaluate,
)
from core.risk_model import load_model, predict_risk


# ------------------------------------------------- dispersion normalizada ---
def test_dispersion_cero_cuando_todos_los_arboles_coinciden():
    assert dispersion_normalizada(np.array([0.7, 0.7, 0.7])) == pytest.approx(0.0, abs=1e-12)


def test_dispersion_uno_cuando_el_ensamble_esta_partido_por_la_mitad():
    """Mitad de los arboles dice 0 y mitad dice 1: es el maximo desacuerdo
    posible, y la escala normalizada debe marcarlo como 1.0."""
    assert dispersion_normalizada(np.array([0.0, 1.0, 0.0, 1.0])) == pytest.approx(1.0)


def test_dispersion_es_comparable_entre_predicciones_extremas_y_centrales():
    """El punto de normalizar: la desviacion estandar CRUDA no es comparable
    entre una prediccion cerca de 0.5 y otra cerca de 0 o 1, porque la
    dispersion maxima posible depende de la media (sqrt(p(1-p))). Sin
    normalizar, toda prediccion extrema pareceria estable solo por ser
    extrema."""
    # Ambos casos estan al 50% de su desacuerdo maximo posible.
    central = np.array([0.25, 0.75])          # media 0.5, std 0.25, max 0.5
    extremo = np.array([0.05, 0.15])          # media 0.1, std 0.05, max 0.3
    assert dispersion_normalizada(central) == pytest.approx(0.5, abs=0.02)
    # La std cruda del caso extremo es 5 veces menor...
    assert extremo.std() < central.std() / 4
    # ...pero su dispersion NORMALIZADA no es despreciable.
    assert dispersion_normalizada(extremo) > 0.15


def test_dispersion_acotada_en_cero_uno():
    for preds in [np.array([0.0]), np.array([1.0]), np.random.default_rng(0).random(50)]:
        d = dispersion_normalizada(preds)
        assert 0.0 <= d <= 1.0


def test_umbrales_ordenados():
    assert 0.0 < UMBRAL_ESTABLE < UMBRAL_INESTABLE < 1.0


# --------------------------------------------------------------- evaluate ---
def test_evaluate_sin_modelo_no_inventa_un_numero():
    """Sin ensamble no hay incertidumbre que medir. Devolver un 0 aqui seria
    peor que devolver 'no disponible': se leeria como 'prediccion perfectamente
    estable'."""
    r = evaluate(None, None)
    assert r["disponible"] is False
    assert r["dispersion"] is None
    assert r["nivel"] == "desconocida"


def test_evaluate_devuelve_banda_coherente(logs, full_history_child_id, has_model):
    if not has_model:
        pytest.skip("modelo no entrenado -- correr 'python core/train_model.py'")
    r = predict_risk(logs, full_history_child_id, {}, compute_question=False)
    u = r.uncertainty
    assert u["disponible"] is True
    assert 0.0 <= u["p10"] <= u["media"] <= u["p90"] <= 1.0
    assert u["nivel"] in ("estable", "moderada", "inestable")


# ----------------------------------- las tres cantidades son independientes --
def test_suficiencia_e_incertidumbre_no_son_el_mismo_numero(logs, has_model):
    """Si `1 - dispersion` fuera siempre igual a la suficiencia, separarlas
    habria sido un renombre y no una correccion. Aqui se comprueba que a lo
    largo de la cohorte las dos series discrepan de verdad."""
    if not has_model:
        pytest.skip("modelo no entrenado -- correr 'python core/train_model.py'")
    # Se recorre una rejilla de nino x grado de completitud: si solo se
    # miraran dias en blanco, la suficiencia valdria 0 para todos y no habria
    # nada que correlacionar.
    campos = ["calidad_sueno", "modo_despertar", "adherencia_medicacion",
              "estado_gastrointestinal", "nivel_regulacion_general_dia",
              "cambios_rutina", "comportamiento_observado"]
    valores = {"calidad_sueno": "Interrumpido", "modo_despertar": "Cansado/Con Sueno",
               "adherencia_medicacion": "Si", "estado_gastrointestinal": "Normal",
               "nivel_regulacion_general_dia": "Estable con Apoyo",
               "cambios_rutina": "No", "comportamiento_observado": "Inquieto"}
    suf, estab = [], []
    for cid in list(logs["child_id"].unique())[:12]:
        for n in (0, 2, 4, 7):
            hoy = {c: valores[c] for c in campos[:n]}
            r = predict_risk(logs, cid, hoy, compute_question=False)
            if r.uncertainty.get("disponible"):
                suf.append(r.sufficiency)
                estab.append(1.0 - r.uncertainty["dispersion"])
    assert len(suf) >= 20
    # Ninguna de las dos series es constante: las dos miden algo que varia.
    # El umbral de `estab` es mucho mas bajo a proposito -- la estabilidad se
    # mueve en una banda estrecha (los arboles rara vez pasan de un acuerdo
    # total a un desacuerdo total), y eso mismo es parte del argumento: no
    # sigue a la suficiencia, que aqui recorre de 0 a ~0.5.
    assert np.std(suf) > 0.05
    assert np.std(estab) > 0.01
    # Discrepan en magnitud...
    assert max(abs(a - b) for a, b in zip(suf, estab)) > 0.2
    # ...y no son una transformacion monotona una de la otra: con la MISMA
    # suficiencia hay predicciones estables e inestables.
    assert abs(float(np.corrcoef(suf, estab)[0, 1])) < 0.9


def test_registro_completo_no_implica_prediccion_estable(logs, has_model):
    """El argumento textual de la revision: "un registro completo y mucho
    historial no implican necesariamente que el modelo este seguro". Con el
    dia COMPLETO registrado la suficiencia es alta para todos; si aun asi hay
    algun nino cuya prediccion no es estable, las dos cosas estan realmente
    separadas."""
    if not has_model:
        pytest.skip("modelo no entrenado -- correr 'python core/train_model.py'")
    dia_completo = {
        "calidad_sueno": "Reparador", "modo_despertar": "Tranquilo/Alegre",
        "adherencia_medicacion": "Si", "estado_gastrointestinal": "Normal",
        "nivel_regulacion_general_dia": "Excelente", "n_eventos_desregulacion": 0,
        "nivel_apoyo_requerido": "Bajo", "cambios_alimentacion": "Sin cambios",
        "cambios_rutina": "No", "comportamiento_observado": "Estable",
        "estado_alerta": "Optimo (Regulado)", "participacion_actividades": "Completa",
        "interacciones_sociales": "Normal", "alimentacion_recreos": "Normal",
    }
    inestables = 0
    alta_suficiencia = 0
    for cid in logs["child_id"].unique():
        r = predict_risk(logs, cid, dia_completo, compute_question=False)
        if r.sufficiency_level == "alta":
            alta_suficiencia += 1
            if r.uncertainty.get("nivel") != "estable":
                inestables += 1
    assert alta_suficiencia >= 10, "el dia completo deberia dar suficiencia alta"
    assert inestables > 0, (
        "ningun nino con suficiencia alta tiene prediccion no-estable: "
        "las dos medidas podrian estar midiendo lo mismo")


def test_alias_confidence_sigue_funcionando(logs, full_history_child_id):
    """Compatibilidad: el nombre viejo se mantiene como alias de solo lectura
    para no romper codigo existente."""
    r = predict_risk(logs, full_history_child_id, {}, compute_question=False)
    assert r.confidence == r.sufficiency
    assert r.confidence_level == r.sufficiency_level
