"""
Prueba el motor de riesgo (Seccion 6.1) y el calculo de confianza
(Seccion 6.2) sobre los datos sinteticos reales del proyecto.
"""
from core.risk_model import RiskResult, predict_risk, score_heuristic, suggest_question

FULL_GOOD_DAY = {
    "calidad_sueno": "Reparador", "modo_despertar": "Tranquilo/Alegre",
    "adherencia_medicacion": "Si", "estado_gastrointestinal": "Normal",
    "nivel_regulacion_general_dia": "Excelente", "n_eventos_desregulacion": 0,
    "nivel_apoyo_requerido": "Bajo", "cambios_alimentacion": "Sin cambios",
    "cambios_rutina": "No", "comportamiento_observado": "Estable",
    "estado_alerta": "Optimo (Regulado)", "participacion_actividades": "Completa",
    "interacciones_sociales": "Normal", "alimentacion_recreos": "Normal",
}

BAD_DAY = {
    "calidad_sueno": "Dificultad de Conciliacion", "modo_despertar": "Irritable/Llorando",
    "adherencia_medicacion": "No", "estado_gastrointestinal": "Diarrea",
    "nivel_regulacion_general_dia": "Desregulacion Frecuente", "n_eventos_desregulacion": 2,
    "nivel_apoyo_requerido": "Alto", "cambios_alimentacion": "Selectividad aumentada",
    "cambios_rutina": "Si", "comportamiento_observado": "Desregulado",
    "estado_alerta": "Alto (Sobreexcitado)", "participacion_actividades": "No participa",
    "interacciones_sociales": "Evitativa", "alimentacion_recreos": "Rechaza",
}


def test_score_heuristic_empty_today_has_zero_confidence(logs, full_history_child_id):
    result = score_heuristic(logs, full_history_child_id, {})
    assert result.confidence == 0.0
    assert result.confidence_level == "baja"
    assert len(result.missing_relevant) > 0


def test_score_heuristic_full_today_has_higher_confidence_than_empty(logs, full_history_child_id):
    empty = score_heuristic(logs, full_history_child_id, {})
    full = score_heuristic(logs, full_history_child_id, FULL_GOOD_DAY)
    assert full.confidence > empty.confidence
    assert len(full.missing_relevant) == 0


def test_confidence_thresholds_match_doc_section_6_2(logs, full_history_child_id):
    # Seccion 6.2: <0.4 baja, 0.4-0.7 moderada, >=0.7 alta.
    result = score_heuristic(logs, full_history_child_id, FULL_GOOD_DAY)
    if result.confidence < 0.4:
        assert result.confidence_level == "baja"
    elif result.confidence < 0.7:
        assert result.confidence_level == "moderada"
    else:
        assert result.confidence_level == "alta"


def test_bad_day_yields_higher_or_equal_risk_than_good_day(logs, full_history_child_id):
    """Chequeo direccional del heuristico: un dia con todas las variables en
    su peor valor no puede dar MENOS riesgo que uno con todas en su mejor
    valor (esto fallaba antes de corregir el generador de datos)."""
    r_good = score_heuristic(logs, full_history_child_id, FULL_GOOD_DAY)
    r_bad = score_heuristic(logs, full_history_child_id, BAD_DAY)
    assert r_bad.risk >= r_good.risk


def test_predict_risk_falls_back_to_heuristic_without_model(logs, full_history_child_id, monkeypatch):
    import core.risk_model as rm
    monkeypatch.setattr(rm, "load_model", lambda: None)
    result = rm.predict_risk(logs, full_history_child_id, {})
    assert result.model_used == "heuristico"


def test_predict_risk_uses_random_forest_when_available(logs, full_history_child_id, has_model):
    if not has_model:
        import pytest
        pytest.skip("modelo no entrenado -- correr 'python core/train_model.py'")
    result = predict_risk(logs, full_history_child_id, {})
    assert result.model_used == "random_forest"
    assert 0.0 <= result.risk <= 1.0
    assert result.base_rate_bayes is not None
    assert 0.0 <= result.base_rate_bayes <= 1.0


def test_predict_risk_bad_day_scores_higher_than_good_day_with_rf(logs, full_history_child_id, has_model):
    if not has_model:
        import pytest
        pytest.skip("modelo no entrenado -- correr 'python core/train_model.py'")
    r_good = predict_risk(logs, full_history_child_id, FULL_GOOD_DAY)
    r_bad = predict_risk(logs, full_history_child_id, BAD_DAY)
    assert r_bad.risk > r_good.risk


def test_suggest_question_heuristic_returns_none_when_nothing_missing():
    result = RiskResult(child_id="x", risk=0.1, sufficiency=1.0, sufficiency_level="alta",
                         missing_relevant=[])
    assert suggest_question(result) is None


def test_suggest_question_heuristic_picks_highest_weighted_missing_field():
    result = RiskResult(child_id="x", risk=0.1, sufficiency=0.2, sufficiency_level="baja",
                         missing_relevant=["adherencia_medicacion", "calidad_sueno"])
    # calidad_sueno tiene peso clinico 3.0, mayor que adherencia_medicacion (1.0).
    assert suggest_question(result) == "calidad_sueno"


def test_predict_risk_populates_suggested_question_when_data_missing(logs, full_history_child_id):
    result = predict_risk(logs, full_history_child_id, {})
    assert result.suggested_question is not None
    assert result.question_method in ("reduccion_varianza", "heuristico")


def test_predict_risk_suggested_question_is_none_when_nothing_missing(logs, full_history_child_id):
    result = predict_risk(logs, full_history_child_id, FULL_GOOD_DAY)
    assert result.suggested_question is None


def test_event_fields_have_genuine_missingness_in_training_data(logs):
    """Regresion: n_eventos_desregulacion debe poder faltar en los datos de
    entrenamiento -- si nunca falta, SimpleImputer(add_indicator=True) no crea
    un indicador de ausencia para el, y el Random Forest no puede distinguir
    'no hubo episodio' (0 confirmado) de 'no sabemos' (no registrado),
    imputando en silencio y violando la Seccion 4.4."""
    assert logs["n_eventos_desregulacion"].isna().any()


def test_training_data_contains_fully_blank_days(logs):
    """Regresion: el dataset debe contener dias SIN NINGUN registro.

    Sortear la ausencia campo por campo hace que un dia completamente vacio
    sea practicamente imposible, y entonces el modelo nunca ve esa region del
    espacio de entrada. Pero es exactamente la que la interfaz consulta al
    abrir cada dia (antes de que nadie registre nada): sin estas filas, el
    numero de la pantalla inicial es una extrapolacion, no una prediccion."""
    from core.question_selector import ASKABLE_FIELDS
    campos = [c for c in ASKABLE_FIELDS if c in logs.columns]
    totalmente_vacios = (logs[campos].isna().sum(axis=1) == len(campos)).sum()
    assert totalmente_vacios >= 50, (
        f"solo {totalmente_vacios} dias sin ningun registro en {len(logs)} filas")


def test_blank_day_risk_stays_in_a_plausible_range(logs, has_model):
    """Un dia en blanco no puede dar un riesgo extremo para todos los ninos: el
    silencio es una senal (MNAR, Seccion 4.4), pero no debe DECIDIR. Si todos
    los ninos sin registro dieran ~0.9, el modelo estaria leyendo la ausencia
    en vez del contenido clinico."""
    if not has_model:
        import pytest
        pytest.skip("modelo no entrenado -- correr 'python core/train_model.py'")
    riesgos = [predict_risk(logs, cid, {}, compute_question=False).risk
               for cid in logs["child_id"].unique()]
    assert max(riesgos) < 0.85, f"riesgo maximo en dia vacio demasiado alto: {max(riesgos)}"
    # Y deben DIFERIR entre ninos: si el dia esta vacio, lo unico que queda es
    # la linea base bayesiana propia de cada nino, que no es la misma para todos.
    assert max(riesgos) - min(riesgos) > 0.15


def test_content_beats_missingness(logs, has_model):
    """El contenido clinico debe pesar mas que el patron de ausencia: un dia
    COMPLETO con todo en su peor valor tiene que dar mas riesgo que el mismo
    dia completo con todo en su mejor valor, para todos los ninos."""
    if not has_model:
        import pytest
        pytest.skip("modelo no entrenado -- correr 'python core/train_model.py'")
    peor = {"calidad_sueno": "Dificultad de Conciliacion", "modo_despertar": "Irritable/Llorando",
            "adherencia_medicacion": "No", "estado_gastrointestinal": "Diarrea",
            "nivel_regulacion_general_dia": "Desregulacion Frecuente", "cambios_rutina": "Si",
            "comportamiento_observado": "Desregulado", "estado_alerta": "Alto (Sobreexcitado)",
            "n_eventos_desregulacion": 2, "nivel_apoyo_requerido": "Alto",
            "interacciones_sociales": "Evitativa", "participacion_actividades": "No participa",
            "cambios_alimentacion": "Selectividad aumentada", "alimentacion_recreos": "Rechaza"}
    mejor = {"calidad_sueno": "Reparador", "modo_despertar": "Tranquilo/Alegre",
             "adherencia_medicacion": "Si", "estado_gastrointestinal": "Normal",
             "nivel_regulacion_general_dia": "Excelente", "cambios_rutina": "No",
             "comportamiento_observado": "Estable", "estado_alerta": "Optimo (Regulado)",
             "n_eventos_desregulacion": 0, "nivel_apoyo_requerido": "Bajo",
             "interacciones_sociales": "Normal", "participacion_actividades": "Completa",
             "cambios_alimentacion": "Sin cambios", "alimentacion_recreos": "Normal"}
    for cid in logs["child_id"].unique():
        r_peor = predict_risk(logs, cid, peor, compute_question=False).risk
        r_mejor = predict_risk(logs, cid, mejor, compute_question=False).risk
        assert r_peor > r_mejor, f"{cid}: peor={r_peor} no supera a mejor={r_mejor}"


def test_confirmed_zero_events_differs_from_unknown_events(logs, full_history_child_id, has_model):
    """Con el Random Forest, 'confirmado que no hubo episodio' y 'no se
    registro si hubo episodio' deben dar un riesgo DISTINTO -- si el modelo
    no puede distinguirlos, esta imputando el dato faltante en silencio."""
    if not has_model:
        import pytest
        pytest.skip("modelo no entrenado -- correr 'python core/train_model.py'")
    r_confirmed_zero = predict_risk(logs, full_history_child_id, {"n_eventos_desregulacion": 0})
    r_unknown = predict_risk(logs, full_history_child_id, {})
    assert r_confirmed_zero.risk != r_unknown.risk


def test_confidence_weights_come_from_model_not_clinical_weights(logs, full_history_child_id, has_model):
    """Seccion 6.2: los pesos de completitud deben venir del feature
    importance del Random Forest, no de VARIABLE_WEIGHTS (los pesos clinicos
    que usa el score heuristico, Seccion 6.1) -- son fuentes distintas."""
    if not has_model:
        import pytest
        pytest.skip("modelo no entrenado -- correr 'python core/train_model.py'")
    from core.risk_model import VARIABLE_WEIGHTS, load_model
    model = load_model()
    confidence_weights = model["confidence_weights"]
    assert confidence_weights  # no vacio
    # Si vinieran de VARIABLE_WEIGHTS, coincidirian con los pesos clinicos
    # (3.0, 2.5, 1.8, ...); el feature importance del RF esta en una escala
    # totalmente distinta (suma de importancias <= 1.0 en toda la matriz).
    assert all(0.0 <= w <= 1.0 for w in confidence_weights.values())
    assert confidence_weights != {f: VARIABLE_WEIGHTS.get(f, 1.0) for f in confidence_weights}
