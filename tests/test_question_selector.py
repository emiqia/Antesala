"""
Prueba el selector de "la pregunta del dia" por reduccion esperada de
varianza del ensamble -- Seccion 6.3, nivel principal. Requiere el Random
Forest entrenado (models/antesala_rf.joblib); los tests se saltan si no
existe (correr 'python core/train_model.py' primero).
"""
import pytest

from core.question_selector import (
    ASKABLE_FIELDS,
    expected_variance_reductions,
    select_question_variance,
)
from core.risk_model import load_model


def test_select_question_variance_none_without_model(logs, full_history_child_id):
    chosen = select_question_variance(logs, full_history_child_id, {}, ASKABLE_FIELDS, model=None)
    assert chosen is None


def test_select_question_variance_none_when_nothing_missing(logs, full_history_child_id, has_model):
    if not has_model:
        pytest.skip("modelo no entrenado -- correr 'python core/train_model.py'")
    model = load_model()
    chosen = select_question_variance(logs, full_history_child_id, {}, [], model=model)
    assert chosen is None


def test_select_question_variance_returns_an_askable_field(logs, full_history_child_id, has_model):
    if not has_model:
        pytest.skip("modelo no entrenado -- correr 'python core/train_model.py'")
    model = load_model()
    chosen = select_question_variance(logs, full_history_child_id, {}, ASKABLE_FIELDS, model=model)
    assert chosen is None or chosen in ASKABLE_FIELDS


def test_expected_variance_reductions_are_finite(logs, full_history_child_id, has_model):
    if not has_model:
        pytest.skip("modelo no entrenado -- correr 'python core/train_model.py'")
    model = load_model()
    reductions = expected_variance_reductions(logs, full_history_child_id, {}, ASKABLE_FIELDS, model)
    assert len(reductions) > 0
    for field, value in reductions.items():
        assert field in ASKABLE_FIELDS
        assert value == value  # descarta NaN
        assert -1.0 <= value <= 1.0  # es una diferencia de varianzas de probabilidades [0,1]


def test_select_question_matches_argmax_of_reductions(logs, full_history_child_id, has_model):
    """La variable elegida debe ser exactamente la de mayor reduccion
    esperada -- verifica que select_question_variance no se desincroniza de
    expected_variance_reductions."""
    if not has_model:
        pytest.skip("modelo no entrenado -- correr 'python core/train_model.py'")
    model = load_model()
    reductions = expected_variance_reductions(logs, full_history_child_id, {}, ASKABLE_FIELDS, model)
    chosen = select_question_variance(logs, full_history_child_id, {}, ASKABLE_FIELDS, model=model)
    if reductions:
        assert chosen == max(reductions, key=reductions.get)
    else:
        assert chosen is None
