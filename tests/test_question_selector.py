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
    """La variable elegida debe ser exactamente la de mayor UTILIDAD NETA
    (ganancia menos carga), no la de mayor reduccion bruta -- verifica que
    select_question_variance no se desincroniza del criterio de eleccion."""
    if not has_model:
        pytest.skip("modelo no entrenado -- correr 'python core/train_model.py'")
    model = load_model()
    from core.question_selector import net_utilities, vale_la_pena_preguntar
    reductions = expected_variance_reductions(logs, full_history_child_id, {}, ASKABLE_FIELDS, model)
    chosen = select_question_variance(logs, full_history_child_id, {}, ASKABLE_FIELDS, model=model)
    if vale_la_pena_preguntar(reductions):
        utilidades = net_utilities(reductions)
        assert chosen == max(utilidades, key=utilidades.get)
    else:
        assert chosen is None


# ======================= Carga de registro (Seccion 6.3 revisada) ===========
# "No siempre la variable mas informativa es la que merece preguntarse. Si una
# pregunta cuesta mucho responder o requiere molestar al nino, probablemente no
# compense una mejora minima de prediccion."

from core.question_selector import (
    COSTO_MAX, LAMBDA_CARGA, REGISTRO_COSTO, costo_registro, net_utilities,
    vale_la_pena_preguntar,
)


def test_costo_ordena_las_preguntas_como_corresponde():
    """El registro de un episodio es la pregunta mas cara del catalogo (cuatro
    campos y ademas obliga a volver sobre un momento dificil); los datos que
    solo sabe el colegio cuestan mas que los que la familia responde de un
    toque, porque la respuesta puede no llegar hoy."""
    assert costo_registro("n_eventos_desregulacion") == COSTO_MAX
    assert costo_registro("participacion_actividades") > costo_registro("calidad_sueno")
    assert costo_registro("nivel_regulacion_general_dia") > costo_registro("modo_despertar")


def test_una_pregunta_barata_gana_a_una_cara_apenas_mejor():
    """El caso que motiva todo el mecanismo: la variable mas informativa NO se
    elige si su ventaja es marginal y su costo es alto."""
    reducciones = {"n_eventos_desregulacion": 1.00e-3,   # la mas informativa, la mas cara
                   "calidad_sueno": 0.95e-3}             # casi igual de buena, un toque
    u = net_utilities(reducciones)
    assert max(u, key=u.get) == "calidad_sueno"


def test_una_pregunta_cara_gana_si_la_ventaja_es_grande():
    """El descuento por carga no puede degenerar en 'preguntar siempre lo mas
    barato': con una ventaja informativa suficiente, la pregunta cara vale la
    pena."""
    reducciones = {"n_eventos_desregulacion": 1.00e-3,
                   "calidad_sueno": 0.20e-3}
    u = net_utilities(reducciones)
    assert max(u, key=u.get) == "n_eventos_desregulacion"


def test_ganancia_negativa_no_puntua_mejor_que_cero():
    """Una variable que AUMENTA la dispersion esperada no puede quedar por
    encima de una que la deja igual solo por ser mas barata de preguntar."""
    u = net_utilities({"calidad_sueno": -5e-3, "modo_despertar": 1e-3})
    assert u["modo_despertar"] > u["calidad_sueno"]


def test_no_preguntar_es_una_salida_valida():
    """Si ninguna variable faltante reduce la incertidumbre, la respuesta
    correcta no es 'pregunta lo mas barato': es no preguntar. Un sistema que
    dice existir para reducir la carga de registro tiene que poder callarse."""
    assert vale_la_pena_preguntar({"calidad_sueno": 1e-4}) is True
    assert vale_la_pena_preguntar({"calidad_sueno": -1e-4, "modo_despertar": 0.0}) is False
    assert vale_la_pena_preguntar({}) is False


def test_todas_las_preguntables_tienen_costo_declarado():
    """Un campo sin costo declarado caeria en el valor por defecto sin que
    nadie se entere, y el descuento por carga dejaria de ser auditable."""
    for campo in ASKABLE_FIELDS:
        assert campo in REGISTRO_COSTO


def test_lambda_esta_en_un_rango_que_no_degenera():
    """Con lambda 0 el costo se ignora; con lambda >= 1 el costo domina
    siempre y el mecanismo colapsa a 'la pregunta mas barata'."""
    assert 0.0 < LAMBDA_CARGA < 1.0
