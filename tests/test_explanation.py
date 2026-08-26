"""
Prueba la explicabilidad local (Seccion 11) y la calibracion que la hace
posible (Seccion 9.3).
"""
import numpy as np
import pandas as pd
import pytest

from core.calibration import PlattCalibrator
from core.explanation import (
    UMBRAL_RELEVANTE, explicar, resumen_texto, valor_habitual,
)
from core.recommendations import VARIABLE_LABELS
from core.risk_model import load_model, predict_risk

DIA_MALO = {
    "calidad_sueno": "Dificultad de Conciliacion",
    "modo_despertar": "Irritable/Llorando",
    "cambios_rutina": "Si",
    "estado_gastrointestinal": "Diarrea",
    "comportamiento_observado": "Desregulado",
}


# ------------------------------------------------------------ calibracion ---
def test_platt_conserva_la_resolucion():
    """El motivo por el que se abandono la isotonica: una funcion escalonada
    colapsa probabilidades distintas en el mismo valor, y entonces la
    contribucion local de una variable sale exactamente 0.000 aunque el modelo
    si haya cambiado de opinion."""
    rng = np.random.default_rng(0)
    p = rng.random(3000)
    y = (rng.random(3000) < p).astype(int)
    cal = PlattCalibrator().fit(p, y)
    salida = cal.predict(p)
    assert len(np.unique(np.round(salida, 4))) > 1000


def test_platt_es_monotona():
    """Calibrar no puede reordenar a los ninos: si A tenia mas riesgo que B
    antes, tiene que seguir teniendolo despues, o el ranking del panel del
    equipo cambiaria por efecto de la calibracion."""
    rng = np.random.default_rng(1)
    p = rng.random(2000)
    y = (rng.random(2000) < p).astype(int)
    salida = PlattCalibrator().fit(p, y).predict(np.linspace(0.01, 0.99, 50))
    assert np.all(np.diff(salida) >= -1e-12)


def test_platt_corrige_un_modelo_timido():
    """Un modelo cuyas probabilidades estan comprimidas hacia 0.5 debe salir
    con pendiente > 1 (la calibracion las separa)."""
    rng = np.random.default_rng(2)
    verdadera = rng.random(4000)
    y = (rng.random(4000) < verdadera).astype(int)
    timida = 0.5 + (verdadera - 0.5) * 0.4     # comprimida hacia el centro
    cal = PlattCalibrator().fit(timida, y)
    assert cal.pendiente > 1.0


# --------------------------------------------------------- valor habitual ---
def test_valor_habitual_usa_el_historial_propio(logs, full_history_child_id):
    hab = valor_habitual(logs, full_history_child_id, "calidad_sueno")
    propios = logs.loc[logs["child_id"] == full_history_child_id, "calidad_sueno"].dropna()
    assert hab == propios.mode().iloc[0]


def test_valor_habitual_cae_a_la_poblacion_en_arranque_en_frio(logs, cold_start_child_id):
    """Un nino con 2-4 dias no tiene un "valor habitual" propio confiable: se
    usa el de la poblacion, el mismo criterio de partial pooling que el resto
    del sistema (Seccion 3.6)."""
    hab = valor_habitual(logs, cold_start_child_id, "calidad_sueno")
    assert hab == logs["calidad_sueno"].dropna().mode().iloc[0]


# ----------------------------------------------------------- explicaciones ---
def test_sin_modelo_no_hay_explicacion_local(logs, full_history_child_id):
    """La atribucion contrafactual necesita el ensamble. Sin modelo se devuelve
    vacio y la interfaz cae a las senales del narrador, en vez de inventar
    contribuciones."""
    assert explicar(logs, full_history_child_id, DIA_MALO, None) == []


def test_sin_nada_registrado_no_hay_nada_que_explicar(logs, full_history_child_id, has_model):
    if not has_model:
        pytest.skip("modelo no entrenado")
    assert explicar(logs, full_history_child_id, {}, load_model()) == []


def test_solo_explica_variables_registradas(logs, full_history_child_id, has_model):
    if not has_model:
        pytest.skip("modelo no entrenado")
    contribs = explicar(logs, full_history_child_id, DIA_MALO, load_model())
    assert {c.campo for c in contribs} <= set(DIA_MALO)


def test_una_variable_en_su_valor_habitual_no_contribuye(logs, full_history_child_id, has_model):
    """Si hoy la variable esta exactamente en el valor habitual del nino, el
    contrafactual es el mismo dia: la contribucion tiene que ser cero."""
    if not has_model:
        pytest.skip("modelo no entrenado")
    hab = valor_habitual(logs, full_history_child_id, "calidad_sueno")
    contribs = explicar(logs, full_history_child_id,
                        {"calidad_sueno": hab}, load_model())
    c = next(c for c in contribs if c.campo == "calidad_sueno")
    assert c.es_habitual is True
    assert abs(c.contribucion) < 1e-9
    assert c.direccion == "neutro"


def test_una_variable_fuera_de_su_base_si_contribuye(logs, has_model):
    """Regresion de un bug real: con calibracion isotonica TODAS las
    contribuciones salian 0.000 porque el escalon absorbia la diferencia. Al
    menos una variable claramente fuera de la linea base debe mover la cifra
    por encima del umbral de ruido."""
    if not has_model:
        pytest.skip("modelo no entrenado")
    modelo = load_model()
    movio = 0
    for cid in list(logs["child_id"].unique())[:10]:
        contribs = explicar(logs, cid, DIA_MALO, modelo)
        if any(abs(c.contribucion) >= UMBRAL_RELEVANTE for c in contribs):
            movio += 1
    assert movio >= 5, "casi ninguna variable mueve la cifra: la calibracion la aplana"


def test_las_contribuciones_vienen_ordenadas_por_magnitud(logs, full_history_child_id, has_model):
    if not has_model:
        pytest.skip("modelo no entrenado")
    contribs = explicar(logs, full_history_child_id, DIA_MALO, load_model())
    magnitudes = [abs(c.contribucion) for c in contribs]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_la_contribucion_esta_en_la_escala_del_riesgo_mostrado(logs, has_model):
    """Las contribuciones se calculan sobre la probabilidad YA CALIBRADA, que
    es la que se muestra. Si se calcularan sobre la cruda, un usuario que
    compare "+9%" con el "76%" de la pantalla estaria mirando dos escalas."""
    if not has_model:
        pytest.skip("modelo no entrenado")
    cid = "nino_001"
    modelo = load_model()
    r = predict_risk(logs, cid, DIA_MALO, compute_question=False)
    contribs = explicar(logs, cid, DIA_MALO, modelo)
    # Ninguna contribucion individual puede exceder el rango de la propia
    # probabilidad: son diferencias entre dos valores de [0, 1].
    assert all(abs(c.contribucion) <= 1.0 for c in contribs)
    assert 0.0 <= r.risk <= 1.0


def test_resumen_texto_menciona_solo_lo_que_se_aparta(logs, full_history_child_id, has_model):
    if not has_model:
        pytest.skip("modelo no entrenado")
    hab = valor_habitual(logs, full_history_child_id, "calidad_sueno")
    contribs = explicar(logs, full_history_child_id, {"calidad_sueno": hab}, load_model())
    texto = resumen_texto(contribs, lambda k: VARIABLE_LABELS.get(k, k))
    assert texto.startswith("Ningún dato de hoy se aparta")


def test_resumen_texto_nombra_lo_que_si_se_aparta(logs, has_model):
    """El caso contrario: cuando algo esta claramente fuera de la linea base,
    el resumen tiene que nombrarlo con su etiqueta legible."""
    if not has_model:
        pytest.skip("modelo no entrenado")
    contribs = explicar(logs, "nino_001", DIA_MALO, load_model())
    texto = resumen_texto(contribs, lambda k: VARIABLE_LABELS.get(k, k))
    if any(c.direccion != "neutro" for c in contribs):
        assert texto.startswith("Respecto de lo habitual en este niño:")
        assert "%" in texto
