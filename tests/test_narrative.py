"""
Tests de la explicacion narrativa (core/narrative.py, Seccion 7).

Lo que se verifica aqui no es "el texto suena bien" sino que el texto NO
contradiga al motor: es una explicacion clinica y una explicacion que afirma
algo que los datos no sostienen es peor que no tener explicacion.
"""

import pandas as pd
import pytest

from core.narrative import (UMBRAL_ELEVADO, UMBRAL_MODERADO, build_narrative,
                            _unir)
from core.risk_model import predict_risk

DIA_MALO = {
    "calidad_sueno": "Dificultad de Conciliacion",
    "modo_despertar": "Irritable/Llorando",
    "adherencia_medicacion": "No",
    "estado_gastrointestinal": "Diarrea",
    "nivel_regulacion_general_dia": "Desregulacion Frecuente",
    "cambios_rutina": "Si",
    "comportamiento_observado": "Desregulado",
    "estado_alerta": "Alto (Sobreexcitado)",
    "n_eventos_desregulacion": 2,
}

DIA_BUENO = {
    "calidad_sueno": "Reparador",
    "modo_despertar": "Tranquilo/Alegre",
    "adherencia_medicacion": "Si",
    "estado_gastrointestinal": "Normal",
    "nivel_regulacion_general_dia": "Excelente",
    "cambios_rutina": "No",
    "comportamiento_observado": "Estable",
    "estado_alerta": "Optimo (Regulado)",
    "n_eventos_desregulacion": 0,
    "nivel_apoyo_requerido": "Bajo",
    "interacciones_sociales": "Normal",
    "participacion_actividades": "Completa",
    "cambios_alimentacion": "Sin cambios",
    "alimentacion_recreos": "Normal",
}


def _narrar(logs, child_id, today, **kw):
    r = predict_risk(logs, child_id, today, compute_question=False)
    return r, build_narrative(logs, child_id, today, r, nombre="Amelia", **kw)


def test_estructura_de_tres_frases_del_ejemplo_de_las_bases(logs, full_history_child_id):
    """Las bases dan el formato esperado: observacion -> veredicto -> sugerencia."""
    _, n = _narrar(logs, full_history_child_id, DIA_MALO)
    assert n.observacion.startswith("Durante los últimos")
    assert "Amelia" in n.observacion
    assert "24 horas" in n.veredicto
    assert n.sugerencia
    for parte in (n.observacion, n.veredicto, n.sugerencia):
        assert parte.rstrip().endswith("."), parte
        assert parte in n.texto


def test_el_nivel_verbal_coincide_con_el_numero(logs, full_history_child_id):
    """El adjetivo del texto no puede contradecir el porcentaje del anillo."""
    for today in (DIA_MALO, DIA_BUENO, {}):
        r, n = _narrar(logs, full_history_child_id, today)
        if r.risk >= UMBRAL_ELEVADO:
            assert n.nivel == "elevado"
        elif r.risk >= UMBRAL_MODERADO:
            assert n.nivel == "moderado"
        else:
            assert n.nivel == "bajo"


def test_suficiencia_baja_no_afirma_un_riesgo_como_hecho(logs, full_history_child_id):
    """Requisito explicito de las bases: alertar cuando la informacion no
    alcanza, en vez de emitir una recomendacion que parezca confiable."""
    r, n = _narrar(logs, full_history_child_id, {})   # nada registrado hoy
    assert r.sufficiency_level == "baja"
    assert n.preliminar is True
    assert "no hay información suficiente" in n.veredicto.lower()
    assert "preliminar" in n.veredicto.lower()
    assert n.salvedad


def test_no_pide_registrar_algo_si_la_interfaz_ya_cerro_el_dia(logs, full_history_child_id):
    """Con el registro del dia cerrado (pregunta_pendiente=None) la sugerencia
    no puede seguir diciendo 'registre X': el motor siempre tiene una pregunta
    candidata, pero la interfaz decide si la esta pidiendo."""
    _, n = _narrar(logs, full_history_child_id, {"calidad_sueno": "Interrumpido"},
                   pregunta_pendiente=None)
    assert n.preliminar is True
    assert not n.sugerencia.lower().startswith("registre")

    _, con = _narrar(logs, full_history_child_id, {"calidad_sueno": "Interrumpido"},
                     pregunta_pendiente="nivel_regulacion_general_dia")
    assert con.sugerencia.lower().startswith("registre")


def test_las_senales_declaran_su_origen(logs, full_history_child_id):
    """Los datos vienen de familia, colegio y equipo profesional; la frase y la
    evidencia tienen que decir de cual."""
    _, n = _narrar(logs, full_history_child_id, DIA_MALO)
    origenes = {s.origen for s in n.senales}
    assert origenes <= {"familia", "colegio", "profesional"}
    colegio = [s for s in n.senales if s.origen == "colegio"]
    if colegio:
        assert any("en el colegio" in s.frase for s in colegio)


def test_senal_activa_hoy_pesa_mas_que_una_ya_resuelta(logs, full_history_child_id):
    """Se predicen las proximas 24 h: una senal que sigue presente hoy debe
    ordenarse antes que la misma senal si ya se resolvio."""
    _, hoy = _narrar(logs, full_history_child_id, {"comportamiento_observado": "Desregulado"})
    _, ayer = _narrar(logs, full_history_child_id, {"comportamiento_observado": "Estable"})

    def score(n, key):
        return next((s.score for s in n.senales if s.key == key), 0.0)

    assert score(hoy, "comportamiento_observado") > score(ayer, "comportamiento_observado")


def test_dia_limpio_tras_dias_malos_se_hace_explicito(logs, full_history_child_id):
    """Si las senales de la ventana ya no estan hoy, el texto lo dice: omitirlo
    haria sonar el presente peor de lo que los datos sostienen."""
    _, n = _narrar(logs, full_history_child_id, DIA_BUENO)
    if n.senales and not any(s.presente_hoy for s in n.senales[:3]):
        assert "hoy no muestra esas señales" in n.observacion


def test_arranque_en_frio_lo_explica_en_vez_de_inventar(logs, cold_start_child_id):
    """Un nino casi sin historial no debe recibir una afirmacion tajante."""
    r, n = _narrar(logs, cold_start_child_id, {"calidad_sueno": "Interrumpido"})
    assert r.n_history_days < 7
    assert n.preliminar is True
    assert "arranque en frío" in (n.salvedad or "")


def test_audiencia_profesional_agrega_cifras_y_familia_no(logs, full_history_child_id):
    _, fam = _narrar(logs, full_history_child_id, DIA_MALO, audiencia="familia")
    _, pro = _narrar(logs, full_history_child_id, DIA_MALO, audiencia="profesional")
    assert fam.ficha_tecnica is None
    assert pro.ficha_tecnica is not None
    assert "información" in pro.ficha_tecnica
    assert "%" in pro.ficha_tecnica
    # La narrativa en si es la misma para ambos: cambia el anexo, no los hechos.
    assert fam.texto == pro.texto


def test_no_falla_con_ningun_nino_ni_registro_vacio(logs):
    """Barrido: la explicacion es parte de la pantalla principal, no puede
    lanzar una excepcion para ninguna combinacion nino/registro."""
    for cid in logs["child_id"].unique():
        for today in ({}, DIA_BUENO, DIA_MALO):
            _, n = _narrar(logs, cid, today)
            assert n.texto and n.texto[0].isupper() and n.texto.endswith(".")


def test_nino_sin_ninguna_fila_en_los_datos(logs):
    """Arranque en frio total (Seccion 3.6): el nino no existe en el CSV."""
    _, n = _narrar(logs, "nino_que_no_existe", {"calidad_sueno": "Interrumpido"})
    assert n.ventana == 1
    assert n.observacion.startswith("Hoy,")


@pytest.mark.parametrize("entrada,esperado", [
    ([], ""),
    (["a"], "a"),
    (["a", "b"], "a y b"),
    (["a", "b", "c"], "a, b y c"),
])
def test_coordinacion_en_castellano(entrada, esperado):
    assert _unir(entrada) == esperado
