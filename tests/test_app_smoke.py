"""
Prueba de humo de la interfaz completa, sin navegador.

POR QUE EXISTE
Los tests de `core/` verifican el motor, pero la interfaz es donde se juntan
todas las piezas -- y donde un renombre a medio aplicar (una variable que ya no
existe, una clave de diccionario que cambio) revienta la pantalla sin que
ningun test de core se entere. Comprobarlo a mano en Chrome es lento y, en esta
maquina, poco fiable: el renderer se cuelga con la pagina cargada aunque el
servidor de Streamlit este ocioso y respondiendo 200.

AppTest ejecuta el script de la app en el mismo proceso, con su runtime real,
y expone las excepciones que se hayan producido. Es la verificacion que de
verdad responde "¿se cae la app?".
"""
from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

# Ruta absoluta: AppTest resuelve las rutas relativas contra el archivo que la
# llama, no contra el directorio de trabajo, asi que "app.py" apuntaria a
# tests/app.py.
APP = Path(__file__).resolve().parents[1] / "app.py"

TIMEOUT = 180
VISTAS = ["👪  Hoy · familia", "🩺  Panel del equipo", "📝  Bitácora completa"]


def _correr(vista: str | None = None) -> AppTest:
    """Renderiza la app con una vista fija, en UN solo run().

    No se usa `at.segmented_control[0].set_value(...).run()` porque el segundo
    run obliga a AppTest a reconstruir el estado de todos los widgets, y ahi
    tropieza con el selectbox del consultante: AppTest no conserva su
    `format_func`, asi que busca el valor crudo ('nino_001') dentro de la lista
    de etiquetas formateadas ('Amelia S. - 150 dias') y lanza ValueError. Es una
    limitacion del arnes de pruebas, no de la app.

    Sembrar session_state antes del primer run evita el problema entero y ademas
    es mas rapido: una sola pasada por vista.
    """
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    if vista is not None:
        at.session_state["vista"] = vista
    at.run()
    return at


def _sin_excepciones(at: AppTest, contexto: str) -> None:
    if at.exception:
        detalle = "\n".join(str(e.value) for e in at.exception)
        pytest.fail(f"la app lanzo una excepcion en {contexto}:\n{detalle}")


def test_la_app_arranca_sin_excepciones():
    _sin_excepciones(_correr(), "el arranque")


@pytest.mark.parametrize("vista", VISTAS)
def test_cada_vista_renderiza(vista):
    """Las tres vistas se dibujan enteras. Cubre en particular el panel del
    equipo, que es el bloque mas grande y el que mas piezas nuevas junta:
    trazabilidad de recomendaciones, exclusiones y el agregado de seguimiento."""
    _sin_excepciones(_correr(vista), f"la vista {vista!r}")


def test_el_panel_del_equipo_muestra_la_trazabilidad():
    """Seccion 12: de cada sugerencia se tiene que poder ver de donde sale."""
    at = _correr("🩺  Panel del equipo")
    _sin_excepciones(at, "el panel del equipo")
    texto = " ".join(m.value for m in at.markdown)
    assert "De dónde sale cada sugerencia" in texto


def test_el_panel_del_equipo_muestra_el_seguimiento():
    """Seccion 13: el agregado de "que ocurrio despues" existe aunque todavia
    no haya ningun seguimiento registrado (en ese caso, explicando como se
    registra en vez de mostrar una tabla vacia)."""
    at = _correr("🩺  Panel del equipo")
    _sin_excepciones(at, "el panel del equipo")
    texto = " ".join(m.value for m in at.markdown)
    assert "Qué ocurrió después" in texto


def test_la_vista_familia_ofrece_registrar_el_resultado():
    """Seccion 13, lado de captura: el formulario "¿Que ocurrio despues?" tiene
    que estar en la pantalla de la familia, que es quien lo sabe."""
    at = _correr("👪  Hoy · familia")
    _sin_excepciones(at, "la vista de familia")
    etiquetas = [e.label for e in at.expander]
    assert any("¿Qué ocurrió después?" in e for e in etiquetas)


def test_no_queda_la_palabra_confianza_en_la_interfaz():
    """Regresion de la revision metodologica: lo que se muestra es un indice de
    suficiencia de informacion, no una confianza estadistica. Si la palabra
    reaparece en pantalla, el renombre se quedo a medias."""
    for vista in VISTAS:
        at = _correr(vista)
        _sin_excepciones(at, f"la vista {vista!r}")
        piezas = [m.value for m in at.markdown]
        piezas += [c.value for c in at.caption]
        texto = " ".join(piezas).lower()
        assert "confianza" not in texto, f"aparece 'confianza' en {vista!r}"


def test_el_registro_de_episodio_pide_texto_libre():
    """El esquema real de Bluba tiene dos campos de TEXTO LIBRE que el
    prototipo no capturaba: que desregulo al nino (`detonante_gatillante`) y
    que se hizo (`estrategia_calma_aplicada`). Un detonante como "etiqueta
    molesta en una prenda nueva" no cabe en una lista cerrada, y obligar a
    elegir una opcion destruiria justo la informacion que hace distinto a cada
    nino."""
    at = _correr("📝  Bitácora completa")
    _sin_excepciones(at, "la bitacora completa")
    etiquetas = [t.label for t in at.text_area]
    assert any("¿Qué lo desreguló?" in e for e in etiquetas), etiquetas
    assert any("estrategia o apoyo se aplicó" in e for e in etiquetas), etiquetas


def test_la_intensidad_usa_las_bandas_reales():
    """Bluba registra la intensidad en tres bandas, no en una escala continua.
    Un deslizador de 0 a 10 prometia una precision que el dato de origen no
    tiene."""
    at = _correr("📝  Bitácora completa")
    _sin_excepciones(at, "la bitacora completa")
    opciones = [o for sb in at.selectbox for o in (sb.options or [])]
    for banda in ("Leve (1-3)", "Moderada (4-7)", "Severa (8-10)"):
        assert any(banda in str(o) for o in opciones), f"falta la banda {banda}"
    assert len(at.slider) == 0, "quedo un deslizador de intensidad"


def test_la_trazabilidad_no_contradice_a_la_sugerencia():
    """Regresion de un bug visible en pantalla.

    El panel del equipo leia `result.drivers` (solo variables registradas HOY
    con desviacion alta) mientras la sugerencia que ve la familia se arma con
    las senales de la narrativa (ventana de 3 dias). En un dia en blanco eso
    daba una contradiccion directa: la familia recibia "considere anticipar una
    rutina de sueno mas temprana" y el panel del equipo, en la misma pantalla,
    decia "Hoy no se activo ninguna regla de la biblioteca".

    Si hay sugerencia, tiene que haber reglas que mostrar.
    """
    import pandas as pd
    from core.risk_model import predict_risk
    from core.narrative import build_narrative, claves_candidatas
    from core.recommendations import (
        DEFAULT_RECOMMENDATION, recomendaciones_activadas)

    logs = pd.read_csv(APP.parent / "data" / "bitacoras.csv", parse_dates=["date"])
    escenarios = [{}, {"participacion_actividades": "No participa"}]
    for hoy in escenarios:
        for cid in list(logs["child_id"].unique())[:6]:
            r = predict_risk(logs, cid, hoy, compute_question=False)
            n = build_narrative(logs, cid, hoy, r, nombre=cid, audiencia="profesional")
            activadas = recomendaciones_activadas(
                claves_candidatas(n.senales, r.drivers))
            propone_algo = DEFAULT_RECOMMENDATION not in n.sugerencia
            if propone_algo:
                assert activadas, (
                    f"{cid}: la familia lee una sugerencia pero el panel no "
                    f"muestra ninguna regla activada")
                # Y lo que llego al texto tiene que estar entre lo activado.
                ids = {x.id for x in activadas}
                from core.recommendations import BIBLIOTECA
                for clave in n.reglas_aplicadas:
                    assert BIBLIOTECA[clave].id in ids


def test_el_panel_marca_que_reglas_llegaron_al_texto():
    """El distintivo «en la sugerencia de hoy» separa las reglas que se
    activaron de las que ademas entraron en el parrafo que lee la familia
    (el texto se limita a 2-3 acciones). Se comparaba r.id ('REC-01') contra
    un conjunto de CLAVES de variable ('calidad_sueno'), asi que no coincidia
    nunca y el distintivo no salia jamas."""
    at = _correr("🩺  Panel del equipo")
    _sin_excepciones(at, "el panel del equipo")
    texto = " ".join(m.value for m in at.markdown)
    assert "EN LA SUGERENCIA DE HOY" in texto
