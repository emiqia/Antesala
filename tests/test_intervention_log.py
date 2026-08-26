"""
Prueba la biblioteca auditable de recomendaciones (Seccion 12), la exclusion
por nino (Seccion 19) y el registro del resultado de la intervencion
(Seccion 13).
"""
import pytest

from core import intervention_log as il
from core.recommendations import (
    BIBLIOTECA, DEFAULT_RECOMMENDATION, RECOMMENDATIONS,
    build_recommendation_text, recomendaciones_activadas,
)


@pytest.fixture
def db(tmp_path):
    return tmp_path / "test.db"


# ------------------------------------------------- biblioteca (Seccion 12) ---
def test_cada_entrada_declara_los_cinco_campos():
    """La Seccion 12 los exige uno por uno: condicion de activacion, accion,
    contexto, fuente/revision responsable y posibilidad de exclusion."""
    for driver, r in BIBLIOTECA.items():
        assert r.id.startswith("REC-")
        assert r.driver == driver
        assert r.condicion.strip() and r.condicion.endswith(".")
        assert r.accion.strip()
        assert r.contexto.strip()
        assert r.fuente.strip()
        assert r.estado_revision in ("pendiente", "revisada")
        assert isinstance(r.excluible, bool)


def test_los_ids_son_unicos():
    ids = [r.id for r in BIBLIOTECA.values()]
    assert len(ids) == len(set(ids))


def test_ninguna_entrada_se_declara_revisada_sin_firma():
    """Marcar algo como revisado sin decir quien lo reviso destruye justo lo
    que hace auditable a la biblioteca."""
    for r in BIBLIOTECA.values():
        if r.estado_revision == "revisada":
            assert r.revisada_por, f"{r.id} dice estar revisada pero no dice por quien"


def test_el_mapa_plano_deriva_de_la_biblioteca():
    """RECOMMENDATIONS existe por compatibilidad, pero no puede divergir: si
    fueran dos tablas escritas a mano, editar una y no la otra haria que la
    pantalla y el parrafo dijeran cosas distintas."""
    assert RECOMMENDATIONS == {d: r.accion for d, r in BIBLIOTECA.items()}


def test_las_entradas_de_seguridad_no_son_excluibles():
    """Verificar la administracion de un medicamento y derivar al equipo
    tratante no son preferencias de estilo."""
    assert BIBLIOTECA["adherencia_medicacion"].excluible is False
    assert BIBLIOTECA["nivel_alerta_sesion"].excluible is False


# --------------------------------------------------- exclusion (Seccion 19) --
def test_excluir_saca_la_accion_del_texto():
    completo = build_recommendation_text(["calidad_sueno", "cambios_rutina"])
    filtrado = build_recommendation_text(["calidad_sueno", "cambios_rutina"],
                                         excluidas={"REC-01"})
    assert BIBLIOTECA["calidad_sueno"].accion in completo
    assert BIBLIOTECA["calidad_sueno"].accion not in filtrado
    assert BIBLIOTECA["cambios_rutina"].accion in filtrado


def test_excluir_todo_cae_a_la_recomendacion_por_defecto():
    texto = build_recommendation_text(["calidad_sueno"], excluidas={"REC-01"})
    assert texto == DEFAULT_RECOMMENDATION


def test_no_se_puede_excluir_una_entrada_de_seguridad():
    texto = build_recommendation_text(["adherencia_medicacion"], excluidas={"REC-06"})
    assert BIBLIOTECA["adherencia_medicacion"].accion in texto


def test_recomendaciones_activadas_respeta_la_exclusion():
    activadas = recomendaciones_activadas(["calidad_sueno", "cambios_rutina"],
                                          excluidas={"REC-01"})
    assert [r.id for r in activadas] == ["REC-09"]


def test_persistencia_de_exclusiones(db):
    assert il.exclusiones_de("nino_x", db) == {}
    il.excluir("nino_x", "REC-01", "ya se probó y no funcionó", db)
    assert il.exclusiones_de("nino_x", db) == {"REC-01": "ya se probó y no funcionó"}
    # No debe afectar a otros ninos.
    assert il.exclusiones_de("nino_y", db) == {}
    il.restaurar("nino_x", "REC-01", db)
    assert il.exclusiones_de("nino_x", db) == {}


# ------------------------------------------------- seguimiento (Seccion 13) --
def test_registrar_y_leer(db):
    il.registrar(il.Seguimiento(
        child_id="nino_x", fecha="2026-08-26", riesgo_estimado=0.7,
        hubo_desregulacion=1, apoyo_usado="Reducción de demandas",
        fue_aceptado=1, parecio_util=0, genero_dificultades=0), db)
    fila = il.seguimiento_de("nino_x", "2026-08-26", db)
    assert fila["hubo_desregulacion"] == 1
    assert fila["apoyo_usado"] == "Reducción de demandas"
    assert fila["parecio_util"] == 0


def test_corregir_no_duplica_el_dia(db):
    """UPSERT y no INSERT: dos filas contradictorias para el mismo nino-dia
    envenenarian cualquier analisis posterior de que apoyo funciono."""
    for hubo in (1, 0):
        il.registrar(il.Seguimiento(child_id="nino_x", fecha="2026-08-26",
                                    hubo_desregulacion=hubo,
                                    apoyo_usado="Espacio de regulación"), db)
    h = il.historial("nino_x", db)
    assert len(h) == 1
    assert h.iloc[0]["hubo_desregulacion"] == 0


def test_historial_separa_por_nino(db):
    il.registrar(il.Seguimiento("nino_a", "2026-08-26", apoyo_usado="Otro apoyo"), db)
    il.registrar(il.Seguimiento("nino_b", "2026-08-26", apoyo_usado="Otro apoyo"), db)
    assert len(il.historial("nino_a", db)) == 1
    assert len(il.historial(None, db)) == 2


def test_resumen_por_apoyo_agrupa(db):
    for fecha, util in [("2026-08-24", 1), ("2026-08-25", 1), ("2026-08-26", 0)]:
        il.registrar(il.Seguimiento("nino_x", fecha, apoyo_usado="Reducción de demandas",
                                    parecio_util=util, hubo_desregulacion=0), db)
    r = il.resumen_por_apoyo("nino_x", db)
    assert len(r) == 1
    assert r.iloc[0]["veces"] == 3
    assert r.iloc[0]["parecio_util"] == pytest.approx(2 / 3)


def test_resumen_vacio_sin_datos(db):
    assert il.resumen_por_apoyo("nadie", db).empty


def test_los_apoyos_incluyen_no_haber_hecho_nada():
    """Sin esta opcion, el registro obliga a declarar un apoyo que quiza no se
    aplico, y el agregado de la Seccion 13 quedaria sesgado desde el origen."""
    assert any("No se aplicó" in a for a in il.APOYOS)
