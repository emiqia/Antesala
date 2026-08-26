"""
Registro del resultado de la intervencion -- Seccion 13 del documento tecnico.

    "Despues de una recomendacion, el sistema podra registrar: si aparecio o
     no una desregulacion; que apoyo se utilizo; si fue aceptado; si parecio
     util; si genero dificultades.
     Esto permite que futuras versiones estudien no solo 'que predice una
     crisis' sino tambien 'que apoyo parece funcionar para esta persona y en
     que contexto'."

POR QUE ESTO IMPORTA MAS DE LO QUE PARECE
Sin este registro, Antesala es un sistema de una sola direccion: predice,
sugiere, y nunca se entera de si sirvio. Es tambien el unico mecanismo por el
que el prototipo puede generar datos que hoy no existen en ninguna parte: la
bitacora de Bluba registra lo que le pasa al nino, no que hizo el adulto
despues de recibir un aviso ni si funciono. El pareo aviso -> apoyo ->
resultado es exactamente el dato que hara falta para las fases 3 y 4 de la
validacion (Seccion 17), y no se puede reconstruir despues.

Tambien cierra el circuito etico: una recomendacion que nadie evalua es una
instruccion; una que se evalua y se puede apagar es apoyo a la decision.

DONDE SE GUARDA
SQLite (Seccion 20 del stack), en data/antesala.db. Se eligio por lo mismo que
el documento: cero configuracion, un archivo, y transaccional -- si la app se
cae a mitad de un guardado no queda media fila escrita, que es justo lo que
pasaria con un CSV.

MINIMIZACION DE DATOS (Seccion 19)
Este registro guarda deliberadamente lo minimo: el identificador seudonimo del
nino, la fecha, las cifras que el sistema mostro y las respuestas del adulto.
No guarda nombres, ni texto libre sobre terceros, ni nada que permita
reidentificar. El campo de notas existe porque un adulto necesita poder
matizar, y por eso mismo la interfaz advierte que no se escriban datos
personales ahi.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "antesala.db"

ESQUEMA = """
CREATE TABLE IF NOT EXISTS seguimiento (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    child_id            TEXT    NOT NULL,
    fecha               TEXT    NOT NULL,   -- dia al que se refiere el aviso
    riesgo_estimado     REAL,               -- lo que el sistema mostro
    suficiencia         REAL,
    recomendaciones     TEXT,               -- ids separados por coma: trazabilidad
    hubo_desregulacion  INTEGER,            -- 0/1/NULL(no se sabe)
    apoyo_usado         TEXT,
    fue_aceptado        INTEGER,            -- 0/1/NULL
    parecio_util        INTEGER,            -- 0/1/NULL
    genero_dificultades INTEGER,            -- 0/1/NULL
    notas               TEXT,
    registrado_en       TEXT    NOT NULL,
    UNIQUE (child_id, fecha)
);

CREATE TABLE IF NOT EXISTS exclusiones (
    child_id        TEXT NOT NULL,
    recomendacion   TEXT NOT NULL,
    motivo          TEXT,
    registrado_en   TEXT NOT NULL,
    PRIMARY KEY (child_id, recomendacion)
);
"""

APOYOS = [
    "Anticipación con apoyos visuales",
    "Reducción de demandas",
    "Espacio de regulación",
    "Ajuste sensorial (ruido/luz)",
    "Acompañamiento en transiciones",
    "Otro apoyo",
    "No se aplicó ningún apoyo",
]


@dataclass
class Seguimiento:
    child_id: str
    fecha: str
    riesgo_estimado: float | None = None
    suficiencia: float | None = None
    recomendaciones: str | None = None
    hubo_desregulacion: int | None = None
    apoyo_usado: str | None = None
    fue_aceptado: int | None = None
    parecio_util: int | None = None
    genero_dificultades: int | None = None
    notas: str | None = None


@contextmanager
def _conexion(db_path: Path | None = None):
    """Una conexion por operacion.

    Streamlit atiende cada interaccion en un hilo distinto y una conexion de
    SQLite no se puede compartir entre hilos; abrir y cerrar por operacion
    evita ese problema entero sin necesidad de un pool. El volumen aqui son
    unas pocas filas por dia, asi que el costo es irrelevante.
    """
    ruta = Path(db_path) if db_path else DB_PATH
    ruta.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(ruta)
    try:
        con.row_factory = sqlite3.Row
        con.executescript(ESQUEMA)
        yield con
        con.commit()
    finally:
        con.close()


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------- seguimiento (S.13) --
def registrar(seg: Seguimiento, db_path: Path | None = None) -> None:
    """Guarda (o reemplaza) el seguimiento de un nino para un dia.

    UPSERT y no INSERT: si el adulto corrige lo que respondio, tiene que
    quedar una sola verdad para ese dia. Dos filas contradictorias para el
    mismo nino-dia envenenarian cualquier analisis posterior de que apoyo
    funciono.
    """
    with _conexion(db_path) as con:
        con.execute("""
            INSERT INTO seguimiento (child_id, fecha, riesgo_estimado, suficiencia,
                recomendaciones, hubo_desregulacion, apoyo_usado, fue_aceptado,
                parecio_util, genero_dificultades, notas, registrado_en)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(child_id, fecha) DO UPDATE SET
                riesgo_estimado=excluded.riesgo_estimado,
                suficiencia=excluded.suficiencia,
                recomendaciones=excluded.recomendaciones,
                hubo_desregulacion=excluded.hubo_desregulacion,
                apoyo_usado=excluded.apoyo_usado,
                fue_aceptado=excluded.fue_aceptado,
                parecio_util=excluded.parecio_util,
                genero_dificultades=excluded.genero_dificultades,
                notas=excluded.notas,
                registrado_en=excluded.registrado_en
        """, (seg.child_id, seg.fecha, seg.riesgo_estimado, seg.suficiencia,
              seg.recomendaciones, seg.hubo_desregulacion, seg.apoyo_usado,
              seg.fue_aceptado, seg.parecio_util, seg.genero_dificultades,
              seg.notas, _ahora()))


def seguimiento_de(child_id: str, fecha: str, db_path: Path | None = None) -> dict | None:
    with _conexion(db_path) as con:
        fila = con.execute(
            "SELECT * FROM seguimiento WHERE child_id=? AND fecha=?",
            (child_id, fecha)).fetchone()
    return dict(fila) if fila else None


def historial(child_id: str | None = None, db_path: Path | None = None) -> pd.DataFrame:
    with _conexion(db_path) as con:
        if child_id:
            filas = con.execute(
                "SELECT * FROM seguimiento WHERE child_id=? ORDER BY fecha DESC",
                (child_id,)).fetchall()
        else:
            filas = con.execute(
                "SELECT * FROM seguimiento ORDER BY fecha DESC").fetchall()
    return pd.DataFrame([dict(f) for f in filas])


def resumen_por_apoyo(child_id: str | None = None,
                      db_path: Path | None = None) -> pd.DataFrame:
    """"Que apoyo parece funcionar para esta persona" -- el objetivo declarado
    de la Seccion 13.

    IMPORTANTE, y hay que decirlo en pantalla: esto es una TABULACION
    DESCRIPTIVA, no evidencia de que el apoyo cause el resultado. Los apoyos no
    se asignan al azar -- se eligen justamente en los dias que pintan peor --
    asi que el apoyo que aparezca con mas desregulaciones puede ser el que se
    usa en los dias mas dificiles, no el que funciona peor. Leerlo como
    efectividad seria invertir la causa.
    """
    df = historial(child_id, db_path)
    if df.empty or "apoyo_usado" not in df.columns:
        return pd.DataFrame()
    df = df[df["apoyo_usado"].notna()]
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("apoyo_usado").agg(
        veces=("apoyo_usado", "size"),
        desregulaciones=("hubo_desregulacion", "mean"),
        aceptado=("fue_aceptado", "mean"),
        parecio_util=("parecio_util", "mean"),
        dio_problemas=("genero_dificultades", "mean"),
    ).reset_index()
    return g.sort_values("veces", ascending=False)


# ------------------------------------------------------- exclusiones (S.19) --
def excluir(child_id: str, recomendacion_id: str, motivo: str = "",
            db_path: Path | None = None) -> None:
    """Apaga una recomendacion para un nino concreto."""
    with _conexion(db_path) as con:
        con.execute("""
            INSERT INTO exclusiones (child_id, recomendacion, motivo, registrado_en)
            VALUES (?,?,?,?)
            ON CONFLICT(child_id, recomendacion) DO UPDATE SET
                motivo=excluded.motivo, registrado_en=excluded.registrado_en
        """, (child_id, recomendacion_id, motivo, _ahora()))


def restaurar(child_id: str, recomendacion_id: str, db_path: Path | None = None) -> None:
    with _conexion(db_path) as con:
        con.execute("DELETE FROM exclusiones WHERE child_id=? AND recomendacion=?",
                    (child_id, recomendacion_id))


def exclusiones_de(child_id: str, db_path: Path | None = None) -> dict[str, str]:
    """{id_recomendacion: motivo} de las apagadas para este nino."""
    with _conexion(db_path) as con:
        filas = con.execute(
            "SELECT recomendacion, motivo FROM exclusiones WHERE child_id=?",
            (child_id,)).fetchall()
    return {f["recomendacion"]: (f["motivo"] or "") for f in filas}
