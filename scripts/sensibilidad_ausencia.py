"""
Sensibilidad al mecanismo de ausencia de datos -- Seccion 4.4 / 7.

POR QUE EXISTE ESTE SCRIPT
La propuesta original afirmaba que "en la practica clinica un vacio de
informacion suele coincidir con los momentos de mayor dificultad para la
familia", y a partir de ahi le daba un papel central al mecanismo MNAR. La
revision metodologica de agosto 2026 senalo el problema: es plausible, pero con
la evidencia disponible no esta demostrado como regla general. Y hay algo peor
que una afirmacion no demostrada: el generador sintetico la horneaba en los
datos, asi que el modelo la aprendia y el resultado parecia confirmarla. Eso es
circular -- se estaba midiendo el supuesto, no el mundo.

La posicion corregida es la prudente:

    "La ausencia de registro PUEDE ser informativa, por lo que Antesala
     conserva explicitamente indicadores de ausencia y antiguedad en lugar de
     asumir que los datos faltantes son neutrales."

Este script convierte esa frase en algo verificable. Genera el mismo mundo bajo
los cuatro mecanismos y entrena el mismo modelo en cada uno:

    mcar    la ausencia es puramente aleatoria: el silencio no dice nada.
    mar     la ausencia depende de cosas observables (fin de semana, colegio).
    mnar    la ausencia depende del resultado: se registra menos los dias malos.
    mixto   los tres a la vez (es el dataset del repositorio).

LO QUE HAY QUE MIRAR
Si el rendimiento se derrumba bajo MCAR, el sistema depende de que el silencio
sea informativo, y entonces todo descansa sobre un supuesto no verificado. Si
se sostiene parecido en los cuatro, el modelo esta leyendo contenido clinico y
el mecanismo de ausencia es una variable de contexto, no su fuente de senal.

Uso:
    python scripts/sensibilidad_ausencia.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.features import (
    build_features, population_baselines,
    FEATURE_NUMERIC, FEATURE_CATEGORICAL, TARGET,
)
from core.question_selector import ASKABLE_FIELDS
from core.train_model import build_pipeline
from core import evaluation as ev

MECANISMOS = ["mcar", "mar", "mnar", "mixto"]


def generar(mecanismo: str, destino: Path) -> pd.DataFrame:
    subprocess.run(
        [sys.executable, str(ROOT / "data" / "generate_synthetic_data.py"),
         "--out", str(destino), "--missingness", mecanismo],
        check=True, capture_output=True)
    return pd.read_csv(destino, parse_dates=["date"])


def evaluar(logs: pd.DataFrame) -> tuple[dict, float, float]:
    mu = population_baselines(logs)
    feat = build_features(logs, mu=mu).reset_index(drop=True)
    X = feat[FEATURE_NUMERIC + FEATURE_CATEGORICAL]
    y = feat[TARGET].astype(int)
    tr, te = ev.split_por_nino(feat)

    pipe = build_pipeline()
    pipe.fit(X.iloc[tr], y.iloc[tr])
    proba = pipe.predict_proba(X.iloc[te])[:, 1]
    sub = feat.iloc[te]
    m = ev.panel(y.iloc[te], proba, groups=sub["child_id"],
                 dias=sub.groupby("child_id")["date"].nunique().mean())

    campos = [c for c in ASKABLE_FIELDS if c in logs.columns]
    pct_faltante = float(logs[campos].isna().mean().mean())

    # Cuanto de la importancia total del modelo se la llevan los indicadores de
    # ausencia: si es alta, el modelo esta leyendo el silencio mas que el dato.
    nombres = pipe.named_steps["pre"].get_feature_names_out()
    imps = pipe.named_steps["rf"].feature_importances_
    peso_ausencia = float(sum(
        i for n, i in zip(nombres, imps)
        if "missingindicator" in n or n.endswith("___missing__")))
    return m, pct_faltante, peso_ausencia


def main():
    print("SENSIBILIDAD AL MECANISMO DE AUSENCIA (Seccion 4.4)")
    print("Mismo mundo, mismo modelo, distinto mecanismo de datos faltantes.")
    print("=" * 96)
    print(f"{'mecanismo':<10s} {'% faltante':>10s} {'peso ausencia':>14s}   " + ev.CABECERA_PANEL[26:])
    print("-" * 96)

    filas = []
    with tempfile.TemporaryDirectory() as tmp:
        for mecanismo in MECANISMOS:
            destino = Path(tmp) / f"bitacoras_{mecanismo}.csv"
            logs = generar(mecanismo, destino)
            m, pct, peso = evaluar(logs)
            filas.append((mecanismo, pct, peso, m))
            print(f"{mecanismo:<10s} {pct:>9.1%} {peso:>13.3f}   "
                  + ev.formato_panel("", m)[26:])

    print("-" * 96)
    print("COMO LEERLO")
    aurocs = {mec: m["auroc"] for mec, _, _, m in filas}
    brecha = max(aurocs.values()) - min(aurocs.values())
    print(f"  Brecha de AUROC entre el mejor y el peor mecanismo: {brecha:.3f}")
    print(f"  MCAR (el silencio no informa): AUROC {aurocs['mcar']:.3f}")
    print(f"  MNAR (el silencio informa mucho): AUROC {aurocs['mnar']:.3f}")
    peso_mnar = next(pe for mec, _, pe, _ in filas if mec == "mnar")
    peso_mcar = next(pe for mec, _, pe, _ in filas if mec == "mcar")
    if brecha < 0.06:
        print("  -> El rendimiento NO depende de que la ausencia sea informativa.")
        print("     El modelo esta leyendo contenido clinico, no el patron de silencio.")
    else:
        print("  -> El rendimiento SI depende del mecanismo de ausencia, y bastante.")
        print(f"     Bajo MNAR fuerte los indicadores de ausencia se llevan el "
              f"{peso_mnar:.0%} de la")
        print(f"     importancia del modelo (bajo MCAR, {peso_mcar:.0%}): el sistema deja de")
        print("     leer al nino y pasa a leer si alguien abrio la app.")
    print()
    print("  LA CIFRA QUE HAY QUE CITAR")
    print(f"     Presentar el {aurocs['mnar']:.3f} de MNAR seria presentar el supuesto como")
    print("     resultado: ese numero se lo gana el generador, no el modelo. La cifra")
    print("     honesta es el PISO, el escenario en que la ausencia no aporta nada:")
    print(f"        AUROC {aurocs['mcar']:.3f} bajo MCAR")
    print("     Todo lo que el sistema logre por encima de eso es un extra que solo se")
    print("     puede reclamar despues de comprobar el mecanismo con datos de Bluba.")
    print()
    print("  'peso ausencia' = fraccion de la importancia total del modelo que se")
    print("  llevan los indicadores de dato faltante. Sube bajo MNAR por construccion:")
    print("  ahi el silencio SI es senal, y el modelo la usa. Que suba es correcto;")
    print("  lo que no seria correcto es que el sistema NECESITE que suba.")
    print()
    print("  ALCANCE: datos sinteticos. Esto mide la sensibilidad del pipeline a un")
    print("  supuesto, no cual de los cuatro mecanismos opera en la realidad. Eso solo")
    print("  lo pueden decir los datos longitudinales de Bluba.")


if __name__ == "__main__":
    main()
