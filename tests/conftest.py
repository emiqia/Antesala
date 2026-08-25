import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "bitacoras.csv"
MODEL_PATH = ROOT / "models" / "antesala_rf.joblib"


@pytest.fixture(scope="session")
def logs():
    """Bitacoras sinteticas reales del proyecto (generadas con
    data/generate_synthetic_data.py). Se usan tal cual -- no se fabrica un
    dataset de juguete aparte, para que los tests corran sobre exactamente
    los mismos datos que ve la app."""
    if not DATA_PATH.exists():
        pytest.skip("data/bitacoras.csv no existe -- correr "
                     "'python data/generate_synthetic_data.py --out data/bitacoras.csv'")
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


@pytest.fixture(scope="session")
def has_model():
    return MODEL_PATH.exists()


@pytest.fixture(scope="session")
def full_history_child_id(logs):
    """El nino con MAS dias de historial -- para tests que necesitan una
    serie temporal larga (evita depender del orden de las filas del CSV,
    ya que el dataset ahora mezcla ninos de historial completo con ninos de
    arranque en frio, Seccion 3.6)."""
    return logs.groupby("child_id").size().idxmax()


@pytest.fixture(scope="session")
def cold_start_child_id(logs):
    """El nino con MENOS dias de historial (arranque en frio)."""
    return logs.groupby("child_id").size().idxmin()
