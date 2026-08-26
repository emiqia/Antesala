"""
Prueba el panel de evaluacion (Seccion 17) y las dos particiones.

Lo que se verifica aqui no es "el modelo es bueno" sino que las HERRAMIENTAS
con las que se mide sean correctas. Una particion con fuga o una metrica mal
calculada producen numeros optimistas que nadie detecta despues.
"""
import numpy as np
import pandas as pd
import pytest

from core import evaluation as ev


# ------------------------------------------------------------------ panel ---
def test_predictor_perfecto_da_auroc_uno():
    y = np.array([0, 0, 1, 1, 0, 1])
    m = ev.panel(y, y.astype(float))
    assert m["auroc"] == pytest.approx(1.0)
    assert m["no_detectados"] == 0


def test_predictor_constante_da_auroc_medio():
    """Una constante no discrimina nada: AUROC 0.5 exacto. Es el baseline A."""
    y = np.array([0, 1] * 25)
    m = ev.panel(y, np.full(50, 0.4))
    assert m["auroc"] == pytest.approx(0.5)


def test_auprc_tiene_como_piso_la_tasa_base():
    """El AUPRC de un modelo inutil no es 0.5, es la tasa base. Por eso el
    panel siempre lo reporta junto a ella: sin esa referencia, un AUPRC de
    0.46 puede parecer un resultado y ser exactamente nada."""
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.3).astype(int)
    m = ev.panel(y, rng.random(2000))
    assert m["auprc"] == pytest.approx(m["tasa_base"], abs=0.05)
    assert m["auprc_lift"] == pytest.approx(1.0, abs=0.2)


def test_umbral_alcanza_la_sensibilidad_objetivo():
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.4).astype(int)
    proba = np.clip(y * 0.4 + rng.random(500) * 0.5, 0, 1)
    m = ev.panel(y, proba, objetivo=0.8)
    assert m["sensibilidad"] >= 0.78


def test_falsas_alertas_por_nino_semana_usa_la_unidad_correcta():
    """2 ninos, 14 dias cada uno = 4 semanas-nino. Con 8 falsos positivos
    deben salir 2 falsas alertas por nino y por semana."""
    y = np.zeros(28, dtype=int)
    y[:4] = 1
    proba = np.zeros(28)
    proba[:12] = 1.0            # 4 aciertos + 8 falsos positivos
    groups = np.array(["a"] * 14 + ["b"] * 14)
    m = ev.panel(y, proba, groups=groups, dias=14, umbral=0.5)
    assert m["falsas_alertas"] == 8
    assert m["fa_por_nino_semana"] == pytest.approx(2.0)


def test_panel_no_estalla_sin_ambas_clases():
    """En subconjuntos por nino puede no haber ningun episodio. Debe devolver
    NaN en las metricas que no estan definidas, no lanzar."""
    m = ev.panel(np.zeros(20, dtype=int), np.full(20, 0.3))
    assert np.isnan(m["auroc"])
    assert m["no_detectados"] == 0


# ------------------------------------------------------------ calibracion ---
def test_ece_cero_para_un_predictor_perfectamente_calibrado():
    rng = np.random.default_rng(7)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert ev.error_calibracion_esperado(y, p) < 0.02


def test_ece_detecta_un_modelo_descalibrado():
    """Un modelo que siempre dice el doble de lo que ocurre tiene que salir
    penalizado."""
    rng = np.random.default_rng(7)
    p_real = rng.random(20000) * 0.5
    y = (rng.random(20000) < p_real).astype(int)
    assert ev.error_calibracion_esperado(y, p_real * 2) > 0.15


# ------------------------------------------------------------ particiones ---
def test_split_por_nino_no_comparte_ninos(logs):
    feat = logs[["child_id", "date"]].copy()
    tr, te = ev.split_por_nino(feat)
    ninos_tr = set(feat.iloc[tr]["child_id"])
    ninos_te = set(feat.iloc[te]["child_id"])
    assert ninos_tr and ninos_te
    assert not (ninos_tr & ninos_te), "hay ninos en entrenamiento Y en test"


def test_split_por_tiempo_no_predice_el_pasado_con_el_futuro(logs):
    """La fuga que la revision senala explicitamente. Para cada nino, todos
    sus dias de test tienen que ser POSTERIORES a todos sus dias de
    entrenamiento; si no, el modelo ve el futuro del propio nino."""
    feat = logs[["child_id", "date"]].sort_values(["child_id", "date"]).reset_index(drop=True)
    tr, te = ev.split_por_tiempo(feat)
    a, b = feat.iloc[tr], feat.iloc[te]
    for cid in set(b["child_id"]):
        ultimo_train = a.loc[a["child_id"] == cid, "date"].max()
        primer_test = b.loc[b["child_id"] == cid, "date"].min()
        assert pd.isna(ultimo_train) or ultimo_train < primer_test, (
            f"{cid}: fuga temporal, entrena hasta {ultimo_train} y prueba desde {primer_test}")


def test_split_por_tiempo_no_pierde_filas(logs):
    feat = logs[["child_id", "date"]].sort_values(["child_id", "date"]).reset_index(drop=True)
    tr, te = ev.split_por_tiempo(feat)
    assert len(tr) + len(te) == len(feat)
    assert not (set(tr.tolist()) & set(te.tolist()))
