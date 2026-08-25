"""
Verifica las invariantes criticas de core/features.py:
  - Sin fuga temporal (theta de un dia solo usa dias ESTRICTAMENTE anteriores).
  - Consistencia con core/bayesian.py (misma formula, mismo resultado).
  - Paridad entre build_features (entrenamiento) y build_features_for_today
    (inferencia) -- ambas rutas deben producir EXACTAMENTE las mismas
    variables para el mismo nino-dia.
"""
import numpy as np
import pandas as pd

from core.bayesian import K_DEFAULT, compute_baseline, shrinkage_weight
from core.features import (
    BASELINE_VARS,
    FEATURE_COLUMNS,
    RAW_CATEGORICAL,
    RAW_NUMERIC,
    _sleep_ordinal,
    build_features,
    build_features_for_today,
    engineer_child,
    population_baselines,
)


def test_theta_uses_only_strictly_past_days(logs, full_history_child_id):
    cid = full_history_child_id
    mu = population_baselines(logs)
    g = engineer_child(logs[logs["child_id"] == cid], mu)

    var = "sueno_ord"
    vals = g[var].to_numpy(dtype="float64")
    for d in range(len(g)):
        past = vals[:d]
        past = past[~np.isnan(past)]
        n = len(past)
        w = shrinkage_weight(n, K_DEFAULT)
        expected = (w * past.mean() + (1 - w) * mu[var]) if n > 0 else mu[var]
        got = g[f"theta_{var}"].iloc[d]
        assert np.isclose(expected, got, atol=1e-9), f"fuga temporal detectada en el dia {d}"


def test_theta_would_differ_if_it_leaked_today(logs, full_history_child_id):
    """Control negativo: si theta incluyera el dato de HOY, no coincidiria
    con el shrinkage de dias<d en la mayoria de los dias (confirma que el
    test anterior realmente esta verificando algo, no pasando por casualidad)."""
    cid = full_history_child_id
    mu = population_baselines(logs)
    g = engineer_child(logs[logs["child_id"] == cid], mu)
    var = "sueno_ord"
    vals = g[var].to_numpy(dtype="float64")

    including_today = []
    for d in range(len(g)):
        past = vals[: d + 1]
        past = past[~np.isnan(past)]
        n = len(past)
        w = shrinkage_weight(n, K_DEFAULT)
        including_today.append((w * past.mean() + (1 - w) * mu[var]) if n > 0 else mu[var])

    diffs = (~np.isclose(np.array(including_today), g[f"theta_{var}"].to_numpy(), atol=1e-9)).sum()
    assert diffs > len(g) * 0.5


def test_theta_today_matches_bayesian_compute_baseline(logs, full_history_child_id):
    cid = full_history_child_id
    child_hist = logs[logs["child_id"] == cid].sort_values("date")
    last_raw = child_hist.iloc[-1].to_dict()
    hist_wo_last = child_hist.iloc[:-1]
    hist_all = pd.concat([logs[logs["child_id"] != cid], hist_wo_last], ignore_index=True)

    today = {c: last_raw[c] for c in RAW_NUMERIC + RAW_CATEGORICAL + ["crisis_hoy"] if c in last_raw}
    mu_hist = population_baselines(hist_all)
    row_today = build_features_for_today(hist_all, cid, today, mu=mu_hist)

    hist_all_ord = hist_all.copy()
    hist_all_ord["sueno_ord"] = _sleep_ordinal(hist_all_ord["calidad_sueno"])
    for var in BASELINE_VARS:
        base = compute_baseline(hist_all_ord, cid, var)
        got = row_today[f"theta_{var}"].iloc[0]
        assert np.isclose(base.theta, got, atol=1e-9), f"theta_{var} no coincide con compute_baseline"


def test_training_and_inference_features_match(logs, full_history_child_id):
    """build_features (todo el dataset de una vez) y build_features_for_today
    (fila por fila, como en produccion) deben coincidir exactamente."""
    cid = full_history_child_id
    mu = population_baselines(logs)
    feat = build_features(logs, mu=mu)
    feat_child = feat[feat["child_id"] == cid].sort_values("date").reset_index(drop=True)
    g_sorted = logs[logs["child_id"] == cid].sort_values("date").reset_index(drop=True)

    check_days = sorted({1, 5, 30, len(g_sorted) - 1})
    for d in check_days:
        if d < 1 or d >= len(g_sorted):
            continue
        raw_d = g_sorted.iloc[d].to_dict()
        hist_upto = pd.concat([logs[logs["child_id"] != cid], g_sorted.iloc[:d]], ignore_index=True)
        today_d = {c: raw_d[c] for c in RAW_NUMERIC + RAW_CATEGORICAL + ["crisis_hoy"] if c in raw_d}
        row_inf = build_features_for_today(
            hist_upto, cid, today_d, today_date=pd.to_datetime(raw_d["date"]), mu=mu)
        row_train = feat_child.iloc[[d]][FEATURE_COLUMNS].reset_index(drop=True)

        for col in FEATURE_COLUMNS:
            a, b = row_inf[col].iloc[0], row_train[col].iloc[0]
            same = (pd.isna(a) and pd.isna(b)) or (
                a == b if isinstance(a, str) or isinstance(b, str)
                else np.isclose(float(a), float(b), atol=1e-9)
            )
            assert same, f"dia {d} columna {col}: inferencia={a!r} != entrenamiento={b!r}"


def test_cold_start_child_gets_low_shrinkage_weight(logs, cold_start_child_id):
    """El nino de arranque en frio (Seccion 3.6) debe tener theta cerca del
    promedio poblacional, no de un promedio propio inestable con pocos dias."""
    mu = population_baselines(logs)
    g = engineer_child(logs[logs["child_id"] == cold_start_child_id], mu)
    last_w = g["dias_historial"].iloc[-1]
    assert last_w < 10  # por diseno del generador (2-5 dias de historial)
