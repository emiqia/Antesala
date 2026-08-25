"""
Validacion de la ingenieria de variables (core/features.py).

Comprueba, sobre data/bitacoras.csv:
  1. Forma y cobertura de la matriz de variables.
  2. Un caso de prueba dia a dia (theta, desviacion, ventanas, dias desde crisis).
  3. Ausencia de fuga temporal: theta_d recalculado a mano con dias < d coincide.
  4. Consistencia con core/bayesian.py (theta de "hoy" == compute_baseline).
  5. Paridad entrenamiento <-> inferencia (build_features vs build_features_for_today).

Uso:
    python scripts/validate_features.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.features import (
    build_features, build_features_for_today, engineer_child, _sleep_ordinal,
    population_baselines, BASELINE_VARS, FEATURE_COLUMNS, RAW_NUMERIC, RAW_CATEGORICAL,
)
from core.bayesian import compute_baseline, shrinkage_weight, K_DEFAULT

DATA = Path(__file__).resolve().parents[1] / "data" / "bitacoras.csv"


def main():
    logs = pd.read_csv(DATA, parse_dates=["date"])
    print("=" * 70)
    print("1. MATRIZ DE VARIABLES")
    feat = build_features(logs)
    print(f"   filas: {len(feat)}  columnas: {feat.shape[1]}")
    print(f"   n features modelo: {len(FEATURE_COLUMNS)}")
    print(f"   target crisis_24h presente: {'crisis_24h' in feat.columns}")
    derived = [c for c in feat.columns if c not in RAW_NUMERIC + RAW_CATEGORICAL
               + ['child_id', 'date', 'crisis_24h']]
    print("   nulos en variables DERIVADAS (deberian ser ~0, salvo desviacion_*):")
    print(feat[derived].isna().sum().to_string())

    print("=" * 70)
    print("2. CASO DE PRUEBA DIA A DIA (primer nino)")
    cid = sorted(logs["child_id"].unique())[0]
    mu = population_baselines(logs)  # incluye BASELINE_VARS + crisis_rate
    print(f"   nino: {cid}   mu poblacional: "
          f"{{'sueno_ord': {mu['sueno_ord']:.3f}, "
          f"'n_eventos_desregulacion': {mu['n_eventos_desregulacion']:.3f}, "
          f"'crisis_rate': {mu['crisis_rate']:.3f}}}")
    g = engineer_child(logs[logs["child_id"] == cid], mu)
    cols = ["date", "calidad_sueno", "sueno_ord", "theta_sueno_ord", "desviacion_sueno_ord",
            "sueno_ma3", "n_eventos_desregulacion", "desreg_sum3",
            "crisis_hoy_num", "dias_desde_ultima_crisis", "antiguedad_calidad_sueno",
            "dias_historial", "crisis_24h"]
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(g[cols].head(12).to_string(index=False))

    print("=" * 70)
    print("3. SIN FUGA TEMPORAL: theta_d == shrinkage(historial dias < d)")
    ok = True
    var = "sueno_ord"
    vals = g[var].to_numpy(dtype="float64")
    for d in range(len(g)):
        past = vals[:d]
        past = past[~np.isnan(past)]
        n = len(past)
        w = shrinkage_weight(n, K_DEFAULT)
        expected = (w * past.mean() + (1 - w) * mu[var]) if n > 0 else mu[var]
        got = g[f"theta_{var}"].iloc[d]
        if not np.isclose(expected, got, atol=1e-9):
            ok = False
            print(f"   MISMATCH dia {d}: esperado {expected:.6f} got {got:.6f}")
    print(f"   theta reproduce shrinkage estricto de dias<d: {'OK' if ok else 'FALLA'}")
    # verificacion negativa: si hubiera fuga (usara dia d), no coincidiria en general
    incl = []
    for d in range(len(g)):
        past = vals[:d + 1]
        past = past[~np.isnan(past)]
        n = len(past)
        w = shrinkage_weight(n, K_DEFAULT)
        incl.append((w * past.mean() + (1 - w) * mu[var]) if n > 0 else mu[var])
    diffs = (~np.isclose(np.array(incl), g[f"theta_{var}"].to_numpy(), atol=1e-9)).sum()
    print(f"   (control) theta que SI incluyera hoy diferiria en {diffs}/{len(g)} dias")

    print("=" * 70)
    print("4. CONSISTENCIA CON core/bayesian.py")
    # theta de 'hoy' (build_features_for_today) debe igualar compute_baseline
    # sobre el historial del nino (formula de shrinkage con historial completo).
    child_hist = logs[logs["child_id"] == cid]
    last_raw = child_hist.sort_values("date").iloc[-1].to_dict()
    hist_wo_last = child_hist.sort_values("date").iloc[:-1]
    # today = ultima fila real; historial = todo menos esa fila
    hist_all = pd.concat([logs[logs["child_id"] != cid], hist_wo_last], ignore_index=True)
    today = {c: last_raw[c] for c in RAW_NUMERIC + RAW_CATEGORICAL + ["crisis_hoy"]
             if c in last_raw}
    # mu se calcula sobre el mismo historial que ve compute_baseline (hist_all).
    mu_hist = population_baselines(hist_all)
    row_today = build_features_for_today(hist_all, cid, today, mu=mu_hist)
    # compute_baseline (core/bayesian.py) trabaja sobre columnas RAW del df: para
    # sueno_ord hay que materializar la columna ordinal antes de llamarlo.
    hist_all_ord = hist_all.copy()
    hist_all_ord["sueno_ord"] = _sleep_ordinal(hist_all_ord["calidad_sueno"])
    for var in BASELINE_VARS:
        base = compute_baseline(hist_all_ord, cid, var)
        got = row_today[f"theta_{var}"].iloc[0]
        status = "OK" if np.isclose(base.theta, got, atol=1e-9) else "FALLA"
        print(f"   theta_{var}: features={got:.6f}  bayesian.compute_baseline="
              f"{base.theta:.6f}  [{status}]")

    print("=" * 70)
    print("5. PARIDAD ENTRENAMIENTO <-> INFERENCIA")
    # Para un dia d intermedio, la fila de inferencia (historial<d + today=d) debe
    # coincidir con la fila de build_features para ese nino-dia.
    feat_child = feat[feat["child_id"] == cid].sort_values("date").reset_index(drop=True)
    g_sorted = child_hist.sort_values("date").reset_index(drop=True)
    mismatches = 0
    checked = 0
    for d in [5, 15, 30, len(g_sorted) - 1]:
        if d < 1 or d >= len(g_sorted):
            continue
        raw_d = g_sorted.iloc[d].to_dict()
        hist_upto = pd.concat(
            [logs[logs["child_id"] != cid], g_sorted.iloc[:d]], ignore_index=True)
        today_d = {c: raw_d[c] for c in RAW_NUMERIC + RAW_CATEGORICAL + ["crisis_hoy"]
                   if c in raw_d}
        # mu congelado del entrenamiento (dataset completo), igual que build_features.
        row_inf = build_features_for_today(
            hist_upto, cid, today_d, today_date=pd.to_datetime(raw_d["date"]), mu=mu)
        row_train = feat_child.iloc[[d]][FEATURE_COLUMNS].reset_index(drop=True)
        checked += 1
        for col in FEATURE_COLUMNS:
            a = row_inf[col].iloc[0]
            b = row_train[col].iloc[0]
            same = (pd.isna(a) and pd.isna(b)) or (
                a == b if isinstance(a, str) or isinstance(b, str)
                else np.isclose(float(a), float(b), atol=1e-9))
            if not same:
                mismatches += 1
                print(f"   dia {d} col {col}: inf={a!r} train={b!r}")
    print(f"   dias verificados: {checked}  discrepancias: {mismatches} "
          f"-> {'PARIDAD OK' if mismatches == 0 else 'REVISAR'}")


if __name__ == "__main__":
    main()
