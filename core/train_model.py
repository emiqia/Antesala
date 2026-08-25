"""
Entrenamiento del modelo de riesgo principal (Seccion 6.1, nivel principal):
un Random Forest sobre crisis_24h, con las variables originales y derivadas
(Secciones 4.1 y 4.3, via core/features.py).

Pipeline (Seccion 8.1):
  - Numericas:   SimpleImputer(median, add_indicator=True)  -> indicador de
                 dato faltante explicito (Seccion 4.4), sin imputar en silencio.
  - Categoricas: SimpleImputer(constant='__missing__') + OneHotEncoder.
  - Modelo:      RandomForestClassifier (ensamble de arboles, Seccion 6.1),
                 con class_weight balanceado por el desbalance de crisis_24h.

Se evalua con un split POR NINO (GroupShuffleSplit): el modelo se prueba sobre
ninos no vistos en entrenamiento, que es el escenario real de "cold start".

Persiste en models/antesala_rf.joblib un dict con el pipeline entrenado, el mu
poblacional (linea base bayesiana) y la lista de columnas, para que la
inferencia (core/risk_model.py) reproduzca exactamente las mismas variables.

Uso:
    python core/train_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.features import (
    build_features, population_baselines,
    FEATURE_NUMERIC, FEATURE_CATEGORICAL, TARGET,
)
from core.question_selector import ASKABLE_FIELDS

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "bitacoras.csv"
MODEL_PATH = ROOT / "models" / "antesala_rf.joblib"

RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    """Construye el pipeline completo: preprocesamiento + Random Forest."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("num", numeric, FEATURE_NUMERIC),
        ("cat", categorical, FEATURE_CATEGORICAL),
    ])
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("rf", model)])


def _missingness_column_name(field: str) -> str:
    """Nombre, en el espacio ya transformado por el ColumnTransformer, de la
    columna que indica si `field` NO fue registrado hoy (el indicador de
    ausencia de SimpleImputer(add_indicator=True) para numericas, o la
    categoria "__missing__" del OneHotEncoder para categoricas)."""
    if field == "calidad_sueno":
        return "num__missingindicator_sueno_ord"
    if field in FEATURE_NUMERIC:
        return f"num__missingindicator_{field}"
    return f"cat__{field}___missing__"


def compute_confidence_weights(feature_importance: list[tuple[str, float]]) -> dict[str, float]:
    """wi de la Seccion 6.2: 'la importancia relativa de la variable i,
    obtenida del feature importance del modelo'. A diferencia de
    VARIABLE_WEIGHTS (pesos clinicos fijos que usa el score heuristico,
    Seccion 6.1), estos pesos vienen DIRECTAMENTE del Random Forest entrenado:
    se toma la importancia de la columna que senala "esta variable no fue
    registrada hoy" -- es, literalmente, cuanto le importa al modelo saber si
    esa variable esta o no disponible."""
    importance_by_name = dict(feature_importance)
    return {
        field: float(importance_by_name.get(_missingness_column_name(field), 0.0))
        for field in ASKABLE_FIELDS
    }


def train(save: bool = True) -> dict:
    logs = pd.read_csv(DATA_PATH, parse_dates=["date"])
    mu = population_baselines(logs)
    feat = build_features(logs, mu=mu)

    X = feat[FEATURE_NUMERIC + FEATURE_CATEGORICAL]
    y = feat[TARGET].astype(int)
    groups = feat["child_id"]

    # --- Split por nino (evalua generalizacion a ninos nuevos) ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)

    proba_te = pipe.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba_te)
    ap = average_precision_score(y_te, proba_te)
    base_rate = y_te.mean()

    print("=" * 64)
    print("ENTRENAMIENTO RANDOM FOREST  (nivel principal, Seccion 6.1)")
    print("=" * 64)
    print(f"Registros totales : {len(feat)}   ninos: {groups.nunique()}")
    print(f"Train: {len(X_tr)} filas ({groups.iloc[train_idx].nunique()} ninos)  "
          f"Test: {len(X_te)} filas ({groups.iloc[test_idx].nunique()} ninos)")
    print(f"Tasa de crisis_24h en test (baseline): {base_rate:.3f}")
    print("-" * 64)
    print(f"ROC AUC (test, ninos no vistos) : {auc:.3f}")
    print(f"PR  AUC / Average Precision     : {ap:.3f}  (baseline={base_rate:.3f})")
    print("-" * 64)
    print("Reporte de clasificacion (umbral 0.5):")
    print(classification_report(y_te, (proba_te >= 0.5).astype(int),
                                target_names=["sin crisis", "crisis_24h"], zero_division=0))

    # --- Feature importance (responde a "que variables aportan mas", Sec. 6.1) ---
    feat_names = pipe.named_steps["pre"].get_feature_names_out()
    importances = pipe.named_steps["rf"].feature_importances_
    imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)
    print("Top 15 variables por importancia:")
    print(imp.head(15).to_string())

    # --- Reentrenar con TODOS los datos para el modelo final que usa la app ---
    final_pipe = build_pipeline()
    final_pipe.fit(X, y)

    # Importancia de variables del modelo FINAL (el que se despliega), no solo
    # el de validacion -- responde al requisito de "identificar que variables
    # aportan mayor valor predictivo". Se persiste para que la app la muestre
    # sin tener que inspeccionar el pipeline en tiempo de inferencia.
    final_feat_names = final_pipe.named_steps["pre"].get_feature_names_out()
    final_importances = final_pipe.named_steps["rf"].feature_importances_
    feature_importance = sorted(
        zip(final_feat_names.tolist(), final_importances.tolist()),
        key=lambda t: t[1], reverse=True)

    # Pesos de completitud para el calculo de confianza (Seccion 6.2) --
    # derivados del feature importance real del modelo, no de los pesos
    # clinicos manuales del score heuristico (Seccion 6.1).
    confidence_weights = compute_confidence_weights(feature_importance)
    print("-" * 64)
    print("Pesos de confianza (Seccion 6.2, del feature importance del RF):")
    for field, w in sorted(confidence_weights.items(), key=lambda kv: -kv[1]):
        print(f"  {field:32s} {w:.4f}")

    bundle = {
        "pipeline": final_pipe,
        "mu": mu,
        "feature_numeric": FEATURE_NUMERIC,
        "feature_categorical": FEATURE_CATEGORICAL,
        "feature_importance": feature_importance,
        "confidence_weights": confidence_weights,
        "metrics": {"roc_auc": float(auc), "average_precision": float(ap),
                    "test_base_rate": float(base_rate)},
    }
    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)
        print("-" * 64)
        print(f"Modelo guardado en: {MODEL_PATH.relative_to(ROOT)}")
    return bundle


if __name__ == "__main__":
    train()
