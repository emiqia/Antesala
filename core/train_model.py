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

CAMBIOS DE LA REVISION METODOLOGICA (agosto 2026)
-------------------------------------------------
1. DOS REGIMENES DE EVALUACION en vez de uno (ver core/evaluation.py):
   ninos no vistos (particion por nino) y dias futuros (particion temporal).
   Responden preguntas distintas y dan numeros distintos; reportar solo uno
   deja fuera la mitad del problema.

2. CALIBRACION ISOTONICA. Un Random Forest promedia arboles, asi que sus
   probabilidades se comprimen hacia el centro: casi nunca dice 5% ni 95%.
   Mientras el numero se muestre como porcentaje en pantalla ("riesgo 68%"),
   eso importa. El calibrador se ajusta FUERA DE MUESTRA (out-of-fold con
   GroupKFold por nino) y se guarda en el bundle; core/risk_model.py lo
   aplica en inferencia. Se reporta Brier y ECE antes y despues.

3. PANEL DE METRICAS en vez de accuracy: AUROC, AUPRC contra tasa base,
   Brier, sensibilidad, PPV, falsas alertas por nino/semana y episodios no
   detectados.

4. k ESTIMADO DE LOS DATOS. El documento fija k=5 a mano; aqui se reporta
   ademas el k que sale del estimador de momentos Beta-Binomial
   (core/bayesian.estimate_prior_strength) para que el valor usado quede
   contrastado con evidencia y no solo declarado.

Persiste en models/antesala_rf.joblib un dict con el pipeline entrenado, el mu
poblacional (linea base bayesiana), el calibrador y la lista de columnas, para
que la inferencia (core/risk_model.py) reproduzca exactamente las mismas
variables.

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
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.features import (
    build_features, population_baselines,
    FEATURE_NUMERIC, FEATURE_CATEGORICAL, TARGET,
)
from core.question_selector import ASKABLE_FIELDS
from core.bayesian import K_DEFAULT, estimate_prior_strength
from core import evaluation as ev

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
    """wi del INDICE DE SUFICIENCIA (Seccion 6.2): 'la importancia relativa de
    la variable i, obtenida del feature importance del modelo'. A diferencia
    de VARIABLE_WEIGHTS (pesos clinicos fijos que usa el score heuristico,
    Seccion 6.1), estos pesos vienen DIRECTAMENTE del Random Forest entrenado:
    se toma la importancia de la columna que senala "esta variable no fue
    registrada hoy" -- es, literalmente, cuanto le importa al modelo saber si
    esa variable esta o no disponible."""
    importance_by_name = dict(feature_importance)
    return {
        field: float(importance_by_name.get(_missingness_column_name(field), 0.0))
        for field in ASKABLE_FIELDS
    }


def _evaluar(nombre: str, feat, X, y, idx_train, idx_test) -> tuple[dict, np.ndarray, np.ndarray]:
    """Entrena en idx_train y evalua en idx_test, devolviendo el panel."""
    pipe = build_pipeline()
    pipe.fit(X.iloc[idx_train], y.iloc[idx_train])
    proba = pipe.predict_proba(X.iloc[idx_test])[:, 1]
    y_te = y.iloc[idx_test]
    sub = feat.iloc[idx_test]
    dias = sub.groupby("child_id")["date"].nunique().mean() if "date" in sub else None
    m = ev.panel(y_te, proba, groups=sub["child_id"], dias=dias)
    m["regimen"] = nombre
    return m, np.asarray(y_te), proba


def _calibrador_out_of_fold(X, y, groups, n_splits: int = 5):
    """Ajusta el calibrador isotonico sobre predicciones FUERA DE MUESTRA.

    Calibrar con predicciones dentro de muestra es inutil: el Random Forest ya
    memorizo esas filas y sus probabilidades son artificialmente extremas, asi
    que el calibrador aprenderia una correccion equivocada. Se usa GroupKFold
    por nino para que las predicciones que ve el calibrador vengan siempre de
    un modelo que NO vio a ese nino.
    """
    oof = np.zeros(len(y), dtype=float)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups=groups):
        p = build_pipeline()
        p.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = p.predict_proba(X.iloc[te])[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    cal.fit(oof, np.asarray(y))
    return cal, oof


def _calibracion_honesta(oof, y, groups, n_splits: int = 5):
    """Predicciones calibradas que el calibrador NO vio al ajustarse.

    Sin esto el reporte miente. Si se ajusta el isotonico sobre `oof` y se
    mide la calibracion sobre ese mismo `oof`, la regresion isotonica -- que
    es libre de doblarse todo lo que haga falta -- reproduce la frecuencia
    observada de cada tramo por construccion, y el ECE sale 0.0000. No es que
    el modelo este perfectamente calibrado: es que se le pregunto la respuesta
    que acababa de memorizar.

    Aqui el calibrador se ajusta en K-1 particiones y se evalua en la
    restante, asi que el numero reportado es el que se puede esperar sobre
    datos nuevos.
    """
    y = np.asarray(y)
    fuera = np.zeros(len(y), dtype=float)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(oof.reshape(-1, 1), y, groups=groups):
        c = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        c.fit(oof[tr], y[tr])
        fuera[te] = c.predict(oof[te])
    return fuera


def train(save: bool = True, verbose: bool = True) -> dict:
    logs = pd.read_csv(DATA_PATH, parse_dates=["date"])
    mu = population_baselines(logs)
    feat = build_features(logs, mu=mu).reset_index(drop=True)

    X = feat[FEATURE_NUMERIC + FEATURE_CATEGORICAL]
    y = feat[TARGET].astype(int)
    groups = feat["child_id"]

    def p(*a):
        if verbose:
            print(*a)

    p("=" * 78)
    p("ENTRENAMIENTO RANDOM FOREST  (nivel principal, Seccion 6.1)")
    p("=" * 78)
    p(f"Registros: {len(feat)}   ninos: {groups.nunique()}   "
      f"tasa base de crisis_24h: {y.mean():.3f}")

    # --- k empirico: contrasta el k=5 del documento con el que dicen los datos ---
    aux = logs.copy()
    aux["crisis_rate"] = aux["crisis_hoy"].astype(float)
    k_est = estimate_prior_strength(aux, "crisis_rate")
    if k_est:
        p("-" * 78)
        p("PARTIAL POOLING -- concentracion del prior (Seccion 3.4)")
        p(f"  k usado en el pipeline (documento) : {K_DEFAULT}")
        p(f"  k estimado de los datos            : {k_est['k']:.1f}  "
          f"(momentos Beta-Binomial, {k_est['n_ninos']} ninos)")
        p(f"  mu poblacional                     : {k_est['mu']:.3f}")
        p(f"  varianza real entre ninos          : {k_est['var_entre_ninos']:.5f}")
        p("  Lectura: los datos piden encoger MAS hacia la poblacion de lo que")
        p("  encoge k=5. Ver scripts/benchmark.py para el efecto en las metricas.")

    # --- Los dos regimenes de generalizacion (Seccion 17) ---
    tr_n, te_n = ev.split_por_nino(feat, seed=RANDOM_STATE)
    tr_t, te_t = ev.split_por_tiempo(feat)

    m_ninos, y_n, p_n = _evaluar("ninos no vistos", feat, X, y, tr_n, te_n)
    m_tiempo, y_t, p_t = _evaluar("dias futuros", feat, X, y, tr_t, te_t)

    p("-" * 78)
    p("PANEL DE EVALUACION  (umbral fijado para sensibilidad "
      f"{ev.SENSIBILIDAD_OBJETIVO:.0%})")
    p("-" * 78)
    p(ev.CABECERA_PANEL)
    p(ev.formato_panel("ninos no vistos", m_ninos))
    p(ev.formato_panel("dias futuros", m_tiempo))
    p("")
    p(f"  tasa base (piso del AUPRC): {m_ninos['tasa_base']:.3f} / "
      f"{m_tiempo['tasa_base']:.3f}")
    p("  No se reporta accuracy a proposito: con esta tasa base premia al")
    p("  modelo que nunca avisa (ver core/evaluation.py).")

    # --- Calibracion ---
    p("-" * 78)
    p("CALIBRACION (isotonica, out-of-fold por nino)")
    cal, oof = _calibrador_out_of_fold(X, y, groups)
    # El calibrador que se GUARDA se ajusta con todos los datos (es el que
    # mejor generaliza), pero lo que se REPORTA se mide con un calibrador que
    # no vio las filas que evalua -- si no, el ECE sale 0 por construccion.
    oof_cal = _calibracion_honesta(oof, y, groups)
    brier_antes = ev.panel(y, oof, groups=groups)["brier"]
    brier_despues = ev.panel(y, oof_cal, groups=groups)["brier"]
    ece_antes = ev.error_calibracion_esperado(y, oof)
    ece_despues = ev.error_calibracion_esperado(y, oof_cal)
    p(f"  Brier  antes: {brier_antes:.4f}   despues: {brier_despues:.4f}")
    p(f"  ECE    antes: {ece_antes:.4f}   despues: {ece_despues:.4f}")
    p("  (ECE = brecha media entre lo que el modelo dice y lo que ocurre;")
    p("   medido con un calibrador que no vio estas filas)")
    p("")
    p("  Tabla de fiabilidad tras calibrar:")
    tabla = ev.calibracion(y, oof_cal)
    for _, r in tabla.iterrows():
        p(f"    n={int(r['n']):4d}  predicho={r['predicho']:.3f}  "
          f"observado={r['observado']:.3f}  brecha={r['brecha']:+.3f}")

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

    p("-" * 78)
    p("Top 12 variables por importancia (modelo final):")
    for nombre, imp in feature_importance[:12]:
        p(f"  {nombre:<48s} {imp:.4f}")

    # Pesos de completitud del indice de suficiencia (Seccion 6.2) --
    # derivados del feature importance real del modelo, no de los pesos
    # clinicos manuales del score heuristico (Seccion 6.1).
    confidence_weights = compute_confidence_weights(feature_importance)

    bundle = {
        "pipeline": final_pipe,
        "calibrator": cal,
        "mu": mu,
        "feature_numeric": FEATURE_NUMERIC,
        "feature_categorical": FEATURE_CATEGORICAL,
        "feature_importance": feature_importance,
        # Mismo dict bajo los dos nombres: "confidence_weights" es el nombre
        # historico que ya leen el codigo y los tests; "sufficiency_weights" es
        # el nombre correcto tras la revision. Se mantienen los dos para no
        # romper compatibilidad con modelos ya entrenados.
        "confidence_weights": confidence_weights,
        "sufficiency_weights": confidence_weights,
        "metrics": {
            "ninos_no_vistos": m_ninos,
            "dias_futuros": m_tiempo,
            "calibracion": {"brier_antes": brier_antes, "brier_despues": brier_despues,
                            "ece_antes": ece_antes, "ece_despues": ece_despues},
            "k_estimado": (k_est or {}).get("k"),
            # Alias planos que ya consumia la interfaz.
            "roc_auc": m_ninos["auroc"],
            "average_precision": m_ninos["auprc"],
            "test_base_rate": m_ninos["tasa_base"],
        },
        "alcance": ("Entrenado y evaluado sobre datos SINTETICOS. Valida el "
                     "funcionamiento del pipeline, no capacidad predictiva "
                     "clinica (Seccion 16)."),
    }
    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)
        p("-" * 78)
        p(f"Modelo guardado en: {MODEL_PATH.relative_to(ROOT)}")
        p("ALCANCE: " + bundle["alcance"])
    return bundle


if __name__ == "__main__":
    train()
