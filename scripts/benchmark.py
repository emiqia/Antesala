"""
Comparadores -- Seccion 18 del documento tecnico.

    "Para demostrar que el sistema aporta valor sera importante compararlo
     contra alternativas simples. Asi podra responderse si la complejidad
     adicional realmente mejora la toma de decisiones."

Sin esta tabla, decir "Random Forest con pooling bayesiano y variables
temporales" no significa nada: puede que un promedio historico de dos lineas
haga lo mismo. Aqui se mide.

MODELOS COMPARADOS
  A  tasa global              constante: la tasa de episodios de toda la
                              poblacion. Es el piso absoluto.
  B  tasa individual          la tasa historica del propio nino, sin encoger.
                              Es lo que haria cualquiera con una planilla.
  C  partial pooling (theta)  la linea base individual encogida hacia la
                              poblacion (Seccion 3.4), usada DIRECTAMENTE como
                              prediccion. Aisla cuanto aporta el pooling por si
                              solo, sin modelo de ML encima.
  D  logistica sin personalizar   regresion logistica regularizada sobre el
                              registro de hoy tal cual. Es el baseline
                              interpretable que pide la Seccion 8.1.
  E  Random Forest sin personalizar   mismo modelo principal, pero sin las
                              variables de linea base, ventanas moviles ni
                              antiguedad. Aisla cuanto aporta el modelo.
  F  ANTESALA (completo)      Random Forest con todo.

ABLACIONES sobre F: se quita un bloque de variables por vez, para ver cual
paga y cual solo agrega complejidad.

Ademas se contrasta k=5 (el valor del documento) contra el k estimado de los
datos por momentos Beta-Binomial.

Los dos regimenes de generalizacion (ninos no vistos / dias futuros) se
reportan por separado: ver core/evaluation.py.

ALCANCE: datos sinteticos. Esto compara ARQUITECTURAS entre si sobre el mismo
problema simulado; no mide capacidad predictiva clinica. Ver Seccion 16.

Uso:
    python scripts/benchmark.py
    python scripts/benchmark.py --rapido     (omite las ablaciones)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from core.features import (
    build_features, population_baselines,
    FEATURE_NUMERIC, FEATURE_CATEGORICAL, TARGET,
)
from core.question_selector import ASKABLE_FIELDS
from core.bayesian import estimate_prior_strength, K_DEFAULT
from core.train_model import build_pipeline
from core import evaluation as ev

DATA_PATH = ROOT / "data" / "bitacoras.csv"

# --- Bloques de variables, para las ablaciones -------------------------------
PERSONALIZACION = [
    "theta_crisis_rate", "dias_historial", "theta_sueno_ord",
    "desviacion_sueno_ord", "theta_n_eventos_desregulacion",
    "desviacion_n_eventos_desregulacion",
]
TEMPORALES = [
    "sueno_ma3", "sueno_ma7", "desreg_sum3", "desreg_sum7",
    "dias_desde_ultima_crisis", "transicion_reciente_3d",
    "cambio_rutina_reciente_3d",
]
ANTIGUEDAD = [f for f in FEATURE_NUMERIC if f.startswith("antiguedad_")]
HOY_NUM = [f for f in FEATURE_NUMERIC
           if f not in PERSONALIZACION + TEMPORALES + ANTIGUEDAD]


def pipeline_logistica(numericas, categoricas) -> Pipeline:
    """Baseline interpretable (Seccion 8.1): logistica regularizada.

    Se escala porque la regularizacion L2 penaliza coeficientes en la escala
    de cada variable; sin escalar, `dias_historial` (0-150) y `sueno_ord`
    (0-2) reciben castigos incomparables.
    """
    num = Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
        ("sc", StandardScaler()),
    ])
    cat = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([("num", num, numericas), ("cat", cat, categoricas)])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(
        max_iter=2000, class_weight="balanced", C=1.0))])


def tasa_individual_expandida(logs: pd.DataFrame, feat: pd.DataFrame) -> pd.Series:
    """Tasa historica del propio nino hasta el dia ANTERIOR (sin fuga).

    El shift(1) es lo que hace legitima la comparacion: la tasa del dia d se
    calcula solo con los dias 1..d-1. Sin el, el baseline B usaria el dia que
    esta prediciendo y ganaria por trampa.

    Se devuelve ya alineada con `feat` por (child_id, date) en vez de por
    posicion: build_features agrupa con sort=False, asi que su orden de filas
    no tiene por que coincidir con el de un sort_values, y alinear por indice
    mezclaria las tasas de un nino con los dias de otro.
    """
    aux = logs.sort_values(["child_id", "date"]).copy()
    aux["tasa_individual"] = (
        aux.groupby("child_id")["crisis_hoy"]
           .transform(lambda s: s.astype(float).shift(1).expanding().mean()))
    unido = feat[["child_id", "date"]].merge(
        aux[["child_id", "date", "tasa_individual"]], on=["child_id", "date"], how="left")
    return unido["tasa_individual"]


def evaluar_constante(nombre, valores, y_te, sub) -> dict:
    dias = sub.groupby("child_id")["date"].nunique().mean()
    m = ev.panel(y_te, valores, groups=sub["child_id"], dias=dias)
    m["modelo"] = nombre
    return m


def evaluar_modelo(nombre, pipe, X, y, tr, te, feat) -> dict:
    pipe.fit(X.iloc[tr], y.iloc[tr])
    proba = pipe.predict_proba(X.iloc[te])[:, 1]
    sub = feat.iloc[te]
    dias = sub.groupby("child_id")["date"].nunique().mean()
    m = ev.panel(y.iloc[te], proba, groups=sub["child_id"], dias=dias)
    m["modelo"] = nombre
    return m


def corre_regimen(nombre_regimen, feat, X, y, tr, te, tasa_ind, rapido=False):
    print()
    print("=" * 92)
    print(f"REGIMEN: {nombre_regimen}")
    print("=" * 92)
    print(ev.CABECERA_PANEL)
    print("-" * 92)

    y_te = y.iloc[te]
    sub = feat.iloc[te]
    filas = []

    # A -- tasa global (constante, estimada SOLO en entrenamiento)
    tasa_global = float(y.iloc[tr].mean())
    filas.append(evaluar_constante("A tasa global", np.full(len(te), tasa_global), y_te, sub))

    # B -- tasa individual cruda (rellena con la global cuando aun no hay historial)
    b = tasa_ind.iloc[te].fillna(tasa_global).to_numpy()
    filas.append(evaluar_constante("B tasa individual", b, y_te, sub))

    # C -- partial pooling (theta) usado directamente como prediccion
    c = feat["theta_crisis_rate"].iloc[te].fillna(tasa_global).to_numpy()
    filas.append(evaluar_constante("C partial pooling", c, y_te, sub))

    # D -- logistica sin personalizacion
    filas.append(evaluar_modelo("D logistica (hoy)", pipeline_logistica(HOY_NUM, FEATURE_CATEGORICAL),
                                feat[HOY_NUM + FEATURE_CATEGORICAL], y, tr, te, feat))

    # E -- Random Forest sin personalizacion
    filas.append(evaluar_modelo("E RF (hoy)", build_pipeline_subset(HOY_NUM, FEATURE_CATEGORICAL),
                                feat[HOY_NUM + FEATURE_CATEGORICAL], y, tr, te, feat))

    # F -- Antesala completo
    filas.append(evaluar_modelo("F ANTESALA (completo)", build_pipeline(), X, y, tr, te, feat))

    # Logistica CON todo, para separar "aporta el modelo" de "aportan las variables"
    filas.append(evaluar_modelo("  logistica (completo)",
                                pipeline_logistica(FEATURE_NUMERIC, FEATURE_CATEGORICAL),
                                X, y, tr, te, feat))

    for m in filas:
        print(ev.formato_panel(m["modelo"], m))

    if not rapido:
        print("-" * 92)
        print("ABLACIONES sobre F (se quita un bloque de variables por vez)")
        print("-" * 92)
        for etiqueta, quitar in [("- personalizacion", PERSONALIZACION),
                                 ("- temporales", TEMPORALES),
                                 ("- antiguedad", ANTIGUEDAD)]:
            nums = [f for f in FEATURE_NUMERIC if f not in quitar]
            m = evaluar_modelo(etiqueta, build_pipeline_subset(nums, FEATURE_CATEGORICAL),
                               feat[nums + FEATURE_CATEGORICAL], y, tr, te, feat)
            print(ev.formato_panel(m["modelo"], m))

    print("-" * 92)
    print(f"  tasa base del test: {filas[0]['tasa_base']:.3f}   "
          f"(el AUPRC de un modelo inutil vale eso)")
    return filas


def build_pipeline_subset(numericas, categoricas) -> Pipeline:
    """Mismo Random Forest y mismo preprocesamiento que el modelo principal,
    pero sobre un subconjunto de columnas -- para las ablaciones."""
    from sklearn.ensemble import RandomForestClassifier
    num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True))])
    cat = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([("num", num, numericas), ("cat", cat, categoricas)])
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                max_features="sqrt", class_weight="balanced_subsample",
                                random_state=42, n_jobs=-1)
    return Pipeline([("pre", pre), ("rf", rf)])


def analisis_por_segmento(feat, X, y, tr, te):
    """Donde paga la personalizacion, si es que paga en alguna parte.

    El promedio global de la tabla anterior esta dominado por dias normales de
    ninos con 150 dias de historial: dias en los que el registro de hoy ya dice
    casi todo, y la linea base individual no tiene nada que agregar. Si la
    personalizacion sirve para algo, tiene que ser en los dos segmentos donde
    el registro de hoy NO alcanza:

      dia sin registro   nadie abrio la app. El contenido clinico de hoy es
                         cero, asi que lo unico que queda para distinguir a un
                         nino de otro es su propia linea base. Es ademas el
                         estado en que la interfaz abre TODOS los dias.
      arranque en frio   el nino tiene pocos dias de historial. Es el caso que
                         motivo el partial pooling (Seccion 3.6).

    Se comparan dos modelos entrenados sobre las MISMAS filas y evaluados sobre
    las MISMAS filas; lo unico que cambia es si ven o no el bloque de variables
    de personalizacion. Cualquier diferencia es atribuible a ese bloque.
    """
    print()
    print("=" * 92)
    print("DONDE PAGA LA PERSONALIZACION  (segmentos del test de ninos no vistos)")
    print("=" * 92)

    sin_person = [f for f in FEATURE_NUMERIC if f not in PERSONALIZACION]
    modelos = {
        "sin personalizacion": (build_pipeline_subset(sin_person, FEATURE_CATEGORICAL),
                                feat[sin_person + FEATURE_CATEGORICAL]),
        "ANTESALA (completo)": (build_pipeline(), X),
    }
    predicciones = {}
    for nombre, (pipe, Xm) in modelos.items():
        pipe.fit(Xm.iloc[tr], y.iloc[tr])
        predicciones[nombre] = pipe.predict_proba(Xm.iloc[te])[:, 1]

    sub = feat.iloc[te].reset_index(drop=True)
    campos_hoy = [c for c in ASKABLE_FIELDS if c in sub.columns]
    dia_vacio = sub[campos_hoy].isna().all(axis=1).to_numpy()
    poco_historial = (sub["dias_historial"] <= 14).to_numpy()

    segmentos = [
        ("todos los dias", np.ones(len(sub), dtype=bool)),
        ("dia SIN registro", dia_vacio),
        ("arranque en frio (<=14 d)", poco_historial),
        ("dia normal, historial largo", ~dia_vacio & ~poco_historial),
    ]

    for etiqueta, mascara in segmentos:
        n = int(mascara.sum())
        y_seg = y.iloc[te].to_numpy()[mascara]
        if n < 30 or not (0 < y_seg.mean() < 1):
            print(f"\n{etiqueta}: {n} filas -- muestra insuficiente para evaluar")
            continue
        print(f"\n{etiqueta}  ({n} filas, tasa base {y_seg.mean():.3f})")
        print("  " + ev.CABECERA_PANEL)
        for nombre, proba in predicciones.items():
            m = ev.panel(y_seg, proba[mascara], groups=sub["child_id"].to_numpy()[mascara])
            print("  " + ev.formato_panel(nombre, m))


def comparar_k(logs, rapido=False):
    """k=5 (documento) contra el k estimado de los datos."""
    aux = logs.copy()
    aux["crisis_rate"] = aux["crisis_hoy"].astype(float)
    est = estimate_prior_strength(aux, "crisis_rate")
    if not est:
        return
    print()
    print("=" * 92)
    print("CONCENTRACION DEL PRIOR (k) -- k del documento vs k estimado de los datos")
    print("=" * 92)
    print(f"  k = {K_DEFAULT} (documento)   |   k = {est['k']:.1f} (momentos Beta-Binomial)")
    print(ev.CABECERA_PANEL)
    print("-" * 92)
    for k in sorted({K_DEFAULT, int(round(est["k"]))}):
        mu = population_baselines(logs)
        feat = build_features(logs, mu=mu, k=k).reset_index(drop=True)
        X = feat[FEATURE_NUMERIC + FEATURE_CATEGORICAL]
        y = feat[TARGET].astype(int)
        tr, te = ev.split_por_nino(feat)
        m = evaluar_modelo(f"k = {k}", build_pipeline(), X, y, tr, te, feat)
        print(ev.formato_panel(m["modelo"], m))
    print("-" * 92)
    print("  Si la diferencia es despreciable, el valor de k no es el punto")
    print("  fragil de la formulacion y k=5 se puede defender como suficiente.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="omite ablaciones y barrido de k")
    args = ap.parse_args()

    logs = pd.read_csv(DATA_PATH, parse_dates=["date"])
    mu = population_baselines(logs)
    feat = build_features(logs, mu=mu).reset_index(drop=True)
    X = feat[FEATURE_NUMERIC + FEATURE_CATEGORICAL]
    y = feat[TARGET].astype(int)

    tasa_ind = tasa_individual_expandida(logs, feat)

    print("COMPARADORES -- Antesala vs alternativas simples (Seccion 18)")
    print(f"Datos: {len(feat)} registros, {feat['child_id'].nunique()} ninos, "
          f"tasa base {y.mean():.3f}")
    print("ALCANCE: datos sinteticos. Compara arquitecturas, no capacidad clinica.")

    tr_n, te_n = ev.split_por_nino(feat)
    corre_regimen("ninos no vistos (particion por nino)", feat, X, y, tr_n, te_n,
                  tasa_ind, args.rapido)

    tr_t, te_t = ev.split_por_tiempo(feat)
    corre_regimen("dias futuros (particion temporal)", feat, X, y, tr_t, te_t,
                  tasa_ind, args.rapido)

    if not args.rapido:
        analisis_por_segmento(feat, X, y, tr_n, te_n)
        comparar_k(logs)


if __name__ == "__main__":
    main()
