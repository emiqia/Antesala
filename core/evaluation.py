"""
Evaluacion del modelo -- Seccion 17 del documento tecnico (revision de agosto
2026: "Ahora hay mucho detalle de arquitectura y poco sobre como sabremos si
Antesala funciona").

Este modulo concentra DOS cosas que antes estaban implicitas:

1. LOS DOS REGIMENES DE GENERALIZACION. Un solo split no responde la pregunta.
   Hay dos preguntas distintas y se contestan con particiones distintas:

     - ninos no vistos   -> particion POR NINO. Responde "cuando llega un nino
                            nuevo a Bluba, sirve?". Es el escenario de cold
                            start (Seccion 3.6).
     - dias futuros      -> particion POR TIEMPO. Responde "entrenado con lo
                            que ya paso, acierta manana?". Es el escenario de
                            uso real.

   Mezclar filas del mismo nino y del mismo periodo entre entrenamiento y test
   infla los resultados: el modelo puede memorizar la linea base del nino en
   vez de aprender el patron. La revision lo senala explicitamente.

2. EL PANEL DE METRICAS. Accuracy no se reporta a proposito. Con una tasa base
   de ~40% de dias con desregulacion, y sobre todo con la tasa mucho mas baja
   que cabe esperar en datos reales, accuracy premia al modelo que nunca avisa.
   Se reportan en cambio las metricas que importan para un sistema de alerta:

     AUROC          discriminacion global (invariante a la tasa base).
     AUPRC          discriminacion en la clase positiva, que es la que importa;
                    se compara SIEMPRE contra la tasa base, que es su piso.
     Brier          error cuadratico de la probabilidad: mide calibracion y
                    discriminacion juntas. Mas bajo es mejor.
     sensibilidad   de los episodios que ocurrieron, cuantos se avisaron.
     PPV            de los avisos emitidos, cuantos acertaron.
     falsas alertas por nino y por semana -- la metrica de CARGA. Un modelo con
                    buena sensibilidad y 4 falsas alertas semanales es
                    inusable: la familia deja de mirarlo (alert fatigue).
     no detectados  episodios sin aviso previo, en numero absoluto.

IMPORTANTE -- ALCANCE. Todo esto corre sobre datos SINTETICOS. Sirve para
verificar que el pipeline funciona, que no hay fuga temporal y que la
complejidad adicional se paga contra baselines simples. NO demuestra capacidad
predictiva clinica: los datos los genera el mismo equipo que escribe el modelo,
asi que la relacion que el modelo encuentra es, por construccion, la que se
programo. La validacion real requiere datos longitudinales de Bluba y
evaluacion prospectiva (Seccion 17, fases 2 a 4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix,
)

# Sensibilidad objetivo para fijar el umbral de alerta. En un sistema de aviso
# temprano el costo de no avisar es mayor que el de avisar de mas, pero solo
# hasta que las falsas alertas destruyen la adherencia -- por eso se reporta
# SIEMPRE junto con las falsas alertas por nino/semana, para que el compromiso
# quede a la vista y lo decida el equipo clinico, no el modelo.
SENSIBILIDAD_OBJETIVO = 0.80


def umbral_para_sensibilidad(y: np.ndarray, proba: np.ndarray,
                             objetivo: float = SENSIBILIDAD_OBJETIVO) -> float:
    """Umbral mas alto que todavia alcanza la sensibilidad objetivo.

    Se elige el umbral MAS ALTO posible porque cada punto que se baja compra
    sensibilidad a cambio de falsas alertas; una vez alcanzado el objetivo
    clinico, seguir bajando solo agrega carga.
    """
    positivos = proba[y == 1]
    if positivos.size == 0:
        return 0.5
    return float(np.quantile(positivos, 1.0 - objetivo))


def panel(y, proba, groups=None, dias=None, umbral: float | None = None,
          objetivo: float = SENSIBILIDAD_OBJETIVO) -> dict:
    """Panel completo de metricas para un conjunto de predicciones.

    `groups` (child_id) y `dias` (numero de dias distintos cubiertos) se usan
    solo para expresar las falsas alertas en la unidad que entiende una
    familia: cuantas por nino y por semana.
    """
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    tasa_base = float(y.mean()) if y.size else 0.0

    if umbral is None:
        umbral = umbral_para_sensibilidad(y, proba, objetivo)

    pred = (proba >= umbral).astype(int)
    # labels=[0,1] evita que confusion_matrix devuelva una matriz 1x1 cuando el
    # conjunto no tiene ambas clases (pasa en subconjuntos por nino).
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    sensibilidad = tp / (tp + fn) if (tp + fn) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    especificidad = tn / (tn + fp) if (tn + fp) else float("nan")

    # AUROC/AUPRC solo estan definidos si hay ambas clases.
    hay_ambas = 0 < y.mean() < 1
    auroc = float(roc_auc_score(y, proba)) if hay_ambas else float("nan")
    auprc = float(average_precision_score(y, proba)) if hay_ambas else float("nan")

    # Falsas alertas por nino y por semana.
    n_ninos = int(pd.Series(groups).nunique()) if groups is not None else 1
    if dias is None and groups is not None:
        dias = len(y) / max(n_ninos, 1)
    semanas = (dias / 7.0) if dias else float("nan")
    fa_nino_semana = fp / (n_ninos * semanas) if semanas and semanas > 0 else float("nan")

    return {
        "n": int(y.size),
        "n_ninos": n_ninos,
        "tasa_base": tasa_base,
        "umbral": float(umbral),
        "auroc": auroc,
        "auprc": auprc,
        "auprc_lift": (auprc / tasa_base) if tasa_base > 0 else float("nan"),
        "brier": float(brier_score_loss(y, proba)),
        "sensibilidad": float(sensibilidad),
        "ppv": float(ppv),
        "especificidad": float(especificidad),
        "falsas_alertas": int(fp),
        "fa_por_nino_semana": float(fa_nino_semana),
        "no_detectados": int(fn),
        "episodios": int(tp + fn),
    }


def calibracion(y, proba, bins: int = 10) -> pd.DataFrame:
    """Tabla de fiabilidad (reliability diagram en forma de tabla).

    Agrupa las predicciones en deciles de probabilidad y compara, en cada
    grupo, la probabilidad media predicha contra la frecuencia observada. Un
    modelo calibrado las tiene pegadas: cuando dice 70%, ocurre ~70% de las
    veces. Es lo que permite mostrar el numero como porcentaje sin mentir.
    """
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    df = pd.DataFrame({"y": y, "p": proba})
    df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(
        n=("y", "size"), predicho=("p", "mean"), observado=("y", "mean"))
    g["brecha"] = g["observado"] - g["predicho"]
    return g.reset_index(drop=True)


def error_calibracion_esperado(y, proba, bins: int = 10) -> float:
    """ECE: brecha media entre lo predicho y lo observado, ponderada por el
    tamano de cada grupo. Un solo numero para comparar calibraciones."""
    tabla = calibracion(y, proba, bins)
    if tabla.empty:
        return float("nan")
    peso = tabla["n"] / tabla["n"].sum()
    return float((peso * tabla["brecha"].abs()).sum())


def split_por_nino(feat: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
    """Particion por nino: el test tiene ninos que el modelo nunca vio."""
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(feat, groups=feat["child_id"]))


def split_por_tiempo(feat: pd.DataFrame, test_size: float = 0.25):
    """Particion temporal: el test son los ULTIMOS dias de cada nino.

    Se corta por nino y no por fecha global porque los ninos de arranque en
    frio tienen historiales de largo muy distinto; un corte por fecha global
    los dejaria enteros de un lado. El principio que importa se mantiene: en
    el test no hay ningun dia anterior a los dias de entrenamiento del mismo
    nino, asi que no se puede predecir el pasado con el futuro.
    """
    feat = feat.sort_values(["child_id", "date"])
    idx_train, idx_test = [], []
    for _, grupo in feat.groupby("child_id", sort=False):
        posiciones = feat.index.get_indexer(grupo.index)
        corte = int(len(posiciones) * (1 - test_size))
        # Un nino con 2-3 dias de historial no aporta un test temporal util:
        # va entero a entrenamiento.
        if corte < 1 or len(posiciones) - corte < 1:
            idx_train.extend(posiciones)
            continue
        idx_train.extend(posiciones[:corte])
        idx_test.extend(posiciones[corte:])
    return np.array(idx_train), np.array(idx_test)


def formato_panel(nombre: str, m: dict) -> str:
    """Una fila de tabla lista para imprimir en consola."""
    def f(x, d=3):
        return "  n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"
    return (f"{nombre:<26s} {f(m['auroc'])}  {f(m['auprc'])}  {f(m['brier'])}  "
            f"{f(m['sensibilidad'])}  {f(m['ppv'])}  {f(m['fa_por_nino_semana'], 2)}  "
            f"{m['no_detectados']:>4d}/{m['episodios']:<5d}")


CABECERA_PANEL = (f"{'modelo':<26s} {'AUROC':>5s}  {'AUPRC':>5s}  {'Brier':>5s}  "
                  f"{'Sens.':>5s}  {'PPV':>5s}  {'FA/n/sem':>8s}  {'no det.':>10s}")
