"""
Generador de datos sinteticos para Antesala.

Como Bluba no entrega un dataset real, este script genera bitacoras diarias
plausibles para un grupo de ninos, con las variables listadas en la Seccion 4.1
del documento tecnico. Cada nino tiene un patron base propio (para poder
probar el pooling bayesiano jerarquico) y se simulan ausencias de registro
siguiendo los tres mecanismos de la Seccion 4.4 (MCAR, MAR, MNAR).

Uso:
    python generate_synthetic_data.py --out ../data/bitacoras.csv
"""

import argparse
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

SLEEP_QUALITY = ["mala", "regular", "buena"]
WAKE_STATE = ["irritable", "neutro", "tranquilo"]
SUPPORT_LEVEL = ["alto", "medio", "bajo"]
GI_HEALTH = ["malestar", "normal"]
ALERT_STATE = ["hiperalerta", "normal", "hipoalerta"]
SOURCES = ["familia", "escuela", "terapeuta"]


def make_children(n_children: int) -> pd.DataFrame:
    """Crea el perfil base (oculto) de cada nino: su probabilidad real de
    crisis y su patron habitual de sueno. Estos valores no los ve el modelo;
    son la 'verdad' que el generador usa para simular datos realistas."""
    rows = []
    for i in range(n_children):
        child_id = f"nino_{i+1:03d}"
        base_crisis_rate = RNG.beta(2, 8)          # la mayoria baja, algunos altos
        base_sleep_hours = RNG.normal(8.5, 1.0)     # patron de sueno propio
        sensitivity_routine = RNG.uniform(0.5, 2.5)  # que tan sensible es a cambios de rutina
        rows.append({
            "child_id": child_id,
            "base_crisis_rate": base_crisis_rate,
            "base_sleep_hours": round(base_sleep_hours, 1),
            "sensitivity_routine": sensitivity_routine,
        })
    return pd.DataFrame(rows)


def simulate_missingness(row: dict, day_of_week: int) -> dict:
    """Aplica los tres mecanismos de ausencia de datos de la Seccion 4.4."""
    # MAR: se registra menos los fines de semana (no hay colegio)
    is_weekend = day_of_week >= 5
    p_missing_school_fields = 0.75 if is_weekend else 0.05

    # MNAR: si el dia fue malo (habra crisis manana), la familia registra menos
    p_missing_mnar = 0.45 if row["_will_crisis_tomorrow"] else 0.08

    # MCAR: olvido aleatorio, parejo para todas las variables
    p_missing_mcar = 0.06

    out = dict(row)
    never_null = {"child_id", "date", "crisis_24h", "fuente_registro"}
    school_fields = ["participacion_actividades", "interacciones_sociales", "alimentacion_recreos"]
    for field in list(out.keys()):
        if field.startswith("_") or field in never_null:
            continue
        p_missing = p_missing_mcar + p_missing_mnar
        if field in school_fields:
            p_missing = max(p_missing, p_missing_school_fields)
        if RNG.random() < min(p_missing, 0.9):
            out[field] = None
    return out


def generate_logs(children: pd.DataFrame, n_days: int) -> pd.DataFrame:
    logs = []
    for _, child in children.iterrows():
        cid = child["child_id"]
        base_rate = child["base_crisis_rate"]
        base_sleep = child["base_sleep_hours"]
        sens_routine = child["sensitivity_routine"]

        # Empezamos generando la serie de "hubo crisis" dia a dia (verdad oculta)
        crisis_series = []
        sleep_series = []
        routine_change_series = []

        for day in range(n_days + 1):
            routine_change = RNG.random() < 0.15
            sleep_hours = round(RNG.normal(base_sleep - (1.5 if routine_change else 0), 0.8), 1)
            sleep_series.append(max(3.0, sleep_hours))
            routine_change_series.append(routine_change)

            sleep_deficit = max(0, base_sleep - sleep_hours)
            crisis_p = base_rate + 0.08 * sleep_deficit + (0.12 * sens_routine if routine_change else 0)
            crisis_p = min(0.95, crisis_p)
            crisis_series.append(RNG.random() < crisis_p)

        for day in range(n_days):
            date = pd.Timestamp("2026-07-01") + pd.Timedelta(days=day)
            will_crisis_tomorrow = crisis_series[day + 1]

            sleep_hours = sleep_series[day]
            sleep_quality = SLEEP_QUALITY[0] if sleep_hours < base_sleep - 1.5 else (
                SLEEP_QUALITY[1] if sleep_hours < base_sleep - 0.3 else SLEEP_QUALITY[2]
            )
            wake_state = WAKE_STATE[0] if sleep_quality == "mala" else RNG.choice(WAKE_STATE, p=[0.15, 0.45, 0.40])
            support_level = SUPPORT_LEVEL[0] if wake_state == "irritable" else RNG.choice(SUPPORT_LEVEL, p=[0.15, 0.35, 0.50])
            gi_health = GI_HEALTH[0] if RNG.random() < 0.12 else GI_HEALTH[1]
            alert_state = RNG.choice(ALERT_STATE, p=[0.25, 0.55, 0.20])
            dysregulation_events = int(RNG.poisson(1.2 if crisis_series[day] else 0.2))

            row = {
                "child_id": cid,
                "date": date,
                "horas_sueno": sleep_hours,
                "calidad_sueno": sleep_quality,
                "estado_basal_despertar": wake_state,
                "nivel_apoyo_requerido": support_level,
                "salud_gastrointestinal": gi_health,
                "cambios_alimentacion": "si" if RNG.random() < 0.10 else "no",
                "cambios_rutina": "si" if routine_change_series[day] else "no",
                "comportamiento_observado": "desregulado" if dysregulation_events > 0 else "estable",
                "estado_alerta": alert_state,
                "regulaciones_desregulaciones": dysregulation_events,
                "participacion_actividades": "si" if RNG.random() < 0.8 else "no",
                "interacciones_sociales": RNG.choice(["baja", "normal", "alta"], p=[0.2, 0.6, 0.2]),
                "alimentacion_recreos": RNG.choice(["normal", "reducida"], p=[0.85, 0.15]),
                "eventos_relevantes": "" if RNG.random() < 0.9 else "evento_atipico",
                "crisis_hoy": crisis_series[day],
                "fuente_registro": RNG.choice(SOURCES, p=[0.6, 0.25, 0.15]),
                "_will_crisis_tomorrow": will_crisis_tomorrow,
                "crisis_24h": will_crisis_tomorrow,  # variable objetivo (Seccion 4.2)
            }
            row = simulate_missingness(row, date.dayofweek)
            logs.append(row)

    return pd.DataFrame(logs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_children", type=int, default=25)
    parser.add_argument("--n_days", type=int, default=45)
    parser.add_argument("--out", type=str, default="bitacoras.csv")
    args = parser.parse_args()

    children = make_children(args.n_children)
    logs = generate_logs(children, args.n_days)
    logs = logs.drop(columns=["_will_crisis_tomorrow"])

    children.to_csv(args.out.replace(".csv", "_ninos.csv"), index=False)
    logs.to_csv(args.out, index=False)
    print(f"Generados {len(logs)} registros para {args.n_children} ninos -> {args.out}")


if __name__ == "__main__":
    main()
