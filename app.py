"""
Interfaz de demostracion de Antesala (Streamlit).
Conecta el pipeline completo: registro del dia -> motor de riesgo (Seccion 6)
-> confianza -> pregunta sugerida -> recomendacion (Seccion 7).

Ejecutar con:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st

from core.risk_model import score_heuristic, suggest_question, VARIABLE_WEIGHTS, CATEGORICAL_RISK_VALUES
from core.recommendations import build_recommendation_text, VARIABLE_LABELS

st.set_page_config(page_title="Antesala", page_icon="🧭", layout="centered")

DATA_PATH = "data/bitacoras.csv"


@st.cache_data
def load_logs():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


logs = load_logs()
children = sorted(logs["child_id"].unique())

st.title("Antesala")
st.caption("Predicción de crisis conductuales — panel de registro diario")

col_a, col_b = st.columns([2, 1])
with col_a:
    child_id = st.selectbox("Niño o niña", children, format_func=lambda c: c.replace("_", " ").title())
with col_b:
    n_days = len(logs[logs["child_id"] == child_id])
    st.metric("Días de historial", n_days)

st.divider()
st.subheader("Registro de hoy")
st.caption("Deja en blanco lo que no se haya registrado — Antesala está diseñado para funcionar con datos incompletos.")

with st.form("registro_hoy"):
    c1, c2 = st.columns(2)
    with c1:
        horas_sueno = st.number_input("Horas de sueño", min_value=0.0, max_value=14.0, value=None, step=0.5, format="%.1f")
        cambios_rutina = st.selectbox("Cambios en la rutina", [None, "si", "no"], format_func=lambda v: "— sin registrar —" if v is None else v)
        estado_basal_despertar = st.selectbox("Estado al despertar", [None, "irritable", "neutro", "tranquilo"], format_func=lambda v: "— sin registrar —" if v is None else v)
        salud_gastrointestinal = st.selectbox("Salud gastrointestinal", [None, "malestar", "normal"], format_func=lambda v: "— sin registrar —" if v is None else v)
    with c2:
        regulaciones_desregulaciones = st.number_input("Eventos de desregulación hoy", min_value=0, max_value=15, value=None, step=1)
        nivel_apoyo_requerido = st.selectbox("Apoyo requerido para iniciar el día", [None, "alto", "medio", "bajo"], format_func=lambda v: "— sin registrar —" if v is None else v)
        estado_alerta = st.selectbox("Estado de alerta", [None, "hiperalerta", "normal", "hipoalerta"], format_func=lambda v: "— sin registrar —" if v is None else v)
        comportamiento_observado = st.selectbox("Comportamiento observado", [None, "desregulado", "estable"], format_func=lambda v: "— sin registrar —" if v is None else v)

    submitted = st.form_submit_button("Calcular riesgo de hoy", use_container_width=True)

if submitted:
    today = {
        "horas_sueno": horas_sueno,
        "cambios_rutina": cambios_rutina,
        "estado_basal_despertar": estado_basal_despertar,
        "salud_gastrointestinal": salud_gastrointestinal,
        "regulaciones_desregulaciones": regulaciones_desregulaciones,
        "nivel_apoyo_requerido": nivel_apoyo_requerido,
        "estado_alerta": estado_alerta,
        "comportamiento_observado": comportamiento_observado,
    }

    result = score_heuristic(logs, child_id, today)
    question = suggest_question(result)
    recommendation = build_recommendation_text(result.drivers)

    st.divider()
    st.subheader("Resultado")

    risk_pct = int(result.risk * 100)
    conf_pct = int(result.confidence * 100)

    r1, r2 = st.columns(2)
    with r1:
        st.metric("Riesgo próximas 24h", f"{risk_pct}%")
    with r2:
        st.metric("Nivel de confianza", f"{conf_pct}% ({result.confidence_level})")

    if result.confidence_level == "baja":
        st.warning("No hay suficiente información para una recomendación confiable todavía.")
    elif result.risk >= 0.6:
        st.error(f"**Riesgo alto** con confianza {result.confidence_level}.")
    else:
        st.success(f"**Riesgo bajo a moderado** con confianza {result.confidence_level}.")

    if result.drivers:
        etiquetas = [VARIABLE_LABELS.get(d, d) for d in result.drivers]
        st.write(f"**Motivo principal:** {', '.join(etiquetas)}.")

    st.write(f"**Recomendación:** {recommendation}.")

    if question:
        st.info(f"**Pregunta sugerida para hoy:** {VARIABLE_LABELS.get(question, question)} — es el dato que más ayudaría a confirmar o descartar este riesgo.")

    with st.expander("Ver detalle técnico"):
        st.json({
            "riesgo": result.risk,
            "confianza": result.confidence,
            "dias_historial": result.n_history_days,
            "variables_faltantes": result.missing_relevant,
            "drivers": result.drivers,
        })
