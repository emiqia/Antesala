"""
Interfaz de demostracion de Antesala (Streamlit).
Conecta el pipeline completo: registro del dia -> motor de riesgo (Seccion 6)
-> confianza -> pregunta del dia -> recomendacion (Seccion 7).

Tres secciones (pestanas):
  - "Hoy": el flujo principal. Muestra UNA SOLA pregunta (la del dia, Seccion 6.3)
    y nada mas. Al responderla, se actualiza la prediccion y se confirma que
    ya no hace falta registrar nada mas hoy. Esto materializa dos requisitos
    de las bases: incentivar el registro oportuno y REDUCIR LA CARGA de registro.
  - "Registro completo": la bitacora extendida con todas las variables de la
    Seccion 4.1, para quien quiera (opcionalmente) registrar mas.
  - "Riesgo": panel visual (gauges, completitud, variables mas predictivas,
    tendencia) que responde a los 6 requisitos explicitos de las bases.

Estetica alineada a la app real de Bluba (docs/5. Presentacion BLUBA.pdf):
fondo claro, morado corporativo, tarjetas redondeadas con icono en circulo.

Ejecutar con:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.risk_model import predict_risk, load_model
from core.recommendations import build_recommendation_text, VARIABLE_LABELS

st.set_page_config(page_title="Antesala", page_icon="🧭", layout="centered")

DATA_PATH = "data/bitacoras.csv"

# --- Paleta Bluba (morado corporativo) + colores de estado reservados ---
BLUBA_PURPLE = "#6b4ec7"
BLUBA_PURPLE_DARK = "#4a3aa7"
BLUBA_PURPLE_SOFT = "#f4f2fb"
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"
SEQ_BLUE = "#256abf"
MUTED = "#898781"
INK = "#0b0b0b"
INK_SOFT = "#52514e"

# --- Catalogo completo de variables de la bitacora (Seccion 4.1) -------------
# Cada entrada define como se pregunta y como se muestra. `opts[0]` es siempre
# el valor de menor riesgo (escala ordinal), salvo estado_alerta que no es
# monotona. Los iconos y colores replican las tarjetas de la app movil de Bluba.
FIELD_CATALOG = {
    "calidad_sueno": {
        "icon": "😴", "color": "#6b4ec7", "label": "Descanso nocturno",
        "question": "¿Cómo durmió anoche?",
        "opts": ["Reparador", "Interrumpido", "Dificultad de Conciliacion"],
        "opt_labels": {"Reparador": "😊 Reparador", "Interrumpido": "😐 Interrumpido",
                        "Dificultad de Conciliacion": "😣 Muy difícil"},
    },
    "modo_despertar": {
        "icon": "🌅", "color": "#eb6834", "label": "Estado al despertar",
        "question": "¿Cómo se despertó hoy?",
        "opts": ["Tranquilo/Alegre", "Cansado/Con Sueno", "Irritable/Llorando"],
        "opt_labels": {"Tranquilo/Alegre": "🙂 Tranquilo / Alegre",
                        "Cansado/Con Sueno": "😪 Cansado / Con sueño",
                        "Irritable/Llorando": "😢 Irritable / Llorando"},
    },
    "adherencia_medicacion": {
        "icon": "💊", "color": "#eda100", "label": "Adherencia a la medicación",
        "question": "¿Se tomó los medicamentos de hoy?",
        "opts": ["No Aplica", "Si", "No"],
        "opt_labels": {"No Aplica": "— No aplica", "Si": "✅ Sí", "No": "❌ No"},
    },
    "estado_gastrointestinal": {
        "icon": "🩺", "color": "#1baf7a", "label": "Salud gastrointestinal",
        "question": "¿Cómo está su digestión hoy?",
        "opts": ["Normal", "Estrenimiento", "Diarrea"],
        "opt_labels": {"Normal": "🟢 Normal", "Estrenimiento": "🟡 Estreñimiento",
                        "Diarrea": "🔴 Diarrea / Irritación"},
    },
    "nivel_regulacion_general_dia": {
        "icon": "🧭", "color": "#4a3aa7", "label": "Regulación general del día",
        "question": "¿Cómo calificarías el día en general?",
        "opts": ["Excelente", "Estable con Apoyo", "Desregulacion Frecuente"],
        "opt_labels": {"Excelente": "🟢 Excelente", "Estable con Apoyo": "🟡 Estable con apoyo",
                        "Desregulacion Frecuente": "🔴 Desregulación frecuente"},
    },
    "nivel_apoyo_requerido": {
        "icon": "🤝", "color": "#2a78d6", "label": "Nivel de apoyo requerido",
        "question": "¿Cuánto apoyo necesitó para iniciar el día?",
        "opts": ["Bajo", "Medio", "Alto"],
        "opt_labels": {"Bajo": "🟢 Bajo", "Medio": "🟡 Medio", "Alto": "🔴 Alto"},
    },
    "comportamiento_observado": {
        "icon": "👀", "color": "#e34948", "label": "Comportamiento observado",
        "question": "¿Cómo describirías su comportamiento hoy?",
        "opts": ["Estable", "Inquieto", "Desregulado"],
        "opt_labels": {"Estable": "🟢 Estable", "Inquieto": "🟡 Inquieto",
                        "Desregulado": "🔴 Desregulado"},
    },
    "estado_alerta": {
        "icon": "⚡", "color": "#e87ba4", "label": "Estado de alerta",
        "question": "¿Cómo estuvo su nivel de activación?",
        "opts": ["Optimo (Regulado)", "Bajo (Letargico)", "Alto (Sobreexcitado)"],
        "opt_labels": {"Optimo (Regulado)": "🟢 Óptimo (regulado)",
                        "Bajo (Letargico)": "🔵 Bajo (letárgico)",
                        "Alto (Sobreexcitado)": "🔴 Alto (sobreexcitado)"},
    },
    "cambios_rutina": {
        "icon": "🔄", "color": "#008300", "label": "Cambios en la rutina",
        "question": "¿Hubo algún cambio en su rutina hoy?",
        "opts": ["No", "Si"],
        "opt_labels": {"No": "🟢 No", "Si": "🔴 Sí, hubo cambios"},
    },
    "cambios_alimentacion": {
        "icon": "🍽️", "color": "#c98500", "label": "Cambios en la alimentación",
        "question": "¿Notaste cambios en su alimentación?",
        "opts": ["Sin cambios", "Menor apetito", "Selectividad aumentada"],
        "opt_labels": {"Sin cambios": "🟢 Sin cambios", "Menor apetito": "🟡 Menor apetito",
                        "Selectividad aumentada": "🔴 Más selectivo"},
    },
    "participacion_actividades": {
        "icon": "🎯", "color": "#199e70", "label": "Participación en actividades",
        "question": "¿Cómo participó en sus actividades?",
        "opts": ["Completa", "Parcial", "No participa"],
        "opt_labels": {"Completa": "🟢 Completa", "Parcial": "🟡 Parcial",
                        "No participa": "🔴 No participó"},
    },
    "interacciones_sociales": {
        "icon": "💬", "color": "#9085e9", "label": "Interacciones sociales",
        "question": "¿Cómo fueron sus interacciones con otros?",
        "opts": ["Normal", "Baja", "Evitativa"],
        "opt_labels": {"Normal": "🟢 Normales", "Baja": "🟡 Bajas",
                        "Evitativa": "🔴 Evitativas"},
    },
    "alimentacion_recreos": {
        "icon": "🥪", "color": "#d95926", "label": "Alimentación y recreos",
        "question": "¿Cómo comió en los recreos?",
        "opts": ["Normal", "Reducida", "Rechaza"],
        "opt_labels": {"Normal": "🟢 Normal", "Reducida": "🟡 Reducida",
                        "Rechaza": "🔴 Rechazó comer"},
    },
    "n_eventos_desregulacion": {
        "icon": "⚡", "color": "#d03b3b", "label": "Episodios de desregulación",
        "question": "¿Hubo algún episodio de desregulación hoy?",
        "opts": None,  # se pregunta con un control propio (checkbox + slider)
    },
}

# Orden en que se muestran/priorizan las variables clave.
ALL_RELEVANT_FIELDS = [
    "calidad_sueno", "nivel_regulacion_general_dia", "n_eventos_desregulacion",
    "comportamiento_observado", "estado_alerta", "modo_despertar",
    "cambios_rutina", "nivel_apoyo_requerido", "estado_gastrointestinal",
    "interacciones_sociales", "adherencia_medicacion", "participacion_actividades",
    "cambios_alimentacion", "alimentacion_recreos",
]

TIPO_EVENTO_OPTS = ["Sobrecarga Sensorial", "Transicion de Actividad",
                     "Desregulacion Emocional", "Alimentacion"]
RESULTADO_OPTS = ["Regulacion Exitosa", "Regulacion Parcial", "Regulacion No Exitosa"]

_EXTRA_LABELS = {
    "intensidad_max_desregulacion": "intensidad máxima del episodio",
    "intensidad_sum_desregulacion": "intensidad acumulada del día",
    "tipo_evento_principal": "tipo de evento principal",
    "resultado_estrategia_principal": "resultado de la estrategia",
    "fuente_registro": "fuente del registro",
    "theta_crisis_rate": "riesgo base bayesiano del niño",
    "theta_sueno_ord": "línea base de sueño",
    "desviacion_sueno_ord": "desviación de sueño vs. su base",
    "theta_n_eventos_desregulacion": "línea base de episodios",
    "desviacion_n_eventos_desregulacion": "desviación de episodios vs. su base",
    "sueno_ma3": "promedio de sueño (3 días)",
    "sueno_ma7": "promedio de sueño (7 días)",
    "desreg_sum3": "episodios acumulados (3 días)",
    "desreg_sum7": "episodios acumulados (7 días)",
    "dias_desde_ultima_crisis": "días desde la última crisis",
    "transicion_reciente_3d": "transición reciente (3 días)",
    "cambio_rutina_reciente_3d": "cambio de rutina reciente (3 días)",
    "antiguedad_calidad_sueno": "antigüedad del registro de sueño",
    "antiguedad_estado_gastrointestinal": "antigüedad del registro gastrointestinal",
    "dia_semana": "día de la semana",
    "es_dia_escolar": "día escolar",
    "dias_historial": "días de historial del niño",
    "sueno_ord": "calidad del sueño",
    "crisis_hoy_num": "crisis hoy",
    "nivel_alerta_sesion": "nivel de alerta en la última sesión",
    "profesion_sesion": "profesión que atendió la sesión",
    "antiguedad_nivel_alerta_sesion": "días desde la última sesión profesional",
}


def _feature_label(base_name: str) -> str:
    return VARIABLE_LABELS.get(base_name) or _EXTRA_LABELS.get(base_name) or base_name.replace("_", " ")


def prettify_feature(raw_name: str) -> str:
    """Traduce un nombre post-ColumnTransformer a una etiqueta legible."""
    name = raw_name.split("__", 1)[1] if "__" in raw_name else raw_name
    if name.startswith("missingindicator_"):
        return f"Sin registrar: {_feature_label(name[len('missingindicator_'):])}"
    if name.endswith("___missing__"):
        return f"{_feature_label(name[: -len('___missing__')])} (sin registrar)"
    for col in FIELD_CATALOG:
        if name.startswith(col + "_"):
            return f"{_feature_label(col)}: {name[len(col) + 1:]}"
    for col in ["tipo_evento_principal", "resultado_estrategia_principal", "fuente_registro",
                "nivel_alerta_sesion", "profesion_sesion"]:
        if name.startswith(col + "_"):
            return f"{_feature_label(col)}: {name[len(col) + 1:]}"
    return _feature_label(name)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"


# --- Estilos globales: fondo claro tipo Bluba -------------------------------
st.markdown(f"""
<style>
  .stApp {{ background: linear-gradient(180deg, {BLUBA_PURPLE_SOFT} 0%, #ffffff 320px); }}
  div[data-testid="stMetricValue"] {{ color: {INK}; }}
  .bluba-card {{
      background:#ffffff; border:1px solid rgba(11,11,11,0.07); border-radius:20px;
      padding:22px 24px; box-shadow:0 4px 16px rgba(74,58,167,0.08); margin-bottom:14px;
  }}
  .bluba-hero {{
      background:#ffffff; border:2px solid {_rgba(BLUBA_PURPLE, 0.35)}; border-radius:24px;
      padding:26px 28px; box-shadow:0 8px 28px rgba(74,58,167,0.14); margin-bottom:8px;
  }}
  .bluba-eyebrow {{
      font-size:12px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase;
      color:{BLUBA_PURPLE};
  }}
  .bluba-title {{ font-size:24px; font-weight:800; color:{INK}; margin-top:4px; line-height:1.25; }}
  .bluba-sub {{ font-size:14px; color:{INK_SOFT}; margin-top:6px; }}
  .bluba-icon {{
      width:60px; height:60px; border-radius:50%; display:flex; align-items:center;
      justify-content:center; font-size:30px; flex:0 0 auto;
  }}
</style>
""", unsafe_allow_html=True)


def render_hero_question(field_key: str, method: str | None = None) -> str:
    """Tarjeta grande de LA pregunta del dia (una sola)."""
    s = FIELD_CATALOG[field_key]
    if method == "reduccion_varianza":
        _tag = "🌲 Elegida por reducción de varianza del Random Forest (Sección 6.3)"
    else:
        _tag = "⚖️ Elegida por peso clínico (respaldo heurístico)"
    return f"""
<div class="bluba-hero">
  <div style="display:flex; align-items:center; gap:18px;">
    <div class="bluba-icon" style="background:{_rgba(s['color'],0.16)};">{s['icon']}</div>
    <div style="flex:1 1 auto; min-width:0;">
      <div class="bluba-eyebrow">La pregunta de hoy · 1 de 1</div>
      <div class="bluba-title">{s['question']}</div>
      <div class="bluba-sub">Es el único dato que más reduce la incertidumbre de la predicción
          de hoy. Con responder esto basta.</div>
      <div style="font-size:11px; color:{MUTED}; margin-top:8px;">{_tag}</div>
    </div>
  </div>
</div>
"""


def make_gauge(value_pct, title, thresholds, colors) -> go.Figure:
    t1, t2 = thresholds
    color = colors[0] if value_pct < t1 else (colors[1] if value_pct < t2 else colors[2])
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value_pct,
        number={"suffix": "%", "font": {"size": 38, "color": INK}},
        title={"text": title, "font": {"size": 15, "color": INK_SOFT}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": MUTED},
            "bar": {"color": color, "thickness": 0.75},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0, t1], "color": _rgba(colors[0], 0.12)},
                {"range": [t1, t2], "color": _rgba(colors[1], 0.12)},
                {"range": [t2, 100], "color": _rgba(colors[2], 0.12)},
            ],
        },
    ))
    fig.update_layout(height=210, margin=dict(l=25, r=25, t=45, b=10),
                       paper_bgcolor="rgba(0,0,0,0)")
    return fig


@st.cache_data
def load_logs():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


@st.cache_data
def historical_risk_trend(child_id: str, n_days: int = 14):
    all_logs = load_logs()
    child_rows = all_logs[all_logs["child_id"] == child_id].sort_values("date")
    points = []
    for _, row in child_rows.tail(n_days).iterrows():
        day = row["date"]
        r = predict_risk(all_logs[all_logs["date"] < day], child_id, row.to_dict(),
                          today_date=day, compute_question=False)
        points.append({"date": day, "risk": r.risk * 100, "confidence": r.confidence * 100})
    return pd.DataFrame(points)


def build_today_dict(answers: dict) -> dict:
    """Convierte las respuestas guardadas en el dict que consume el modelo."""
    today = {k: v for k, v in answers.items() if v is not None and not str(k).startswith("_")}
    if "n_eventos_desregulacion" in answers and answers["n_eventos_desregulacion"] is not None:
        n_ev = answers["n_eventos_desregulacion"]
        intensidad = answers.get("_intensidad")
        today["n_eventos_desregulacion"] = n_ev
        today["intensidad_max_desregulacion"] = float(intensidad) if (n_ev and intensidad) else np.nan
        today["intensidad_sum_desregulacion"] = float(intensidad) if (n_ev and intensidad) else 0.0
        today["tipo_evento_principal"] = answers.get("_tipo_evento") if n_ev else None
        today["resultado_estrategia_principal"] = answers.get("_resultado") if n_ev else None
    return today


logs = load_logs()

# --- Estado de sesion: respuestas por nino + ninos nuevos agregados en vivo ---
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "new_children" not in st.session_state:
    st.session_state.new_children = {}  # child_id -> nombre para mostrar

st.title("Antesala")
st.caption("Predicción temprana de crisis conductuales · complemento para la bitácora de Bluba")

# El backend (predict_risk, compute_baseline) ya funciona correctamente para
# un child_id que no tiene NINGUNA fila en `logs` -- theta cae directo en mu
# (Seccion 3.6, arranque en frio total). Lo que faltaba era la forma de
# darlo de alta en la interfaz: un nino "absolutamente nuevo" no existe en
# el CSV, asi que no puede salir del selector si no se agrega aqui.
with st.expander("➕ Registrar un niño o niña nuevo/a (arranque en frío)"):
    nuevo_nombre = st.text_input("Nombre", key="nuevo_nombre",
                                  placeholder="Ej: Martina R.")
    if st.button("Crear perfil"):
        nombre = nuevo_nombre.strip()
        if nombre:
            new_id = f"nuevo::{nombre}"
            st.session_state.new_children[new_id] = nombre
            st.session_state.answers.setdefault(new_id, {})
            st.session_state["_select_child"] = new_id
            st.rerun()

children = sorted(logs["child_id"].unique()) + list(st.session_state.new_children.keys())


def _display_name(c: str) -> str:
    if c in st.session_state.new_children:
        return f"🆕 {st.session_state.new_children[c]}"
    return c.replace("_", " ").title()


col_a, col_b = st.columns([2, 1])
with col_a:
    _default_idx = children.index(st.session_state["_select_child"]) \
        if st.session_state.get("_select_child") in children else 0
    child_id = st.selectbox("Niño o niña", children, index=_default_idx,
                             format_func=_display_name)
with col_b:
    st.metric("Días de historial", len(logs[logs["child_id"] == child_id]))

answers = st.session_state.answers.setdefault(child_id, {})
today_dict = build_today_dict(answers)
result = predict_risk(logs, child_id, today_dict)
recommendation = build_recommendation_text(result.drivers)

# --- La pregunta del dia: UNA SOLA (Seccion 6.3) ---------------------------
# Regla de producto explicita: se pide UN dato por dia, no una cadena de
# preguntas. Una vez respondida, el flujo se da por cerrado ("ya registraste
# lo de hoy") aunque queden otras variables sin registrar -- eso es
# precisamente el requisito de REDUCIR LA CARGA de registro de las bases.
# El resto de variables queda disponible en "Registro completo", opcional.
# result.suggested_question ya viene calculado por predict_risk: usa reduccion
# de varianza del ensamble (nivel principal) cuando el Random Forest esta
# disponible, y cae al proxy heuristico si no.
_pending_question = result.suggested_question
_already_answered = bool(answers.get("_question_answered"))
question = None if _already_answered else _pending_question

tab_hoy, tab_completo, tab_riesgo = st.tabs(
    ["✨ Hoy", "📝 Registro completo", "📊 Riesgo"])

# ================================================================== HOY ====
with tab_hoy:
    if question:
        st.markdown(render_hero_question(question, result.question_method), unsafe_allow_html=True)
        spec = FIELD_CATALOG[question]

        if question == "n_eventos_desregulacion":
            with st.form("hero_evento"):
                hubo = st.radio("¿Hubo algún episodio de desregulación hoy?",
                                 [False, True],
                                 format_func=lambda v: "🟢 No, ningún episodio" if not v else "🔴 Sí, hubo un episodio",
                                 label_visibility="collapsed")
                intensidad = st.slider("Intensidad del episodio", 0, 10, 5,
                                        help="0-3 leve · 4-7 moderada · 8-10 fuerte")
                c1, c2 = st.columns(2)
                with c1:
                    tipo_ev = st.selectbox("Tipo de evento", TIPO_EVENTO_OPTS)
                with c2:
                    resultado = st.selectbox("Resultado de la estrategia", RESULTADO_OPTS)
                if st.form_submit_button("Guardar respuesta", use_container_width=True, type="primary"):
                    answers["n_eventos_desregulacion"] = 1 if hubo else 0
                    answers["_intensidad"] = intensidad if hubo else None
                    answers["_tipo_evento"] = tipo_ev if hubo else None
                    answers["_resultado"] = resultado if hubo else None
                    answers["_question_answered"] = True
                    st.rerun()
        else:
            with st.form("hero_pregunta"):
                choice = st.radio(spec["question"], spec["opts"],
                                   format_func=lambda v: spec["opt_labels"].get(v, v),
                                   label_visibility="collapsed")
                if st.form_submit_button("Guardar respuesta", use_container_width=True, type="primary"):
                    answers[question] = choice
                    answers["_question_answered"] = True
                    st.rerun()

        st.caption("¿Quieres registrar más de todos modos? Ve a **📝 Registro completo** — "
                   "pero no es necesario: Antesala funciona con datos incompletos.")
    else:
        _n_reg = len(ALL_RELEVANT_FIELDS) - len(result.missing_relevant)
        _n_tot = len(ALL_RELEVANT_FIELDS)
        if not _already_answered:
            _eyebrow, _icon, _icon_bg = "Registro de hoy completo", "✅", _rgba(GOOD, 0.16)
            _titulo = "No hay nada más que preguntar hoy"
            _sub = "Ya tenemos lo necesario para una predicción confiable. Revisa el resultado abajo."
        elif result.confidence_level == "baja":
            # Honesto: cumplimos la regla de 1 pregunta/dia, pero no fingimos
            # una confianza que no tenemos (requisito de alertas de las bases).
            _eyebrow, _icon, _icon_bg = "Gracias por registrar", "👍", _rgba(BLUBA_PURPLE, 0.16)
            _titulo = "Listo — eso era lo más importante de hoy"
            _sub = (f"No te pediremos nada más hoy. Con {_n_reg} de {_n_tot} variables la "
                     f"confianza todavía es <b>baja</b>: la predicción es orientativa. Si tienes "
                     f"un minuto, <b>Registro completo</b> la haría más precisa (opcional).")
        else:
            _eyebrow, _icon, _icon_bg = "Registro de hoy listo", "🎉", _rgba(GOOD, 0.16)
            _titulo = "¡Listo por hoy!"
            _sub = (f"Con esa respuesta la predicción ya tiene confianza "
                     f"<b>{result.confidence_level}</b>. Registraste {_n_reg} de {_n_tot} "
                     f"variables — <b>no hace falta completar el resto</b>.")
        st.markdown(f"""
<div class="bluba-hero">
  <div style="display:flex; align-items:center; gap:18px;">
    <div class="bluba-icon" style="background:{_icon_bg};">{_icon}</div>
    <div>
      <div class="bluba-eyebrow">{_eyebrow}</div>
      <div class="bluba-title">{_titulo}</div>
      <div class="bluba-sub">{_sub}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Resultado compacto, siempre visible en esta pestana
    st.markdown("### Resultado de hoy")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Riesgo próximas 24h", f"{int(result.risk*100)}%")
    with m2:
        st.metric("Confianza", f"{int(result.confidence*100)}% ({result.confidence_level})")
    with m3:
        if result.base_rate_bayes is not None:
            st.metric("Riesgo base del niño", f"{int(result.base_rate_bayes*100)}%",
                       help="Línea base bayesiana jerárquica (pooling, Sección 3).")

    if result.confidence_level == "baja":
        if _already_answered:
            st.warning("⚠️ **Confianza baja** — esta predicción es orientativa. Se irá "
                        "afinando con los registros de los próximos días.")
        else:
            st.error("⚠️ **Confianza insuficiente** — responde la pregunta de arriba para "
                      "poder emitir una recomendación confiable.")
    elif result.risk >= 0.6:
        st.warning(f"**Riesgo alto** para las próximas 24h (confianza {result.confidence_level}).")
    else:
        st.success(f"Riesgo bajo a moderado (confianza {result.confidence_level}).")

    if result.drivers:
        st.write("**Motivo principal:** " +
                  ", ".join(VARIABLE_LABELS.get(d, d) for d in result.drivers) + ".")
    st.write(f"**Recomendación:** {recommendation}.")

    if answers:
        if st.button("↩️ Empezar de nuevo el registro de hoy"):
            st.session_state.answers[child_id] = {}
            st.rerun()

# ====================================================== REGISTRO COMPLETO ==
with tab_completo:
    st.subheader("Bitácora completa del día")
    st.caption("Todas las variables de la bitácora (Sección 4.1). Es **opcional**: deja en "
               "blanco lo que no se haya registrado.")

    with st.form("registro_completo"):
        for i, field in enumerate(ALL_RELEVANT_FIELDS):
            spec = FIELD_CATALOG[field]
            if field == "n_eventos_desregulacion":
                st.markdown(f"**{spec['icon']} {spec['question']}**")
                hubo_c = st.checkbox("Sí, hubo un episodio de desregulación",
                                      value=bool(answers.get("n_eventos_desregulacion")))
                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    inten_c = st.slider("Intensidad", 0, 10,
                                         int(answers.get("_intensidad") or 5))
                with cc2:
                    tipo_c = st.selectbox("Tipo", TIPO_EVENTO_OPTS,
                                           index=TIPO_EVENTO_OPTS.index(answers["_tipo_evento"])
                                           if answers.get("_tipo_evento") in TIPO_EVENTO_OPTS else 0)
                with cc3:
                    res_c = st.selectbox("Resultado", RESULTADO_OPTS,
                                          index=RESULTADO_OPTS.index(answers["_resultado"])
                                          if answers.get("_resultado") in RESULTADO_OPTS else 0)
            else:
                current = answers.get(field)
                opts = [None] + spec["opts"]
                st.selectbox(
                    f"{spec['icon']} {spec['question']}", opts,
                    index=opts.index(current) if current in opts else 0,
                    format_func=lambda v, s=spec: "— sin registrar —" if v is None else s["opt_labels"].get(v, v),
                    key=f"full_{field}",
                )

        if st.form_submit_button("Guardar bitácora completa", use_container_width=True, type="primary"):
            for field in ALL_RELEVANT_FIELDS:
                if field == "n_eventos_desregulacion":
                    answers["n_eventos_desregulacion"] = 1 if hubo_c else 0
                    answers["_intensidad"] = inten_c if hubo_c else None
                    answers["_tipo_evento"] = tipo_c if hubo_c else None
                    answers["_resultado"] = res_c if hubo_c else None
                else:
                    val = st.session_state.get(f"full_{field}")
                    if val is not None:
                        answers[field] = val
            st.rerun()

# =============================================================== RIESGO ====
with tab_riesgo:
    if question:
        st.markdown(render_hero_question(question, result.question_method), unsafe_allow_html=True)
    elif _already_answered:
        st.success("✅ Ya respondiste la pregunta de hoy — no hace falta registrar nada más.")
    else:
        st.success("✅ Todas las variables clave están registradas hoy.")

    if result.confidence_level == "baja":
        st.error("⚠️ **Confianza insuficiente** — no hay suficiente información para emitir "
                  "una recomendación confiable todavía.")
    elif result.risk >= 0.6:
        st.warning(f"**Riesgo alto** para las próximas 24h, con confianza {result.confidence_level}.")
    else:
        st.success(f"Riesgo bajo a moderado, con confianza {result.confidence_level}.")

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(make_gauge(result.risk * 100, "Riesgo próximas 24h",
                                    (30, 60), (GOOD, WARNING, CRITICAL)),
                         use_container_width=True, config={"displayModeBar": False})
    with g2:
        st.plotly_chart(make_gauge(result.confidence * 100, "Confianza de la predicción",
                                    (40, 70), (CRITICAL, WARNING, GOOD)),
                         use_container_width=True, config={"displayModeBar": False})

    k1, k2, k3 = st.columns(3)
    with k1:
        if result.base_rate_bayes is not None:
            st.metric("Riesgo base del niño", f"{int(result.base_rate_bayes*100)}%",
                       help="Línea base bayesiana jerárquica (pooling, Sección 3): probabilidad "
                            "habitual de crisis de este niño, combinando su historial propio con "
                            "el promedio poblacional.")
    with k2:
        st.metric("Modelo", "Random Forest" if result.model_used == "random_forest" else "Heurístico")
    with k3:
        st.metric("Días de historial", result.n_history_days)

    st.divider()
    st.subheader("Completitud del registro de hoy")
    n_missing = len(result.missing_relevant)
    n_total = len(ALL_RELEVANT_FIELDS)
    st.progress((n_total - n_missing) / n_total,
                 text=f"{n_total - n_missing} de {n_total} variables registradas hoy")
    cols = st.columns(2)
    for i, field in enumerate(ALL_RELEVANT_FIELDS):
        present = field not in result.missing_relevant
        with cols[i % 2]:
            st.markdown(f"{'✅' if present else '⬜'} {VARIABLE_LABELS.get(field, field)}")

    if result.drivers:
        st.write("**Motivo principal del riesgo:** " +
                  ", ".join(VARIABLE_LABELS.get(d, d) for d in result.drivers) + ".")
    st.write(f"**Recomendación:** {recommendation}.")

    st.divider()
    st.subheader("Variables con mayor valor predictivo")
    model = load_model()
    if model is not None and model.get("feature_importance"):
        top = model["feature_importance"][:8]
        fig_imp = go.Figure(go.Bar(
            x=[imp for _, imp in top][::-1],
            y=[prettify_feature(n) for n, _ in top][::-1],
            orientation="h", marker_color=BLUBA_PURPLE))
        fig_imp.update_layout(
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Importancia relativa (modelo global)",
                        gridcolor="rgba(137,135,129,0.25)"),
            yaxis=dict(title=None))
        st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})
        st.caption("Calculado sobre el Random Forest entrenado (Sección 6.1) — refleja qué "
                   "variables explican mejor el riesgo en toda la población.")
    else:
        st.caption("Entrena el modelo (`python core/train_model.py`) para ver este panel.")

    st.subheader("Tendencia de riesgo — últimos días")
    trend = historical_risk_trend(child_id, n_days=14)
    if len(trend) >= 2:
        fig_t = go.Figure()
        fig_t.add_hrect(y0=60, y1=100, fillcolor=_rgba(CRITICAL, 0.08), line_width=0)
        fig_t.add_trace(go.Scatter(x=trend["date"], y=trend["risk"], mode="lines+markers",
                                    line=dict(color=BLUBA_PURPLE, width=2), marker=dict(size=6)))
        fig_t.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             xaxis=dict(title=None, gridcolor="rgba(137,135,129,0.15)"),
                             yaxis=dict(title="Riesgo (%)", range=[0, 100],
                                         gridcolor="rgba(137,135,129,0.25)"),
                             showlegend=False)
        st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar": False})
        st.caption("Riesgo recalculado día a día usando solo el historial disponible hasta ese "
                   "momento (sin fuga temporal). La banda roja marca riesgo alto (≥60%).")

    with st.expander("Ver detalle técnico"):
        st.json({
            "riesgo": result.risk, "confianza": result.confidence,
            "dias_historial": result.n_history_days,
            "variables_faltantes": result.missing_relevant,
            "drivers": result.drivers, "modelo_usado": result.model_used,
            "pregunta_sugerida": result.suggested_question,
            "metodo_seleccion_pregunta": result.question_method,
        })
