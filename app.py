"""
Antesala -- interfaz de demostracion (Streamlit).

Conecta el pipeline completo: registro del dia -> ingenieria de variables ->
partial pooling -> modelo de riesgo -> suficiencia + incertidumbre ->
pregunta del dia ->
explicacion narrativa -> recomendacion.

La interfaz tiene DOS audiencias, porque las bases piden explicaciones
"comprensibles y accionables para las familias y profesionales", que no son la
misma persona ni necesitan la misma pantalla:

  - "Hoy" (familia): replica la app movil de Bluba. UNA sola pregunta al dia,
    la explicacion en lenguaje cotidiano y nada mas. Al lado, un panel
    "detras de la pantalla" que muestra que esta haciendo el modelo para
    producir esa pantalla (es una demo: el valor esta en que se vea el
    mecanismo, no solo el resultado).
  - "Panel del equipo" (profesional): triage de toda la cohorte ordenada por
    riesgo, con suficiencia e incertidumbre, cifras, variables mas
    predictivas y tendencia.
  - "Bitacora completa": las 14 variables de la Seccion 4.1, opcional.

Estetica alineada a la app real de Bluba (docs/5. Presentacion BLUBA.pdf):
fondo lavanda claro, morado corporativo, tarjetas blancas redondeadas,
opciones como tarjetas seleccionables con punto de color.

Ejecutar con:
    streamlit run app.py
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.narrative import (ETIQUETA_ORIGEN, UMBRAL_ELEVADO, UMBRAL_MODERADO,
                            build_narrative)
from core.recommendations import VARIABLE_LABELS
from core.risk_model import load_model, predict_risk
from core.question_selector import (
    REGISTRO_COSTO, REGISTRO_COSTO_DETALLE, INFORMANTE_FAMILIA, LAMBDA_CARGA)

st.set_page_config(page_title="Antesala", page_icon="🧭", layout="wide",
                   initial_sidebar_state="expanded")

DATA_PATH = "data/bitacoras.csv"

# --- Paleta ----------------------------------------------------------------
# Morado corporativo de Bluba + colores de estado RESERVADOS (verde/ambar/rojo
# solo significan bueno/atencion/critico; nunca se usan como color decorativo).
PURPLE = "#6b4ec7"
PURPLE_DARK = "#4a3aa7"
PURPLE_DEEP = "#2f2470"
PURPLE_SOFT = "#f5f2fe"
LAVENDER = "#f6f4fc"
BORDER = "#e7e3f5"
GOOD = "#0ca30c"
WARNING = "#e08600"
CRITICAL = "#d03b3b"
INK = "#171423"
INK_SOFT = "#5c5875"
MUTED = "#8b89a3"

NIVEL_COLOR = {"bajo": GOOD, "moderado": WARNING, "elevado": CRITICAL}
CONF_COLOR = {"baja": CRITICAL, "moderada": WARNING, "alta": GOOD}
# Estabilidad de la prediccion (dispersion entre arboles). Es un eje
# SEPARADO del anterior: se puede tener mucha informacion y aun asi una
# prediccion inestable, y al reves.
ESTAB_COLOR = {"estable": GOOD, "moderada": WARNING, "inestable": CRITICAL,
               "desconocida": MUTED}


# Bajo este numero de dias se considera arranque en frio: es el mismo corte
# que usa core/narrative.py para avisar que la estimacion aun no es propia
# del nino sino poblacional.
UMBRAL_HISTORIAL_CORTO = 7


def nivel_de(risk: float) -> str:
    """Nivel verbal del riesgo. Usa los MISMOS umbrales que core/narrative.py:
    si el color del anillo y el adjetivo del párrafo se calcularan por separado,
    bastaría tocar uno para que la pantalla se contradijera a sí misma."""
    if risk >= UMBRAL_ELEVADO:
        return "elevado"
    return "moderado" if risk >= UMBRAL_MODERADO else "bajo"

# --- Catalogo de variables de la bitacora (Seccion 4.1) ---------------------
# Cada entrada define como se pregunta y como se muestra. Los circulos de color
# de las opciones replican las tarjetas de la app movil de Bluba.
FIELD_CATALOG = {
    "calidad_sueno": {
        "icon": "😴", "color": "#6b4ec7", "label": "Descanso nocturno",
        "question": "¿Cómo durmió anoche?",
        "opts": ["Reparador", "Interrumpido", "Dificultad de Conciliacion"],
        "opt_labels": {"Reparador": "🟢 Reparador", "Interrumpido": "🟡 Interrumpido",
                       "Dificultad de Conciliacion": "🔴 Muy difícil"},
    },
    "modo_despertar": {
        "icon": "🌅", "color": "#e07b3c", "label": "Estado al despertar",
        "question": "¿Cómo se despertó hoy?",
        "opts": ["Tranquilo/Alegre", "Cansado/Con Sueno", "Irritable/Llorando"],
        "opt_labels": {"Tranquilo/Alegre": "🟢 Tranquilo / alegre",
                       "Cansado/Con Sueno": "🟡 Cansado / con sueño",
                       "Irritable/Llorando": "🔴 Irritable / llorando"},
    },
    "adherencia_medicacion": {
        "icon": "💊", "color": "#c98500", "label": "Adherencia a la medicación",
        "question": "¿Se tomó los medicamentos de hoy?",
        "opts": ["Si", "No", "No Aplica"],
        "opt_labels": {"Si": "🟢 Sí", "No": "🔴 No", "No Aplica": "⚪ No aplica"},
    },
    "estado_gastrointestinal": {
        "icon": "🩺", "color": "#1baf7a", "label": "Salud gastrointestinal",
        "question": "¿Cómo está su digestión hoy?",
        "opts": ["Normal", "Estrenimiento", "Diarrea"],
        "opt_labels": {"Normal": "🟢 Normal", "Estrenimiento": "🟡 Estreñimiento",
                       "Diarrea": "🔴 Diarrea / irritación"},
    },
    "nivel_regulacion_general_dia": {
        "icon": "🧭", "color": "#4a3aa7", "label": "Regulación general del día",
        "question": "¿Cómo estuvo el día en general?",
        "opts": ["Excelente", "Estable con Apoyo", "Desregulacion Frecuente"],
        "opt_labels": {"Excelente": "🟢 Excelente",
                       "Estable con Apoyo": "🟡 Estable con apoyo",
                       "Desregulacion Frecuente": "🔴 Desregulación frecuente"},
    },
    "nivel_apoyo_requerido": {
        "icon": "🤝", "color": "#2a78d6", "label": "Nivel de apoyo requerido",
        "question": "¿Cuánto apoyo necesitó para iniciar el día?",
        "opts": ["Bajo", "Medio", "Alto"],
        "opt_labels": {"Bajo": "🟢 Bajo", "Medio": "🟡 Medio", "Alto": "🔴 Alto"},
    },
    "comportamiento_observado": {
        "icon": "👀", "color": "#c0392b", "label": "Comportamiento observado",
        "question": "¿Cómo describirías su comportamiento hoy?",
        "opts": ["Estable", "Inquieto", "Desregulado"],
        "opt_labels": {"Estable": "🟢 Estable", "Inquieto": "🟡 Inquieto",
                       "Desregulado": "🔴 Desregulado"},
    },
    "estado_alerta": {
        "icon": "🔆", "color": "#d1608f", "label": "Estado de alerta",
        "question": "¿Cómo estuvo su nivel de activación?",
        "opts": ["Optimo (Regulado)", "Bajo (Letargico)", "Alto (Sobreexcitado)"],
        "opt_labels": {"Optimo (Regulado)": "🟢 Óptimo (regulado)",
                       "Bajo (Letargico)": "🔵 Bajo (letárgico)",
                       "Alto (Sobreexcitado)": "🔴 Alto (sobreexcitado)"},
    },
    "cambios_rutina": {
        "icon": "🔄", "color": "#1f8a4c", "label": "Cambios en la rutina",
        "question": "¿Hubo algún cambio en su rutina hoy?",
        "opts": ["No", "Si"],
        "opt_labels": {"No": "🟢 No, rutina habitual", "Si": "🔴 Sí, hubo cambios"},
    },
    "cambios_alimentacion": {
        "icon": "🍽️", "color": "#b8860b", "label": "Cambios en la alimentación",
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
        "icon": "💬", "color": "#7d6fe0", "label": "Interacciones sociales",
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
        "opts": None,  # control propio (elección + intensidad + tipo + resultado)
    },
}

# Orden de presentacion / prioridad clinica de las variables.
ALL_RELEVANT_FIELDS = [
    "calidad_sueno", "nivel_regulacion_general_dia", "n_eventos_desregulacion",
    "comportamiento_observado", "estado_alerta", "modo_despertar",
    "cambios_rutina", "nivel_apoyo_requerido", "estado_gastrointestinal",
    "interacciones_sociales", "adherencia_medicacion", "participacion_actividades",
    "cambios_alimentacion", "alimentacion_recreos",
]

TIPO_EVENTO_OPTS = ["Sobrecarga Sensorial", "Transicion de Actividad",
                    "Desregulacion Emocional", "Alimentacion"]
TIPO_EVENTO_LABELS = {"Sobrecarga Sensorial": "🎧 Sobrecarga sensorial",
                      "Transicion de Actividad": "🔄 Transición de actividad",
                      "Desregulacion Emocional": "⚡ Desregulación emocional",
                      "Alimentacion": "🍎 Alimentación"}
RESULTADO_OPTS = ["Regulacion Exitosa", "Regulacion Parcial", "Regulacion No Exitosa"]
RESULTADO_LABELS = {"Regulacion Exitosa": "🟢 Regulación exitosa",
                    "Regulacion Parcial": "🟡 Regulación parcial",
                    "Regulacion No Exitosa": "🔴 No se logró regular"}

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
    "dia_semana": "día de la semana",
    "es_dia_escolar": "día escolar",
    "dias_historial": "días de historial del niño",
    "sueno_ord": "calidad del sueño",
    "crisis_hoy_num": "crisis hoy",
    "profesion_sesion": "profesión que atendió la sesión",
}


def _feature_label(base: str) -> str:
    if base.startswith("antiguedad_"):
        return f"días desde el último registro de {_feature_label(base[11:])}"
    return VARIABLE_LABELS.get(base) or _EXTRA_LABELS.get(base) or base.replace("_", " ")


def prettify_feature(raw_name: str) -> str:
    """Traduce un nombre post-ColumnTransformer a una etiqueta legible."""
    name = raw_name.split("__", 1)[1] if "__" in raw_name else raw_name
    if name.startswith("missingindicator_"):
        return f"Sin registrar: {_feature_label(name[len('missingindicator_'):])}"
    if name.endswith("___missing__"):
        return f"{_feature_label(name[: -len('___missing__')])} (sin registrar)"
    for col in list(FIELD_CATALOG) + ["tipo_evento_principal", "resultado_estrategia_principal",
                                      "fuente_registro", "nivel_alerta_sesion", "profesion_sesion"]:
        if name.startswith(col + "_") and name != col:
            return f"{_feature_label(col)}: {name[len(col) + 1:].replace('_', ' ')}"
    return _feature_label(name)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"


# ============================================================== ESTILOS =====
st.markdown(f"""
<style>
  .stApp {{ background:{LAVENDER}; }}
  /* La cabecera propia de Streamlit flota sobre el contenido: con 2.2rem
     la barra de vistas quedaba cortada por debajo de ella. */
  .block-container {{ padding-top: 4.5rem; max-width: 1320px; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  #MainMenu, footer {{ visibility:hidden; }}

  /* --- Barra lateral: identidad Bluba --- */
  section[data-testid="stSidebar"] {{
      background: linear-gradient(170deg, {PURPLE_DARK} 0%, {PURPLE_DEEP} 100%);
  }}
  section[data-testid="stSidebar"] * {{ color:#ffffff !important; }}
  /* El selector y el campo de texto traen fondo blanco por defecto: sobre la
     barra morada el texto (ya forzado a blanco) quedaba invisible.
     Streamlit 1.62 monta el selector como react-aria-ComboBox, no como
     data-baseweb="select". */
  section[data-testid="stSidebar"] .react-aria-ComboBox > div,
  section[data-testid="stSidebar"] div[data-testid="stTextInputRootElement"] {{
      background: rgba(255,255,255,.14) !important;
      border: 1px solid rgba(255,255,255,.30) !important;
      border-radius: 12px !important;
  }}
  section[data-testid="stSidebar"] .react-aria-ComboBox input,
  section[data-testid="stSidebar"] div[data-testid="stTextInputRootElement"] input {{
      background: transparent !important; color:#ffffff !important;
  }}
  section[data-testid="stSidebar"] input::placeholder {{ color: rgba(255,255,255,.55) !important; }}
  section[data-testid="stSidebar"] svg {{ fill: #ffffff !important; }}
  section[data-testid="stSidebar"] div[data-testid="stExpander"] {{
      background: rgba(255,255,255,.08); border:none; border-radius:12px;
  }}
  /* Al abrirse, el encabezado del desplegable toma su fondo claro por defecto
     y el texto (forzado a blanco) se vuelve ilegible. */
  section[data-testid="stSidebar"] div[data-testid="stExpander"] summary,
  section[data-testid="stSidebar"] div[data-testid="stExpander"] details,
  section[data-testid="stSidebar"] div[data-testid="stExpander"] > div {{
      background: transparent !important; border-color: rgba(255,255,255,.18) !important;
  }}
  section[data-testid="stSidebar"] .stButton > button {{
      background: rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.3);
  }}
  /* El menu desplegable del selector se monta fuera de la barra lateral. */
  div[data-baseweb="popover"] li {{ font-size:14px; }}

  /* --- Tarjetas --- */
  .card {{
      background:#fff; border:1px solid {BORDER}; border-radius:20px;
      padding:22px 24px; box-shadow:0 2px 14px rgba(47,36,112,.06); margin-bottom:16px;
  }}
  /* Misma tarjeta, pero como CONTENEDOR de Streamlit: sirve cuando adentro
     van widgets (graficos, botones) y no solo HTML. */
  .st-key-card-pregunta, .st-key-card-detalle {{
      background:#fff; border:1px solid {BORDER}; border-radius:20px;
      padding:22px 24px 10px 24px; box-shadow:0 2px 14px rgba(47,36,112,.06);
      margin-bottom:16px;
  }}
  .eyebrow {{
      font-size:11px; font-weight:700; letter-spacing:.09em; text-transform:uppercase;
      color:{MUTED}; margin-bottom:6px;
  }}
  .h-title {{ font-size:20px; font-weight:800; color:{INK}; line-height:1.3; }}
  .h-sub {{ font-size:13.5px; color:{INK_SOFT}; margin-top:6px; line-height:1.55; }}

  /* --- Marco de telefono: la pantalla que ve la familia ---
     El relleno lateral va en el CONTENEDOR, no enumerando widgets: cualquier
     elemento que se agregue despues queda alineado solo. La barra superior lo
     compensa con margenes negativos para ocupar todo el ancho. */
  .st-key-telefono {{
      background:#fff; border:10px solid #241d3f; border-radius:38px;
      padding:0 18px 22px 18px; box-shadow:0 18px 44px rgba(47,36,112,.22);
      overflow:hidden; max-width:430px; margin:0 auto;
  }}
  .phone-bar {{
      background:linear-gradient(135deg,#7b5fd6 0%,{PURPLE_DARK} 100%);
      /* El margen inferior compensa el margin-bottom:-16px que Streamlit
         pone en los contenedores de markdown y que anula el gap del
         layout: sin esto la pregunta arranca pegada a la barra. */
      margin:0 -18px 26px -18px; padding:17px 20px 18px 20px; color:#fff;
  }}
  .phone-bar .brand {{ display:flex; align-items:center; gap:9px; font-weight:800;
      font-size:15px; letter-spacing:.2px; }}
  .phone-bar .who {{ font-size:12.5px; opacity:.86; margin-top:9px; }}
  /* El contenido del telefono no debe desbordar su marco por ningun lado --
     salvo la barra superior, que necesita salirse del relleno lateral a
     proposito para ocupar todo el ancho. */
  .st-key-telefono *:not(.phone-bar) {{ max-width:100%; }}
  .st-key-telefono .phone-bar {{ max-width:none; }}
  .st-key-telefono div[data-testid="stElementContainer"] {{ min-width:0; }}
  /* Streamlit le aplica margin-bottom:-16px a los contenedores de markdown
     para compensar el gap del layout flex. En el ULTIMO elemento de un
     contenedor propio eso se come el relleno inferior y la tarjeta queda
     pegada al borde del marco (o lo invade). Se neutraliza solo ahi. */
  .st-key-telefono > div:last-child div[data-testid="stMarkdownContainer"],
  .st-key-card-pregunta > div:last-child div[data-testid="stMarkdownContainer"] {{
      margin-bottom: 0;
  }}

  /* --- Opciones tipo Bluba: cada alternativa es una tarjeta --- */
  /* st.radio se dimensiona al CONTENIDO por defecto en Streamlit 1.62, asi que
     las tarjetas quedaban mas angostas que el boton de guardar. Se estira via
     width="stretch" en cada llamada; esto asegura ademas el grupo interno. */
  div[data-testid="stRadio"] div[role="radiogroup"] {{
      gap:8px; width:100%; align-items:stretch;
  }}
  div[data-testid="stRadio"] div[role="radiogroup"] > label {{
      border:1.5px solid {BORDER}; background:#fff; border-radius:14px;
      padding:11px 14px; margin:0; width:100%;
      transition:border-color .12s ease, background .12s ease, box-shadow .12s ease;
  }}
  div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {{
      border-color:#c3b6ee; background:#fbfaff;
  }}
  div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {{
      border-color:{PURPLE}; background:{PURPLE_SOFT};
      box-shadow:0 2px 10px rgba(107,78,199,.15);
  }}
  div[data-testid="stRadio"] div[role="radiogroup"] > label p {{
      font-size:14.5px !important; font-weight:500; color:{INK};
  }}

  /* --- Boton primario tipo Bluba ---
     Los botones de formulario usan kind="primaryFormSubmit", no "primary": sin
     incluirlos, el boton de la bitacora quedaba rectangular y plano. */
  .stButton > button[kind="primary"],
  div[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {{
      background:linear-gradient(135deg,#7b5fd6 0%,{PURPLE_DARK} 100%);
      border:none; border-radius:999px; font-weight:700; padding:11px 18px;
      box-shadow:0 4px 14px rgba(74,58,167,.30);
  }}
  .stButton > button[kind="secondary"] {{
      border-radius:999px; border:1.5px solid {BORDER}; background:#fff; color:{INK_SOFT};
  }}
  /* Deshabilitado: el degradado con texto gris claro encima se leia como un
     error de contraste. Se aplana a un morado apagado, que si dice "todavia
     no". */
  .stButton > button[kind="primary"]:disabled {{
      background:#cdc4ec; box-shadow:none; color:#ffffff; opacity:1;
  }}

  /* --- Chips de evidencia --- */
  .chip {{
      display:inline-flex; align-items:center; gap:7px; background:#fff;
      border:1px solid {BORDER}; border-radius:999px; padding:6px 12px;
      font-size:12.5px; color:{INK}; margin:0 6px 8px 0;
  }}
  .chip .src {{ font-size:10.5px; color:{MUTED}; text-transform:uppercase;
      letter-spacing:.05em; }}

  /* --- Fila de la lista de triage --- */
  .triage {{
      display:flex; align-items:center; gap:14px; background:#fff;
      border:1px solid {BORDER}; border-left-width:5px; border-radius:14px;
      padding:11px 16px; margin-bottom:8px;
  }}
  .triage .nom {{ font-weight:700; color:{INK}; font-size:14.5px; flex:1 1 auto; }}
  .triage .pct {{ font-weight:800; font-size:17px; min-width:56px; text-align:right; }}
  .triage .meta {{ font-size:11.5px; color:{MUTED}; min-width:150px; text-align:right; }}

  /* --- Navegacion segmentada --- */
  div[data-testid="stSegmentedControl"] button {{ border-radius:999px !important; }}
</style>
""", unsafe_allow_html=True)


# En una sola linea, sin saltos: este SVG se interpola dentro de otros bloques
# HTML, y un salto de linea suelto rompe el bloque de Markdown que lo contiene
# (ver la nota en risk_ring).
LOGO = (
    '<svg width="24" height="24" viewBox="0 0 40 40" fill="none" aria-hidden="true">'
    '<rect width="40" height="40" rx="12" fill="rgba(255,255,255,.18)"/>'
    '<path d="M12.5 29.5V19a7.5 7.5 0 0 1 15 0v10.5" stroke="#fff" stroke-width="2.6" '
    'stroke-linecap="round"/>'
    '<circle cx="20" cy="21.5" r="2.6" fill="#fff"/>'
    '</svg>'
)


# ======================================================= COMPONENTES HTML ===
def risk_ring(pct: int, color: str, caption: str, size: int = 148,
              sub: str = "próximas 24 h") -> str:
    """Anillo de riesgo dibujado como SVG: mas liviano y mas legible que un
    gauge generico, y se ve identico en una captura de pantalla."""
    r = 56.0
    circ = 2 * math.pi * r
    dash = circ * max(0.0, min(1.0, pct / 100.0))
    # Con pct=0 el arco mide 0, pero stroke-linecap="round" igual pinta un punto
    # suelto arriba que se lee como un dato erroneo. Bajo 1% no se dibuja arco.
    arco = "" if pct < 1 else (
        f'<circle cx="70" cy="70" r="{r}" fill="none" stroke="{color}" stroke-width="13"'
        f' stroke-linecap="round" stroke-dasharray="{dash:.1f} {circ:.1f}"'
        f' transform="rotate(-90 70 70)"/>')
    # SIN saltos de linea: Markdown corta un bloque de HTML crudo en la primera
    # linea en blanco y renderiza el resto como texto. Con el anillo al 0% el
    # arco queda vacio, y si estuviera en su propia linea dejaria justo esa
    # linea en blanco -- el SVG se imprimia literalmente en pantalla.
    return (
        '<div style="display:flex; flex-direction:column; align-items:center;">'
        f'<svg width="{size}" height="{size}" viewBox="0 0 140 140" role="img" '
        f'aria-label="{caption}: {pct} por ciento">'
        f'<circle cx="70" cy="70" r="{r}" fill="none" '
        f'stroke="{_rgba(color, 0.14)}" stroke-width="13"/>'
        f'{arco}'
        f'<text x="70" y="72" text-anchor="middle" font-size="34" font-weight="800" '
        f'fill="{INK}" font-family="system-ui,sans-serif">{pct}%</text>'
        f'<text x="70" y="92" text-anchor="middle" font-size="12" fill="{MUTED}" '
        f'font-family="system-ui,sans-serif">{sub}</text>'
        '</svg>'
        f'<div style="font-size:13px; font-weight:700; color:{color}; '
        f'margin-top:-4px;">{caption}</div>'
        '</div>'
    )


def _barra(etiqueta: str, valor: str, pct: int, color: str, margen: str = "14px") -> str:
    """Una barra etiquetada. Sin saltos de linea dentro del bloque HTML: una
    linea en blanco corta el bloque y Markdown escapa lo que viene despues
    (mismo motivo por el que risk_ring() se emite en una sola linea)."""
    return (f'<div style="margin-top:{margen};">'
            f'<div style="display:flex; justify-content:space-between; font-size:12px;'
            f' color:{INK_SOFT}; margin-bottom:5px;">'
            f'<span>{etiqueta}</span>'
            f'<span style="font-weight:700; color:{color};">{valor}</span>'
            f'</div>'
            f'<div style="height:8px; background:{_rgba(color,0.15)};'
            f' border-radius:999px; overflow:hidden;">'
            f'<div style="height:100%; width:{pct}%; background:{color};'
            f' border-radius:999px;"></div>'
            f'</div></div>')


def bloque_certeza(suf_pct: int, nivel: str, incert: dict | None) -> str:
    """Los DOS ejes que antes se mostraban fundidos en una sola "confianza".

    La revision metodologica de agosto 2026 fue explicita en este punto: que un
    registro este completo y el nino tenga historial largo no implica que el
    modelo este seguro. Son preguntas distintas y se responden por separado:

      Información disponible   completitud x historial. Cuanto SABEMOS hoy.
      Estabilidad              cuanto coinciden entre si los arboles del
                               ensamble. Cuanto se SOSTIENE la cifra.

    Se puede tener el registro completo y una prediccion inestable (el dia cae
    en una zona donde el modelo no tiene una respuesta clara), y tambien lo
    contrario. Mostrar un solo numero obligaba a elegir cual de las dos cosas
    ocultar.
    """
    color = CONF_COLOR.get(nivel, MUTED)
    html = _barra("Información disponible", f"{suf_pct}% · {nivel}", suf_pct, color)

    if incert and incert.get("disponible"):
        nivel_e = incert["nivel"]
        color_e = ESTAB_COLOR.get(nivel_e, MUTED)
        estab_pct = int(round((1.0 - incert["dispersion"]) * 100))
        html += _barra("Estabilidad de la predicción",
                       f"{estab_pct}% · {nivel_e}", estab_pct, color_e, margen="11px")
        nota = ("Son dos cosas distintas: arriba, cuánta información hay; abajo, "
                "cuánto coinciden entre sí los 400 árboles del modelo.")
    else:
        nota = "Cuánta de la información relevante está disponible hoy."

    html += (f'<div style="font-size:11px; color:{MUTED}; margin-top:7px;'
             f' line-height:1.45;">{nota}</div>')
    return html


def chips(senales, limite: int = 6) -> str:
    if not senales:
        return f"<div style='color:{MUTED}; font-size:13px;'>Sin señales de riesgo en la ventana.</div>"
    out = []
    for s in senales[:limite]:
        color = CRITICAL if s.intensidad >= 0.8 else (WARNING if s.intensidad >= 0.5 else MUTED)
        punto = f"<span style='width:8px;height:8px;border-radius:50%;background:{color};display:inline-block;'></span>"
        out.append(
            f"<span class='chip'>{punto}{s.icon} {_feature_label(s.key)}"
            f"<span class='src'>{ETIQUETA_ORIGEN[s.origen]} · {s.detalle}</span></span>")
    return "".join(out)


# ================================================================ DATOS =====
@st.cache_data(show_spinner=False)
def load_logs() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


def _key(today: dict) -> tuple:
    """Clave hashable del registro de hoy, para cachear la prediccion."""
    return tuple(sorted((k, (None if v is None else str(v))) for k, v in today.items()))


@st.cache_data(show_spinner=False)
def risk_of(child_id: str, today_key: tuple, con_pregunta: bool = True):
    today = {k: v for k, v in today_key}
    # Los valores viajan como str para poder cachear; se reconvierten los numericos.
    for num in ("n_eventos_desregulacion", "intensidad_max_desregulacion",
                "intensidad_sum_desregulacion"):
        if today.get(num) is not None:
            try:
                today[num] = float(today[num])
            except (TypeError, ValueError):
                today[num] = np.nan
    today = {k: v for k, v in today.items() if v is not None}
    return predict_risk(load_logs(), child_id, today, compute_question=con_pregunta)


def ranking_preguntas(result) -> pd.DataFrame:
    """Ranking de las variables candidatas por reduccion esperada de varianza
    del ensamble (Seccion 6.3). El motor ya las calcula TODAS para elegir el
    argmax, asi que aqui solo se ordenan: mostrar el ranking entero es lo que
    permite ver que la eleccion no es arbitraria, y no cuesta un segundo
    calculo del ensamble."""
    scores = getattr(result, "question_scores", None)
    if not scores:
        return pd.DataFrame()
    utilidades = getattr(result, "question_utilities", None) or {}
    df = pd.DataFrame({
        "campo": list(scores),
        "reduccion": list(scores.values()),
        "utilidad": [utilidades.get(c, 0.0) for c in scores],
        "carga": [REGISTRO_COSTO.get(c, 1.0) for c in scores],
    })
    # Se ordena por UTILIDAD NETA, que es el criterio real de eleccion; la
    # reduccion bruta queda como columna para poder mostrar el descuento.
    return df.sort_values("utilidad", ascending=False).reset_index(drop=True)


def texto_carga(campo: str) -> str:
    """Como se explica a una persona el costo de una pregunta."""
    d = REGISTRO_COSTO_DETALLE.get(campo)
    if not d:
        return "1 toque"
    partes = ["1 toque" if d["campos"] == 1 else f"{d['campos']} campos"]
    partes.append("lo responde la familia" if d["informante"] == INFORMANTE_FAMILIA
                  else "hay que pedirlo al colegio")
    if d["emocional"] >= 0.8:
        partes.append("obliga a volver sobre un episodio")
    elif d["emocional"] >= 0.3:
        partes.append("exige valorar el día completo")
    return " · ".join(partes)


@st.cache_data(show_spinner=False)
def historical_risk_trend(child_id: str, n_days: int = 14) -> pd.DataFrame:
    """Riesgo recalculado dia a dia usando solo el historial previo a cada dia
    (sin fuga temporal)."""
    all_logs = load_logs()
    child_rows = all_logs[all_logs["child_id"] == child_id].sort_values("date")
    points = []
    for _, row in child_rows.tail(n_days).iterrows():
        day = row["date"]
        r = predict_risk(all_logs[all_logs["date"] < day], child_id, row.to_dict(),
                         today_date=day, compute_question=False)
        points.append({"date": day, "risk": r.risk * 100, "suficiencia": r.sufficiency * 100,
                       "crisis": bool(row.get("crisis_hoy"))})
    return pd.DataFrame(points)


@st.cache_data(show_spinner="Evaluando la cohorte…")
def cohorte() -> pd.DataFrame:
    """Triage de todos los ninos: riesgo de las proximas 24 h evaluado sobre su
    ULTIMO dia registrado (el escenario operativo real de un equipo Bluba que
    revisa la manana siguiente). No usa las respuestas de la sesion: es el
    estado que traen los datos.

    El riesgo se calcula en UN solo paso vectorizado sobre toda la matriz de
    variables en vez de una llamada a predict_risk por nino (0.9 s en vez de
    6.5 s). Da exactamente el mismo numero: la ultima fila que produce
    build_features para un nino es, por construccion, la misma que produce
    build_features_for_today con su historial previo -- es la paridad
    entrenamiento/inferencia que verifica tests/test_features.py."""
    from core.features import build_features
    from core.risk_model import score_heuristic

    all_logs = load_logs()
    model = load_model()
    conf_w = model.get("confidence_weights") if model else None

    riesgo_rf: dict[str, float] = {}
    if model is not None:
        feat = build_features(all_logs, mu=model["mu"])
        ultimas = feat.loc[feat.groupby("child_id")["date"].idxmax()]
        cols = model["feature_numeric"] + model["feature_categorical"]
        probas = model["pipeline"].predict_proba(ultimas[cols])[:, 1]
        riesgo_rf = {c: float(p) for c, p in zip(ultimas["child_id"], probas)}

    filas = []
    for cid, g in all_logs.groupby("child_id"):
        g = g.sort_values("date")
        last = g.iloc[-1]
        day = last["date"]
        # La suficiencia y las faltantes salen del heuristico, que es barato y no
        # necesita el modelo (Seccion 6.2).
        h = score_heuristic(all_logs[all_logs["date"] < day], cid, last.to_dict(),
                            confidence_weights=conf_w)
        filas.append({"child_id": cid, "risk": riesgo_rf.get(cid, h.risk),
                      "sufficiency": h.sufficiency, "sufficiency_level": h.sufficiency_level,
                      "fecha": day, "n_history": h.n_history_days,
                      "faltantes": len(h.missing_relevant)})
    return pd.DataFrame(filas).sort_values("risk", ascending=False).reset_index(drop=True)


def build_today_dict(answers: dict) -> dict:
    """Convierte las respuestas guardadas en el dict que consume el modelo."""
    today = {k: v for k, v in answers.items() if v is not None and not str(k).startswith("_")}
    if answers.get("n_eventos_desregulacion") is not None:
        n_ev = answers["n_eventos_desregulacion"]
        intensidad = answers.get("_intensidad")
        today["n_eventos_desregulacion"] = n_ev
        today["intensidad_max_desregulacion"] = float(intensidad) if (n_ev and intensidad) else np.nan
        today["intensidad_sum_desregulacion"] = float(intensidad) if (n_ev and intensidad) else 0.0
        today["tipo_evento_principal"] = answers.get("_tipo_evento") if n_ev else None
        today["resultado_estrategia_principal"] = answers.get("_resultado") if n_ev else None
    return {k: v for k, v in today.items() if v is not None}


logs = load_logs()

if "answers" not in st.session_state:
    st.session_state.answers = {}
if "new_children" not in st.session_state:
    st.session_state.new_children = {}


# Nombres de fantasia para los ninos sinteticos. Un identificador como
# "nino_007" no permite juzgar si la pantalla se lee bien; un nombre con
# inicial de apellido es ademas el registro correcto para datos de salud
# (asi aparecen anonimizados en las capturas reales de Bluba).
NOMBRES_DEMO = [
    "Amelia S.", "Tomás R.", "Emilia C.", "Vicente M.", "Josefa P.",
    "Benjamín A.", "Isidora L.", "Matías V.", "Antonia G.", "Agustín F.",
    "Florencia D.", "Joaquín T.", "Catalina N.", "Lucas B.", "Renata H.",
    "Martín Q.", "Trinidad E.", "Gaspar O.", "Julieta Z.", "Emiliano I.",
    "Rafaela U.", "Bruno Y.", "Consuelo X.", "Facundo W.", "Amanda K.",
]


def display_name(cid: str) -> str:
    if cid in st.session_state.new_children:
        return st.session_state.new_children[cid]
    if cid.startswith("nino_nuevo_"):
        return f"Ingreso reciente {cid.rsplit('_', 1)[-1].lstrip('0') or '0'}"
    if cid.startswith("nino_"):
        try:
            return NOMBRES_DEMO[int(cid.rsplit("_", 1)[-1]) - 1]
        except (ValueError, IndexError):
            pass
    return cid.replace("_", " ").title()


# ============================================================== SIDEBAR =====
with st.sidebar:
    st.markdown(f"""
<div style="display:flex; align-items:center; gap:11px; margin:2px 0 4px 0;">
  {LOGO}
  <div>
    <div style="font-size:19px; font-weight:800; letter-spacing:.2px;">Antesala</div>
    <div style="font-size:11px; opacity:.75; margin-top:-2px;">para la plataforma Bluba</div>
  </div>
</div>
<div style="height:1px; background:rgba(255,255,255,.18); margin:14px 0 12px 0;"></div>
""", unsafe_allow_html=True)

    children = sorted(logs["child_id"].unique()) + list(st.session_state.new_children)
    dias_por_nino = logs.groupby("child_id").size().to_dict()

    def etiqueta_selector(cid: str) -> str:
        """Muestra los dias de historial en la propia lista. Sin esto, elegir a
        un nino de arranque en frio parecia un dato roto ('solo 2 dias') en vez
        de lo que es: el escenario 'nino nuevo' de la Seccion 3.6, incluido a
        proposito para poder demostrarlo."""
        d = dias_por_nino.get(cid, 0)
        if d >= UMBRAL_HISTORIAL_CORTO:
            return f"{display_name(cid)}  ·  {d} días"
        return f"🆕 {display_name(cid)}  ·  {d} días"

    default_idx = (children.index(st.session_state["_select_child"])
                   if st.session_state.get("_select_child") in children else 0)
    child_id = st.selectbox("Consultante", children, index=default_idx,
                            format_func=etiqueta_selector)

    n_hist = dias_por_nino.get(child_id, 0)
    if n_hist >= UMBRAL_HISTORIAL_CORTO:
        st.markdown(f"""
<div style="font-size:12.5px; opacity:.82; margin:-4px 0 12px 2px;">
  {n_hist} días de bitácora registrados
</div>""", unsafe_allow_html=True)
    else:
        # Es un caso de demostracion, no un dato incompleto por accidente.
        st.markdown(f"""
<div style="font-size:12px; line-height:1.55; background:rgba(255,255,255,.12);
            border-radius:12px; padding:10px 12px; margin:2px 0 12px 0;">
  <b>🆕 Arranque en frío</b><br>
  Solo {n_hist} días de bitácora. Se le aplica el promedio de la población
  hasta que acumule historial propio (Sección 3.6). Está incluido a propósito
  para mostrar el caso «niño nuevo».
</div>""", unsafe_allow_html=True)

    # Arranque en frio total: un nino que no existe en el CSV. El backend ya lo
    # soporta (theta cae en mu, Seccion 3.6); esto es solo el alta en la interfaz.
    with st.expander("➕  Niño o niña nuevo/a"):
        nombre_nuevo = st.text_input("Nombre", key="nuevo_nombre", placeholder="Ej: Martina R.")
        if st.button("Crear perfil", width="stretch"):
            nombre = nombre_nuevo.strip()
            if nombre:
                new_id = f"nuevo::{nombre}"
                st.session_state.new_children[new_id] = nombre
                st.session_state.answers.setdefault(new_id, {})
                st.session_state["_select_child"] = new_id
                st.rerun()

    st.markdown("<div style='height:1px; background:rgba(255,255,255,.18); margin:8px 0 12px;'></div>",
                unsafe_allow_html=True)
    _model = load_model()
    if _model is not None:
        auc = _model.get("metrics", {}).get("roc_auc")
        st.markdown(f"""
<div style="font-size:11.5px; opacity:.8; line-height:1.7;">
  <b>Modelo activo</b><br>Random Forest · 400 árboles<br>
  {'ROC AUC ' + format(auc, '.2f') + ' (niños no vistos)' if auc else ''}<br>
  {len(logs)} niño-día de entrenamiento
</div>""", unsafe_allow_html=True)
    else:
        st.warning("Sin modelo entrenado: `python core/train_model.py`")


# =============================================================== ESTADO =====
answers = st.session_state.answers.setdefault(child_id, {})
today_dict = build_today_dict(answers)
result = risk_of(child_id, _key(today_dict), True)
nombre = display_name(child_id)

# Regla de producto (Seccion 6.3): se pide UN dato por dia, no una cadena de
# preguntas. Una vez respondida, el flujo se cierra aunque queden variables sin
# registrar -- ese es el requisito de REDUCIR LA CARGA de las bases.
ya_respondio = bool(answers.get("_question_answered"))
question = None if ya_respondio else result.suggested_question

risk_pct = int(round(result.risk * 100))
# Indice de SUFICIENCIA DE INFORMACION, no "confianza": no es la probabilidad
# de que la prediccion sea correcta (ver core/risk_model.py).
suf_pct = int(round(result.sufficiency * 100))
nivel = nivel_de(result.risk)
color_riesgo = NIVEL_COLOR[nivel]

vista = st.segmented_control(
    "Vista", ["👪  Hoy · familia", "🩺  Panel del equipo", "📝  Bitácora completa"],
    default="👪  Hoy · familia", label_visibility="collapsed")
vista = vista or "👪  Hoy · familia"


# ================================================================== HOY =====
if vista.startswith("👪"):
    col_tel, col_detras = st.columns([1, 1.05], gap="large")

    # ---------- Lo que ve la familia, dentro del marco del telefono ----------
    with col_tel:
        with st.container(key="telefono"):
            st.markdown(f"""
<div class="phone-bar">
  <div class="brand">{LOGO}<span>Antesala</span></div>
  <div class="who">{nombre} · Antesala de hoy</div>
</div>""", unsafe_allow_html=True)

            narrativa = build_narrative(logs, child_id, today_dict, result,
                                        nombre=nombre, pregunta_pendiente=question)

            # Riesgo y suficiencia son numeros independientes (Seccion 6.2):
            # "un riesgo alto con informacion insuficiente no debe presentarse a
            # la familia con la misma contundencia". Mientras la suficiencia sea
            # baja el anillo va en gris: la cifra sigue visible, pero el enfasis
            # visual se lo llevan las barras, no una alarma roja que el propio
            # sistema no puede sostener.
            color_anillo = MUTED if narrativa.preliminar else color_riesgo

            def bloque_resultado():
                caption = ("Estimación preliminar" if narrativa.preliminar
                           else f"Riesgo {nivel}")
                st.markdown(risk_ring(risk_pct, color_anillo, caption), unsafe_allow_html=True)
                st.markdown(bloque_certeza(suf_pct, result.sufficiency_level,
                                           result.uncertainty),
                            unsafe_allow_html=True)
                st.markdown(f"""
<div style="background:{_rgba(color_anillo,.07)}; border:1px solid {_rgba(color_anillo,.22)};
            border-radius:16px; padding:15px 17px; margin:16px 0 6px 0;">
  <div class="eyebrow" style="color:{color_anillo};">Qué está pasando</div>
  <div style="font-size:14px; color:{INK}; line-height:1.6;">{narrativa.observacion}
      {narrativa.veredicto}</div>
</div>
<div style="background:{PURPLE_SOFT}; border-radius:16px; padding:15px 17px; margin-bottom:6px;">
  <div class="eyebrow" style="color:{PURPLE_DARK};">Qué puedes hacer hoy</div>
  <div style="font-size:14px; color:{INK}; line-height:1.6;">{narrativa.sugerencia}</div>
</div>
""", unsafe_allow_html=True)
                if narrativa.salvedad:
                    st.markdown(f"""
<div style="font-size:12.5px; color:{INK_SOFT}; background:#fff; border:1px dashed {BORDER};
            border-radius:12px; padding:11px 14px; margin-bottom:4px;">
  ⚠️ {narrativa.salvedad}
</div>""", unsafe_allow_html=True)

            # Orden de la pantalla: PRIMERO se pregunta, despues se muestra el
            # resultado. Abrir con un porcentaje antes de que la familia haya
            # registrado nada invierte el flujo real de la bitacora de Bluba
            # (se registra el dia, luego se ve la lectura) y entierra abajo lo
            # unico que la app pide hacer.
            if question:
                spec = FIELD_CATALOG[question]
                st.markdown(f"""
<div style="display:flex; align-items:center; gap:13px; margin:12px 0 16px 0;">
  <div style="width:46px;height:46px;border-radius:14px;flex:0 0 auto;display:flex;
              align-items:center;justify-content:center;font-size:22px;
              background:{_rgba(spec['color'],.14)};">{spec['icon']}</div>
  <div>
    <div class="eyebrow" style="color:{PURPLE};">La pregunta de hoy · 1 de 1</div>
    <div style="font-size:16.5px; font-weight:800; color:{INK}; line-height:1.3;">
      {spec['question']}</div>
  </div>
</div>""", unsafe_allow_html=True)

                if question == "n_eventos_desregulacion":
                    hubo = st.radio(
                        "episodio", [False, True], index=None, width="stretch",
                        format_func=lambda v: "🟢 No hubo episodios" if not v else "🔴 Sí, hubo un episodio",
                        label_visibility="collapsed", key="hero_hubo")
                    intensidad = tipo_ev = resultado = None
                    if hubo:
                        intensidad = st.slider("Intensidad del episodio", 0, 10, 5,
                                               help="0-3 leve · 4-7 moderada · 8-10 fuerte")
                        tipo_ev = st.selectbox("Tipo de evento", TIPO_EVENTO_OPTS,
                                               format_func=lambda v: TIPO_EVENTO_LABELS[v])
                        resultado = st.selectbox("Resultado de la estrategia", RESULTADO_OPTS,
                                                 format_func=lambda v: RESULTADO_LABELS[v])
                    if st.button("Guardar registro de hoy", type="primary",
                                 width="stretch", disabled=hubo is None):
                        answers["n_eventos_desregulacion"] = 1 if hubo else 0
                        answers["_intensidad"] = intensidad if hubo else None
                        answers["_tipo_evento"] = tipo_ev if hubo else None
                        answers["_resultado"] = resultado if hubo else None
                        answers["_question_answered"] = True
                        st.rerun()
                else:
                    choice = st.radio("opciones", spec["opts"], index=None, width="stretch",
                                      format_func=lambda v: spec["opt_labels"].get(v, v),
                                      label_visibility="collapsed", key=f"hero_{question}")
                    if st.button("Guardar registro de hoy", type="primary",
                                 width="stretch", disabled=choice is None):
                        answers[question] = choice
                        answers["_question_answered"] = True
                        st.rerun()

                st.markdown(f"""
<div style="font-size:11.5px; color:{MUTED}; text-align:center; margin:10px 0 4px;">
  De las {len(result.missing_relevant)} variables sin registrar, esta es la que mejor
  compensa hoy entre lo que aporta y lo que cuesta responder
  <span style="color:{INK_SOFT};">({texto_carga(question)})</span>.
  No se pedirá ninguna otra.
</div>
<div style="height:1px;background:{BORDER};margin:14px 0 12px;"></div>
<div class="eyebrow" style="text-align:center;">Lectura de hoy · antes de responder</div>
""", unsafe_allow_html=True)
                bloque_resultado()
            else:
                registradas = len(ALL_RELEVANT_FIELDS) - len(result.missing_relevant)
                total = len(ALL_RELEVANT_FIELDS)
                if not ya_respondio and result.question_method == "sin_pregunta_util":
                    # El selector evaluo TODAS las variables faltantes y ninguna
                    # reduce la incertidumbre: preguntar solo agregaria carga.
                    # Poder callarse es parte del producto, no una falla.
                    icono, fondo = "🤫", PURPLE
                    titulo = "Hoy no te preguntamos nada"
                    sub = (f"Quedan {len(result.missing_relevant)} variables sin registrar, "
                           f"pero ninguna mejoraría la estimación de hoy. "
                           f"<b>Preguntar por preguntar es la carga que Antesala existe para "
                           f"evitar</b>, así que el sistema se queda callado.")
                elif not ya_respondio:
                    icono, fondo = "✅", GOOD
                    titulo = "No hay nada que preguntar hoy"
                    sub = "Ya tenemos lo necesario para la predicción de mañana."
                elif result.sufficiency_level == "baja":
                    # Se cumple la regla de 1 pregunta al dia, pero no se finge
                    # una suficiencia que no existe: completar es una OFERTA, no
                    # un requisito. Prometer "con esto basta" cuando el motor
                    # dice lo contrario seria mentirle a la familia.
                    icono, fondo = "👍", PURPLE
                    titulo = "Listo — eso era lo más importante de hoy"
                    sub = (f"No te pediremos nada más. Con {registradas} de {total} variables "
                           f"la información disponible sigue siendo <b>baja</b>, así que la "
                           f"predicción es "
                           f"orientativa. Si tienes un minuto, <b>Bitácora completa</b> la "
                           f"afinaría — pero es opcional.")
                else:
                    icono, fondo = "🎉", GOOD
                    titulo = "¡Listo por hoy!"
                    sub = (f"Con esa respuesta la información disponible ya es "
                           f"<b>{result.sufficiency_level}</b>. Registraste {registradas} de "
                           f"{total} variables y <b>no hace falta completar el resto</b>.")
                st.markdown(f"""
<div style="text-align:center; padding:14px 8px 4px;">
  <div style="width:62px;height:62px;border-radius:50%;background:{_rgba(fondo,.14)};
              display:flex;align-items:center;justify-content:center;font-size:30px;
              margin:0 auto 12px;">{icono}</div>
  <div style="font-size:17px;font-weight:800;color:{INK};">{titulo}</div>
  <div style="font-size:13.5px;color:{INK_SOFT};margin-top:6px;line-height:1.55;">{sub}</div>
</div>
<div style="height:1px;background:{BORDER};margin:14px 0 12px;"></div>""",
                            unsafe_allow_html=True)
                bloque_resultado()
                if ya_respondio and st.button("↩️  Empezar de nuevo el registro de hoy",
                                              width="stretch"):
                    st.session_state.answers[child_id] = {}
                    st.rerun()

    # ------------------------- Detras de la pantalla ------------------------
    with col_detras:
        n_filas = f"{len(logs):,}".replace(",", ".")
        st.markdown(f"""
<div class="card">
  <div class="eyebrow">Detrás de la pantalla</div>
  <div class="h-title">Cómo se produjo esa pantalla</div>
  <div class="h-sub">La familia ve una pregunta y un párrafo. Debajo corre el pipeline
      completo: {n_filas} niño-día de bitácora, <i>partial pooling</i> por niño, un Random
      Forest de 400 árboles calibrado, y un selector que decide qué preguntar —
      o si hoy conviene no preguntar nada.</div>
</div>
<div style="background:#fffbf0; border:1px solid #f0e2bd; border-radius:14px;
            padding:12px 15px; margin:-6px 0 16px 0; font-size:12.5px;
            color:#7a6a44; line-height:1.55;">
  <b>Alcance de esta demo.</b> Corre sobre bitácoras <b>sintéticas</b>. Demuestra que el
  pipeline funciona, que reacciona a los datos que faltan, que la pregunta cambia según
  el niño y el día, y que el sistema declara cuándo no sabe. <b>No demuestra capacidad
  predictiva clínica</b>: eso requiere datos longitudinales reales de Bluba y validación
  prospectiva posterior.
</div>""", unsafe_allow_html=True)

        # 1. Senales detectadas
        _ventana_txt = ("el registro de hoy" if narrativa.ventana <= 1
                        else f"los últimos {narrativa.ventana} días")
        st.markdown(f"""
<div class="card">
  <div class="eyebrow">1 · Señales en {_ventana_txt}</div>
  <div style="margin:10px 0 2px;">{chips(narrativa.senales)}</div>
  <div style="font-size:12px; color:{MUTED}; margin-top:8px;">
    Cada señal indica de qué fuente viene (familia, colegio o equipo profesional)
    y en cuántos días de la ventana aparece.
  </div>
</div>""", unsafe_allow_html=True)

        # 2. Por que esta pregunta
        if question:
            metodo = result.question_method
            if metodo == "reduccion_varianza":
                explic = ("Se simularon los valores probables de cada variable que falta y se "
                          "midió cuánto se estrecha la dispersión entre los 400 árboles del "
                          "Random Forest con cada respuesta posible. Ganó la que más reduce "
                          "esa varianza: es la que más información aporta hoy.")
                etiqueta = "🌲 Reducción esperada de varianza del ensamble"
            else:
                explic = ("Sin modelo entrenado disponible, se usa el proxy de respaldo: la "
                          "variable sin registrar con mayor peso clínico.")
                etiqueta = "⚖️ Peso clínico (respaldo heurístico)"
            # Contenedor real (no un div de markdown) para que el grafico quede
            # DENTRO de la tarjeta: un st.plotly_chart despues de un markdown
            # cerrado se dibuja fuera del recuadro y se ve suelto.
            with st.container(key="card-pregunta"):
                st.markdown(f"""
<div class="eyebrow">2 · Por qué esta pregunta y no otra</div>
<div style="font-size:13.5px;color:{INK};font-weight:700;margin:4px 0 8px;">{etiqueta}</div>
<div class="h-sub" style="margin-top:0;">{explic}</div>
<div style="margin-top:12px; padding:11px 14px; background:{PURPLE_SOFT};
            border-radius:12px; font-size:13px; color:{INK};">
  Quedaban <b>{len(result.missing_relevant)}</b> variables sin registrar. Se pregunta
  <b>1</b>. Eso es la reducción de carga que piden las bases.
</div>""", unsafe_allow_html=True)

                # El ranking completo: la prueba de que la eleccion no es arbitraria.
                rank = ranking_preguntas(result)
                if len(rank) >= 2:
                    top = rank.head(6).iloc[::-1]
                    colores = [PURPLE if c == question else "#cfc7ee" for c in top["campo"]]
                    # Se grafica la UTILIDAD NETA, que es lo que decide de
                    # verdad. Graficar la ganancia bruta mostraria una barra
                    # ganadora que a veces NO es la pregunta elegida, y eso se
                    # ve como un error del sistema en vez de como el descuento
                    # por carga que efectivamente es.
                    fig_q = go.Figure(go.Bar(
                        x=top["utilidad"], y=[_feature_label(c) for c in top["campo"]],
                        orientation="h", marker_color=colores,
                        customdata=top[["reduccion", "carga"]].to_numpy(),
                        hovertemplate=("%{y}<br>utilidad neta: %{x:.3f}"
                                       "<br>ganancia bruta: %{customdata[0]:.2e}"
                                       "<br>carga de registro: %{customdata[1]:.2f}"
                                       "<extra></extra>")))
                    fig_q.update_layout(
                        height=34 * len(top) + 78, margin=dict(l=8, r=8, t=10, b=34),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(title="Utilidad neta = ganancia informativa − carga",
                                   gridcolor="rgba(139,137,163,.22)", zeroline=True,
                                   zerolinecolor="rgba(139,137,163,.5)"),
                        yaxis=dict(title=None), font=dict(color=INK_SOFT, size=11))
                    st.plotly_chart(fig_q, config={"displayModeBar": False})
                    st.caption(
                        f"En morado, la pregunta elegida. La barra NO es la ganancia bruta: "
                        f"es la ganancia menos la carga de responder (λ = {LAMBDA_CARGA}). "
                        f"Por eso una variable muy informativa que exige pedirle datos al "
                        f"colegio puede perder contra una que la familia responde de un "
                        f"toque. Pasa el cursor para ver los dos términos por separado.")
        else:
            if result.question_method == "sin_pregunta_util":
                _porque = (f"Se evaluaron las <b>{len(result.missing_relevant)}</b> variables "
                           f"que faltan y <b>ninguna tiene ganancia esperada positiva</b>: "
                           f"conocerlas no estrecharía la dispersión del ensamble. Como toda "
                           f"pregunta tiene un costo y ninguna tiene beneficio, la utilidad "
                           f"neta es negativa para todas y el sistema decide no preguntar. "
                           f"Sobre las bitácoras sintéticas esto pasa en 7 de 28 niños en un "
                           f"día sin ningún registro, y solo en 2 de 28 cuando ya hay tres "
                           f"campos anotados.")
            elif result.sufficiency_level == "baja":
                _porque = (f"Se pide <b>un</b> dato por día, y ya se pidió. La información "
                           f"disponible sigue siendo <b>baja</b>: la regla de una pregunta "
                           f"diaria se respeta igual, y en vez de insistir con las "
                           f"{len(result.missing_relevant)} variables restantes se marca la "
                           f"predicción como preliminar. La suficiencia sube sola a medida "
                           f"que se acumulan respuestas día tras día.")
            else:
                _porque = (f"Se pide <b>un</b> dato por día. Con la respuesta de hoy la "
                           f"información disponible ya es "
                           f"<b>{result.sufficiency_level}</b>; insistir con las "
                           f"{len(result.missing_relevant)} variables restantes agregaría "
                           f"carga de registro sin cambiar la decisión.")
            st.markdown(f"""
<div class="card">
  <div class="eyebrow">2 · Por qué no se pregunta nada más</div>
  <div class="h-sub" style="margin-top:0;">{_porque}</div>
</div>""", unsafe_allow_html=True)

        # 3. Los tres numeros que la revision metodologica pidio separar,
        #    mas la linea base del nino.
        _inc = result.uncertainty or {}
        if _inc.get("disponible"):
            _estab = f"{int(round((1 - _inc['dispersion']) * 100))}%"
            _estab_lbl = f"Estabilidad · {_inc['nivel']}"
            _estab_col = ESTAB_COLOR.get(_inc["nivel"], MUTED)
        else:
            _estab, _estab_lbl, _estab_col = "—", "Estabilidad · sin ensamble", MUTED
        cols = st.columns(4)
        for col, (valor, etiqueta, color) in zip(cols, [
            (f"{risk_pct}%", "Riesgo 24 h" + (" · preliminar" if narrativa.preliminar else ""),
             color_anillo),
            (f"{suf_pct}%", f"Información · {result.sufficiency_level}",
             CONF_COLOR[result.sufficiency_level]),
            (_estab, _estab_lbl, _estab_col),
            (f"{int((result.base_rate_bayes or 0)*100)}%", "Línea base del niño", PURPLE),
        ]):
            with col:
                st.markdown(f"""
<div class="card" style="padding:16px; text-align:center; margin-bottom:10px;">
  <div style="font-size:26px;font-weight:800;color:{color};">{valor}</div>
  <div style="font-size:11.5px;color:{MUTED};margin-top:2px;line-height:1.35;">{etiqueta}</div>
</div>""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="card">
  <div class="eyebrow">3 · El párrafo completo que genera el sistema</div>
  <div style="font-size:13.5px; color:{INK}; line-height:1.7; font-style:italic;
              border-left:3px solid {PURPLE}; padding-left:14px; margin-top:10px;">
    {narrativa.texto}
  </div>
  <div style="font-size:11.5px;color:{MUTED};margin-top:12px;">
    Estructura pedida en las bases: observación → nivel de riesgo → estrategia preventiva.
    Es una tabla de reglas determinista, no un generador de texto: cada frase es trazable
    al dato que la originó.
  </div>
</div>""", unsafe_allow_html=True)


# ================================================== PANEL DEL EQUIPO ========
elif vista.startswith("🩺"):
    st.markdown(f"""
<div class="card" style="margin-bottom:18px;">
  <div class="eyebrow">Panel del equipo profesional</div>
  <div class="h-title">Triage de la mañana</div>
  <div class="h-sub">Riesgo de desregulación en las próximas 24 h para cada consultante,
     calculado sobre su último día registrado. Ordenado por riesgo y anotado con la
     suficiencia de datos: una alerta con información insuficiente se marca como
     <b>no accionable</b> en vez
     de presentarse como certeza.</div>
</div>""", unsafe_allow_html=True)

    coh = cohorte()
    alto = coh[(coh["risk"] >= UMBRAL_ELEVADO) & (coh["sufficiency_level"] != "baja")]
    sin_datos = coh[coh["sufficiency_level"] == "baja"]

    m1, m2, m3, m4 = st.columns(4)
    for col, (valor, etiqueta, color) in zip([m1, m2, m3, m4], [
        (len(coh), "Consultantes activos", PURPLE),
        (len(alto), "Alertas accionables hoy", CRITICAL),
        (len(sin_datos), "Sin datos suficientes", WARNING),
        (f"{coh['risk'].mean():.0%}", "Riesgo promedio", INK_SOFT),
    ]):
        with col:
            st.markdown(f"""
<div class="card" style="padding:18px; text-align:center;">
  <div style="font-size:30px;font-weight:800;color:{color};">{valor}</div>
  <div style="font-size:12px;color:{MUTED};margin-top:2px;">{etiqueta}</div>
</div>""", unsafe_allow_html=True)

    c_lista, c_detalle = st.columns([1, 1.15], gap="large")

    def fila_triage(r) -> str:
        pct = int(round(r["risk"] * 100))
        col_r = NIVEL_COLOR[nivel_de(r["risk"])]
        es_actual = r["child_id"] == child_id
        if r["sufficiency_level"] == "baja":
            meta = f"⚠️ información insuficiente · {r['faltantes']} sin registrar"
            col_r = MUTED
        else:
            meta = f"información {r['sufficiency_level']} · {r['fecha']:%d %b}"
        marca = f"border:2px solid {PURPLE}; border-left-width:5px;" if es_actual else ""
        etiqueta = (f'<span style="color:{PURPLE};font-size:11px;font-weight:700;">'
                    f'· SELECCIONADO</span>' if es_actual else "")
        return f"""
<div class="triage" style="border-left-color:{col_r}; {marca}">
  <div class="nom">{display_name(r['child_id'])} {etiqueta}</div>
  <div class="meta">{meta}</div>
  <div class="pct" style="color:{col_r};">{pct}%</div>
</div>"""

    with c_lista:
        st.markdown("##### Cohorte ordenada por riesgo")
        VISIBLES = 10
        st.markdown("".join(fila_triage(r) for _, r in coh.head(VISIBLES).iterrows()),
                    unsafe_allow_html=True)
        resto = coh.iloc[VISIBLES:]
        if len(resto):
            with st.expander(f"Ver los {len(resto)} consultantes restantes"):
                st.markdown("".join(fila_triage(r) for _, r in resto.iterrows()),
                            unsafe_allow_html=True)
        st.caption("⚠️ Datos sintéticos: valida el funcionamiento del sistema, no su "
                   "capacidad predictiva clínica.")
        st.caption("Las filas grises son alertas suprimidas por información insuficiente: "
                   "el sistema avisa que no puede pronunciarse, en vez de emitir un número "
                   "que nadie debería usar.")

    with c_detalle:
        st.markdown(f"##### {nombre} · registro de hoy")
        st.caption("Este bloque usa el registro **en curso** de la pestaña «Hoy», que puede "
                   "estar incompleto. La lista de la izquierda usa el último día ya cerrado "
                   "de cada consultante, por eso las cifras pueden diferir.")
        narrativa_pro = build_narrative(logs, child_id, today_dict, result, nombre=nombre,
                                        audiencia="profesional", pregunta_pendiente=question)
        st.markdown(f"""
<div class="card">
  <div style="font-size:14px; color:{INK}; line-height:1.7;">{narrativa_pro.texto}</div>
  <div style="font-size:11.5px; color:{MUTED}; margin-top:14px; padding-top:12px;
              border-top:1px solid {BORDER}; font-family:ui-monospace,monospace;">
    {narrativa_pro.ficha_tecnica}
  </div>
</div>""", unsafe_allow_html=True)

        g1, g2 = st.columns(2)
        with g1:
            # Mismo criterio que en la vista de familia (Seccion 6.2): mientras
            # la informacion sea insuficiente el riesgo no se pinta como alarma.
            st.markdown(risk_ring(
                risk_pct, MUTED if narrativa_pro.preliminar else color_riesgo,
                "Estimación preliminar" if narrativa_pro.preliminar else f"Riesgo {nivel}",
                138), unsafe_allow_html=True)
        with g2:
            st.markdown(risk_ring(suf_pct, CONF_COLOR[result.sufficiency_level],
                                  f"Información {result.sufficiency_level}", 138,
                                  sub="de la predicción"), unsafe_allow_html=True)

        st.markdown("###### Completitud del registro de hoy")
        n_missing = len(result.missing_relevant)
        n_total = len(ALL_RELEVANT_FIELDS)
        st.progress((n_total - n_missing) / n_total,
                    text=f"{n_total - n_missing} de {n_total} variables registradas")
        cc = st.columns(2)
        for i, field in enumerate(ALL_RELEVANT_FIELDS):
            presente = field not in result.missing_relevant
            with cc[i % 2]:
                st.markdown(
                    f"<div style='font-size:12.5px;color:{INK if presente else MUTED};'>"
                    f"{'✅' if presente else '⬜'} {VARIABLE_LABELS.get(field, field)}</div>",
                    unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    d1, d2 = st.columns([1.1, 1], gap="large")

    with d1:
        st.markdown("##### Tendencia de riesgo — últimos 14 días")
        trend = historical_risk_trend(child_id, n_days=14)
        if len(trend) >= 2:
            fig = go.Figure()
            fig.add_hrect(y0=60, y1=100, fillcolor=_rgba(CRITICAL, .07), line_width=0)
            fig.add_trace(go.Scatter(x=trend["date"], y=trend["risk"], mode="lines+markers",
                                     line=dict(color=PURPLE, width=2.5), marker=dict(size=6),
                                     name="Riesgo", hovertemplate="%{x|%d %b}: %{y:.0f}%<extra></extra>"))
            crisis = trend[trend["crisis"]]
            if len(crisis):
                fig.add_trace(go.Scatter(
                    x=crisis["date"], y=crisis["risk"], mode="markers", name="Crisis registrada",
                    marker=dict(size=13, color=CRITICAL, symbol="x", line=dict(width=0)),
                    hovertemplate="%{x|%d %b}: crisis<extra></extra>"))
            fig.update_layout(height=290, margin=dict(l=8, r=8, t=8, b=8),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              xaxis=dict(gridcolor="rgba(139,137,163,.15)"),
                              yaxis=dict(title="Riesgo (%)", range=[0, 100],
                                         gridcolor="rgba(139,137,163,.22)"),
                              legend=dict(orientation="h", y=1.12, x=0),
                              font=dict(color=INK_SOFT))
            st.plotly_chart(fig, config={"displayModeBar": False})
            st.caption("Cada punto se recalcula usando solo el historial anterior a ese día "
                       "(sin fuga temporal). Las ✕ marcan días con crisis registrada: sirven "
                       "para ver si el riesgo subía antes del evento.")
        else:
            st.info("Este consultante todavía no tiene historial suficiente para una tendencia.")

    with d2:
        st.markdown("##### Variables con mayor valor predictivo")
        model = load_model()
        if model is not None and model.get("feature_importance"):
            top = model["feature_importance"][:8]
            fig_i = go.Figure(go.Bar(
                x=[v for _, v in top][::-1],
                y=[prettify_feature(n) for n, _ in top][::-1],
                orientation="h", marker_color=PURPLE,
                hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
            fig_i.update_layout(height=290, margin=dict(l=8, r=8, t=8, b=8),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                xaxis=dict(title="Importancia relativa",
                                           gridcolor="rgba(139,137,163,.22)"),
                                yaxis=dict(title=None), font=dict(color=INK_SOFT))
            st.plotly_chart(fig_i, config={"displayModeBar": False})
            st.caption("Feature importance del Random Forest entrenado. Estos mismos pesos "
                       "son los que determinan cuánto aporta cada variable al índice de "
                       "suficiencia "
                       "(Sección 6.2).")
        else:
            st.info("Entrena el modelo (`python core/train_model.py`) para ver este panel.")

    with st.expander("Ver detalle técnico de la predicción actual"):
        st.json({
            "riesgo": result.risk,
            "suficiencia_de_informacion": result.sufficiency,
            "nivel_suficiencia": result.sufficiency_level,
            "incertidumbre_predictiva": result.uncertainty,
            "utilidad_neta_preguntas": result.question_utilities,
            "dias_historial": result.n_history_days,
            "linea_base_bayesiana": result.base_rate_bayes,
            "variables_faltantes": result.missing_relevant,
            "drivers": result.drivers, "modelo_usado": result.model_used,
            "pregunta_sugerida": result.suggested_question,
            "metodo_seleccion_pregunta": result.question_method,
        })


# ================================================== BITACORA COMPLETA ======
else:
    st.markdown(f"""
<div class="card" style="margin-bottom:18px;">
  <div class="eyebrow">Registro extendido</div>
  <div class="h-title">Bitácora completa del día · {nombre}</div>
  <div class="h-sub">Las 14 variables de la bitácora de Bluba. Es <b>opcional</b>: el modelo
     funciona con datos incompletos y deja en blanco lo que no se registró — cada ausencia
     se guarda explícitamente como tal, con su antigüedad, en vez de imputarse en silencio.</div>
</div>""", unsafe_allow_html=True)

    # El registro de un episodio es un control compuesto (4 campos) y mucho mas
    # alto que un selector simple: dentro del grid de 2 columnas desalineaba
    # todas las filas siguientes. Va aparte, a lo ancho, al final.
    campos_simples = [f for f in ALL_RELEVANT_FIELDS if f != "n_eventos_desregulacion"]

    with st.form("registro_completo"):
        cols = st.columns(2, gap="medium")
        for i, field in enumerate(campos_simples):
            spec = FIELD_CATALOG[field]
            with cols[i % 2]:
                current = answers.get(field)
                opts = [None] + spec["opts"]
                st.selectbox(
                    f"{spec['icon']} {spec['question']}", opts,
                    index=opts.index(current) if current in opts else 0,
                    format_func=lambda v, s=spec: ("— sin registrar —" if v is None
                                                   else s["opt_labels"].get(v, v)),
                    key=f"full_{field}")

        # Todo el bloque va como HTML: si la linea empieza con una etiqueta HTML,
        # Markdown deja de procesar el resto y los ** se imprimen literales.
        spec_ev = FIELD_CATALOG["n_eventos_desregulacion"]
        st.markdown(
            f"<div style='height:10px'></div>"
            f"<div style='font-weight:700;font-size:15px;color:{INK};'>"
            f"{spec_ev['icon']} {spec_ev['question']}</div>",
            unsafe_allow_html=True)
        hubo_c = st.checkbox("Sí, hubo un episodio de desregulación",
                             value=bool(answers.get("n_eventos_desregulacion")))
        e1, e2, e3 = st.columns(3, gap="medium")
        with e1:
            inten_c = st.slider("Intensidad", 0, 10, int(answers.get("_intensidad") or 5))
        with e2:
            tipo_c = st.selectbox(
                "Tipo de evento", TIPO_EVENTO_OPTS, format_func=lambda v: TIPO_EVENTO_LABELS[v],
                index=TIPO_EVENTO_OPTS.index(answers["_tipo_evento"])
                if answers.get("_tipo_evento") in TIPO_EVENTO_OPTS else 0)
        with e3:
            res_c = st.selectbox(
                "Resultado", RESULTADO_OPTS, format_func=lambda v: RESULTADO_LABELS[v],
                index=RESULTADO_OPTS.index(answers["_resultado"])
                if answers.get("_resultado") in RESULTADO_OPTS else 0)

        if st.form_submit_button("Guardar bitácora completa", width="stretch",
                                 type="primary"):
            for field in campos_simples:
                val = st.session_state.get(f"full_{field}")
                if val is not None:
                    answers[field] = val
            answers["n_eventos_desregulacion"] = 1 if hubo_c else 0
            answers["_intensidad"] = inten_c if hubo_c else None
            answers["_tipo_evento"] = tipo_c if hubo_c else None
            answers["_resultado"] = res_c if hubo_c else None
            answers["_question_answered"] = True
            st.rerun()
