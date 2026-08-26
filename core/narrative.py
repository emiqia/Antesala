"""
Explicacion narrativa del riesgo (Seccion 7 del documento tecnico).

Las bases del desafio no piden solo un numero: piden "explicaciones
comprensibles y accionables para las familias y profesionales", y dan el
formato exacto esperado como ejemplo:

    "Durante los ultimos tres dias Juan ha presentado alteraciones en el
     sueno, mayor irritabilidad en el colegio y cambios en su rutina
     habitual. Se detecta un riesgo alto de desregulacion para las proximas
     24 horas. Considere anticipar apoyos visuales, espacios de regulacion y
     disminuir estimulos ambientales."

Este modulo produce exactamente esa estructura de tres frases
(observacion -> veredicto -> sugerencia) a partir del historial real del nino
y del resultado del motor de riesgo.

Decisiones de diseno:

1. Es una TABLA DE REGLAS deterministica, no otro modelo. Igual que
   core/recommendations.py: la explicacion tiene que ser auditable por un
   profesional, reproducible y trazable a los datos que la originaron. Un
   generador de texto probabilistico podria afirmar cosas que los datos no
   sostienen, que es exactamente lo contrario de lo que pide el criterio de
   explicabilidad de las bases.

2. Cada senal declara su ORIGEN (familia / colegio / equipo profesional),
   porque los datos de la plataforma Bluba vienen de esas tres fuentes y la
   frase cambia segun quien lo observo ("mayor irritabilidad EN EL COLEGIO").

3. Hay dos audiencias (Seccion 7): `familia` (lenguaje cotidiano, sin cifras
   ni jerga) y `profesional` (misma narrativa + cifras, modelo usado y
   procedencia del dato). Es el mismo hecho contado para quien tiene que
   actuar hoy y para quien tiene que decidir clinicamente.

4. Si el modelo da riesgo alto pero no hay senales recientes que lo
   expliquen, se dice explicitamente en vez de inventar una causa.

Nota de estilo: los comentarios van en ASCII (convencion del repositorio),
pero TODO texto que termina en pantalla va acentuado correctamente.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .recommendations import (
    BIBLIOTECA, RECOMMENDATIONS, DEFAULT_RECOMMENDATION, VARIABLE_LABELS)

# Ventana por defecto: "durante los ultimos tres dias" del ejemplo de las bases.
VENTANA_DIAS = 3

# Origen del dato dentro de la plataforma Bluba. Determina como se redacta la
# frase y que etiqueta de procedencia se muestra en la interfaz.
ORIGEN_FAMILIA = "familia"
ORIGEN_COLEGIO = "colegio"
ORIGEN_PROFESIONAL = "profesional"

_SUFIJO_ORIGEN = {
    ORIGEN_FAMILIA: "",
    ORIGEN_COLEGIO: " en el colegio",
    ORIGEN_PROFESIONAL: " en sesión",
}

ETIQUETA_ORIGEN = {
    ORIGEN_FAMILIA: "familia",
    ORIGEN_COLEGIO: "colegio",
    ORIGEN_PROFESIONAL: "equipo profesional",
}


# --- Tabla de senales -------------------------------------------------------
# Una entrada por tema observable. `valores` mapea el valor crudo de la
# bitacora (tal como lo guarda Bluba, sin tildes) a su intensidad 0-1.
# `frase` es como se nombra la senal dentro del parrafo; `protector` es el
# valor que, cuando aparece, cuenta como factor protector.
THEMES = [
    dict(
        key="calidad_sueno", origen=ORIGEN_FAMILIA, icon="😴", peso=3.0,
        valores={"Dificultad de Conciliacion": 1.0, "Interrumpido": 0.6},
        frase="alteraciones en el sueño",
        frase_una="una noche de sueño alterado",
        protector="Reparador", frase_protectora="un sueño reparador",
    ),
    dict(
        key="nivel_regulacion_general_dia", origen=ORIGEN_FAMILIA, icon="🧭", peso=3.0,
        valores={"Desregulacion Frecuente": 1.0, "Estable con Apoyo": 0.4},
        frase="días con desregulación frecuente",
        frase_una="un día con desregulación frecuente",
        protector="Excelente", frase_protectora="una regulación general excelente",
    ),
    dict(
        key="n_eventos_desregulacion", origen=ORIGEN_FAMILIA, icon="⚡", peso=2.5,
        numerico=True, umbral=1.0, escala=3.0,
        frase="episodios de desregulación",
        frase_una="un episodio de desregulación",
        protector=0.0, frase_protectora="días sin episodios de desregulación",
    ),
    dict(
        key="comportamiento_observado", origen=ORIGEN_COLEGIO, icon="👀", peso=2.0,
        valores={"Desregulado": 1.0, "Inquieto": 0.5},
        frase="mayor irritabilidad",
        frase_una="un día de mayor irritabilidad",
        protector="Estable", frase_protectora="una conducta estable",
    ),
    dict(
        key="estado_alerta", origen=ORIGEN_COLEGIO, icon="🔆", peso=1.8,
        valores={"Alto (Sobreexcitado)": 1.0, "Bajo (Letargico)": 0.8},
        frase="un estado de alerta fuera de su rango habitual",
        frase_una="un día con el estado de alerta alterado",
        protector="Optimo (Regulado)", frase_protectora="un estado de alerta óptimo",
    ),
    dict(
        key="cambios_rutina", origen=ORIGEN_FAMILIA, icon="🔄", peso=1.8,
        valores={"Si": 1.0},
        frase="cambios en su rutina habitual",
        frase_una="un cambio en su rutina habitual",
        protector="No", frase_protectora="su rutina sin cambios",
    ),
    dict(
        key="modo_despertar", origen=ORIGEN_FAMILIA, icon="🌅", peso=1.5,
        valores={"Irritable/Llorando": 1.0, "Cansado/Con Sueno": 0.5},
        frase="despertares irritables o costosos",
        frase_una="un despertar irritable",
        protector="Tranquilo/Alegre", frase_protectora="despertares tranquilos",
    ),
    dict(
        key="estado_gastrointestinal", origen=ORIGEN_FAMILIA, icon="🩺", peso=1.5,
        valores={"Diarrea": 1.0, "Estrenimiento": 0.4},
        frase="molestias gastrointestinales",
        frase_una="una molestia gastrointestinal",
        protector="Normal", frase_protectora="buena salud gastrointestinal",
    ),
    dict(
        key="nivel_apoyo_requerido", origen=ORIGEN_FAMILIA, icon="🤝", peso=1.3,
        valores={"Alto": 1.0, "Medio": 0.5},
        frase="mayor necesidad de apoyo para iniciar el día",
        frase_una="un día que necesitó más apoyo para empezar",
        protector="Bajo", frase_protectora="autonomía para iniciar el día",
    ),
    dict(
        key="interacciones_sociales", origen=ORIGEN_COLEGIO, icon="💬", peso=1.2,
        valores={"Evitativa": 1.0, "Baja": 0.6},
        frase="retraimiento en sus interacciones",
        frase_una="un día de retraimiento social",
        protector="Normal", frase_protectora="interacciones sociales normales",
    ),
    dict(
        key="adherencia_medicacion", origen=ORIGEN_FAMILIA, icon="💊", peso=1.0,
        valores={"No": 1.0},
        frase="dosis de medicación sin administrar",
        frase_una="una dosis de medicación sin administrar",
        protector="Si", frase_protectora="la medicación al día",
    ),
    dict(
        key="participacion_actividades", origen=ORIGEN_COLEGIO, icon="🎯", peso=1.0,
        valores={"No participa": 1.0, "Parcial": 0.5},
        frase="menor participación en sus actividades",
        frase_una="un día de menor participación",
        protector="Completa", frase_protectora="participación completa en sus actividades",
    ),
    dict(
        key="cambios_alimentacion", origen=ORIGEN_FAMILIA, icon="🍽️", peso=0.8,
        valores={"Selectividad aumentada": 1.0, "Menor apetito": 0.6},
        frase="cambios en su alimentación",
        frase_una="un cambio en su alimentación",
        protector="Sin cambios", frase_protectora="su alimentación sin cambios",
    ),
    dict(
        key="alimentacion_recreos", origen=ORIGEN_COLEGIO, icon="🥪", peso=0.8,
        valores={"Rechaza": 1.0, "Reducida": 0.5},
        frase="rechazo de comida en los recreos",
        frase_una="un recreo con rechazo de comida",
        protector="Normal", frase_protectora="buena alimentación en los recreos",
    ),
    dict(
        key="nivel_alerta_sesion", origen=ORIGEN_PROFESIONAL, icon="🧑‍⚕️", peso=1.6,
        valores={"Alto (Sobreexcitado)": 1.0, "Bajo (Letargico)": 0.8},
        frase="alertas registradas por el equipo profesional",
        frase_una="una alerta registrada por el equipo profesional",
        protector="Optimo (Regulado)", frase_protectora="sesiones profesionales sin alertas",
    ),
]

_THEME_BY_KEY = {t["key"]: t for t in THEMES}

# Umbrales de verbalizacion del riesgo. Coinciden con los cortes que usa la
# interfaz para el color del indicador (bajo / moderado / elevado).
UMBRAL_MODERADO = 0.30
UMBRAL_ELEVADO = 0.60


@dataclass
class Signal:
    """Una senal detectada en la ventana reciente, con su evidencia."""
    key: str
    icon: str
    origen: str
    frase: str            # como se nombra dentro del parrafo
    detalle: str          # evidencia concreta, para mostrar como chip en la UI
    intensidad: float     # 0-1, promedio de los dias observados
    dias: int             # cuantos dias de la ventana la presentan
    dias_observados: int  # cuantos dias de la ventana tienen el dato registrado
    score: float          # prioridad final (peso clinico x intensidad x recencia)
    presente_hoy: bool = False  # la senal sigue activa en el registro de hoy
    protectora: bool = False


@dataclass
class Narrative:
    """Explicacion completa, ya redactada y tambien desagregada en partes."""
    texto: str                       # el parrafo completo (formato de las bases)
    observacion: str                 # frase 1: que se observo
    veredicto: str                   # frase 2: nivel de riesgo
    sugerencia: str                  # frase 3: que hacer
    salvedad: str | None             # frase 4: aviso de informacion insuficiente
    nivel: str                       # "bajo" | "moderado" | "elevado"
    senales: list[Signal] = field(default_factory=list)
    protectores: list[Signal] = field(default_factory=list)
    ventana: int = VENTANA_DIAS
    sin_causa_aparente: bool = False  # riesgo alto sin senales que lo expliquen
    preliminar: bool = False          # suficiencia baja: no es una prediccion confiable
    ficha_tecnica: str | None = None  # linea de cifras (solo audiencia profesional)


# --- utilidades -------------------------------------------------------------

def _val(row: dict, campo: str):
    """Valor de un campo tratando NaN/None/'' como ausente."""
    v = row.get(campo)
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, str) and not v.strip():
        return None
    return v


def _unir(frases: list[str]) -> str:
    """'a', 'b', 'c' -> 'a, b y c' (coordinacion en castellano)."""
    frases = [f for f in frases if f]
    if not frases:
        return ""
    if len(frases) == 1:
        return frases[0]
    return ", ".join(frases[:-1]) + " y " + frases[-1]


def _ventana(logs: pd.DataFrame, child_id: str, today: dict, ventana: int) -> list[dict]:
    """Los ultimos `ventana` dias del nino: los (ventana-1) dias ya registrados
    en su bitacora, mas el registro parcial de hoy."""
    dias: list[dict] = []
    if logs is not None and len(logs) and "child_id" in logs.columns:
        hist = logs[logs["child_id"] == child_id]
        if len(hist):
            hist = hist.sort_values("date").tail(max(0, ventana - 1))
            dias = [r.to_dict() for _, r in hist.iterrows()]
    dias.append(dict(today or {}))
    return dias


def _detectar(dias: list[dict], theme: dict) -> tuple[Signal | None, Signal | None]:
    """Evalua un tema sobre la ventana. Devuelve (senal_de_riesgo, protector);
    cualquiera de los dos puede ser None."""
    campo = theme["key"]
    intensidades: list[float] = []
    observados = 0
    protectores = 0
    hoy_en_riesgo = False
    hoy_protector = False
    ultimo = len(dias) - 1

    for i, d in enumerate(dias):
        v = _val(d, campo)
        if v is None:
            continue
        observados += 1
        if theme.get("numerico"):
            try:
                num = float(v)
            except (TypeError, ValueError):
                observados -= 1
                continue
            if num >= theme["umbral"]:
                intensidades.append(min(1.0, num / theme["escala"]))
                hoy_en_riesgo = hoy_en_riesgo or i == ultimo
            elif num <= theme["protector"]:
                protectores += 1
                hoy_protector = hoy_protector or i == ultimo
        else:
            inten = theme["valores"].get(v)
            if inten:
                intensidades.append(inten)
                hoy_en_riesgo = hoy_en_riesgo or i == ultimo
            elif v == theme["protector"]:
                protectores += 1
                hoy_protector = hoy_protector or i == ultimo

    senal = None
    if intensidades:
        n = len(intensidades)
        media = sum(intensidades) / n
        frase = theme["frase_una"] if n == 1 else theme["frase"]
        frase += _SUFIJO_ORIGEN[theme["origen"]]
        if observados <= 1:
            detalle = "registrado hoy" if hoy_en_riesgo else "registrado un día"
        elif hoy_en_riesgo:
            detalle = f"{n} de {observados} días registrados, incluido hoy"
        else:
            detalle = f"{n} de {observados} días registrados, no hoy"
        # Recencia: predecimos las proximas 24 horas, asi que una senal que
        # sigue presente HOY pesa mas que una que ya se resolvio ayer.
        recencia = 1.0 if hoy_en_riesgo else 0.55
        senal = Signal(
            key=campo, icon=theme["icon"], origen=theme["origen"], frase=frase,
            detalle=detalle, intensidad=round(media, 2), dias=n,
            dias_observados=observados, presente_hoy=hoy_en_riesgo,
            score=theme["peso"] * media * (1.0 + 0.25 * (n - 1)) * recencia,
        )

    protector = None
    if protectores and not intensidades:
        protector = Signal(
            key=campo, icon=theme["icon"], origen=theme["origen"],
            frase=theme["frase_protectora"] + _SUFIJO_ORIGEN[theme["origen"]],
            detalle=f"{protectores} de {observados} días registrados",
            intensidad=0.0, dias=protectores, dias_observados=observados,
            score=theme["peso"], presente_hoy=hoy_protector, protectora=True,
        )
    return senal, protector


def _nivel(risk: float) -> str:
    if risk >= UMBRAL_ELEVADO:
        return "elevado"
    if risk >= UMBRAL_MODERADO:
        return "moderado"
    return "bajo"


def _acciones(senales: list[Signal], drivers: list[str], maximo: int = 3,
              excluidas: set[str] | None = None) -> list[str]:
    """Estrategias preventivas, en el mismo orden de prioridad que las senales.
    Reutiliza la biblioteca auditable de core/recommendations.py (Seccion 12).

    `excluidas` son ids de recomendacion apagadas para este nino (Seccion 19).
    El filtro se aplica AQUI y no en la interfaz: una estrategia que la familia
    o el equipo tratante descartaron no debe llegar siquiera al parrafo que se
    genera, o reaparece en el texto aunque la pantalla no la liste.
    """
    excluidas = excluidas or set()
    orden = [s.key for s in senales] + [d for d in drivers if d not in {s.key for s in senales}]
    acciones: list[str] = []
    for key in orden:
        entrada = BIBLIOTECA.get(key)
        if entrada is None:
            continue
        if entrada.excluible and entrada.id in excluidas:
            continue
        if entrada.accion not in acciones:
            acciones.append(entrada.accion)
        if len(acciones) >= maximo:
            break
    return acciones


# --- API --------------------------------------------------------------------

def build_narrative(
    logs: pd.DataFrame,
    child_id: str,
    today: dict,
    result,
    nombre: str | None = None,
    audiencia: str = "familia",
    ventana: int = VENTANA_DIAS,
    max_senales: int = 3,
    pregunta_pendiente: str | None = "auto",
    excluidas: set[str] | None = None,
) -> Narrative:
    """Redacta la explicacion del riesgo en el formato de las bases.

    `result` es el RiskResult de core/risk_model.predict_risk: aporta el nivel
    de riesgo, la suficiencia y los drivers que el motor identifico (que se usan
    para desempatar el orden de las senales, de modo que la narrativa siga al
    modelo y no a una lista fija).

    `pregunta_pendiente` es la variable que la interfaz esta pidiendo AHORA, o
    None si ya cerro el registro del dia. Es un parametro explicito y no se
    deduce de `result.suggested_question` porque el motor siempre calcula una
    pregunta candidata, incluso cuando la interfaz decidio no preguntar nada
    mas hoy: leerla de ahi haria que el texto dijera "registre X" justo debajo
    de un "listo por hoy". Con "auto" (por defecto) se usa la del motor.
    """
    nombre = nombre or child_id.replace("_", " ").title()
    dias = _ventana(logs, child_id, today, ventana)
    n_dias = len(dias)

    senales: list[Signal] = []
    protectores: list[Signal] = []
    drivers = list(getattr(result, "drivers", []) or [])

    for theme in THEMES:
        senal, protector = _detectar(dias, theme)
        if senal is not None:
            if senal.key in drivers:      # el motor ya lo marco como driver
                senal.score *= 1.5
            senales.append(senal)
        if protector is not None:
            protectores.append(protector)

    senales.sort(key=lambda s: s.score, reverse=True)
    protectores.sort(key=lambda s: s.score, reverse=True)
    top = senales[:max_senales]

    nivel = _nivel(float(result.risk))
    periodo = "Hoy" if n_dias <= 1 else f"Durante los últimos {n_dias} días"

    # --- Frase 1: la observacion --------------------------------------------
    sin_causa = False
    if top:
        observacion = f"{periodo}, {nombre} ha presentado {_unir([s.frase for s in top])}"
        # Si ninguna de esas senales sigue activa hoy, decirlo: predecimos las
        # proximas 24 horas y un dia limpio despues de dias malos es informacion
        # relevante, no un detalle -- omitirlo haria sonar peor el presente de
        # lo que los datos sostienen.
        if not any(s.presente_hoy for s in top) and any(p.presente_hoy for p in protectores):
            observacion += ", aunque el registro de hoy no muestra esas señales."
        else:
            observacion += "."
    elif protectores:
        observacion = (f"{periodo}, {nombre} ha mantenido "
                       f"{_unir([p.frase for p in protectores[:max_senales]])}.")
        sin_causa = nivel == "elevado"
    else:
        observacion = (f"{periodo} no hay registros suficientes de {nombre} "
                       f"para describir un patrón reciente.")
        sin_causa = nivel == "elevado"

    # --- Frase 2: el veredicto ----------------------------------------------
    # Con suficiencia baja NO se afirma un nivel de riesgo como hecho: las bases
    # piden explicitamente "generar alertas cuando la informacion disponible no
    # sea suficiente para emitir una recomendacion confiable". Se muestra la
    # estimacion, pero marcada como preliminar.
    preliminar = result.sufficiency_level == "baja"
    if preliminar:
        veredicto = (f"Todavía no hay información suficiente para emitir una "
                     f"predicción confiable de las próximas 24 horas. La estimación "
                     f"preliminar es de riesgo {nivel}.")
    elif nivel == "bajo":
        veredicto = ("El riesgo de desregulación para las próximas 24 horas se "
                     "mantiene bajo.")
    else:
        veredicto = (f"Se identifica un riesgo {nivel} de desregulación para las "
                     f"próximas 24 horas.")
    if sin_causa and not preliminar:
        veredicto += (" El riesgo proviene principalmente de su línea base "
                      "histórica, no de un cambio observado en estos días.")

    # --- Frase 3: la sugerencia ---------------------------------------------
    acciones = _acciones(top, drivers, excluidas=excluidas)
    if pregunta_pendiente == "auto":
        pregunta_pendiente = getattr(result, "suggested_question", None)
    etiqueta_pendiente = VARIABLE_LABELS.get(pregunta_pendiente) if pregunta_pendiente else None

    if preliminar and etiqueta_pendiente:
        sugerencia = f"Registre {etiqueta_pendiente} para poder confirmarla."
    elif preliminar and acciones:
        # Ya no se esta pidiendo nada mas hoy: se entrega la accion preventiva
        # como precaucion, con la salvedad de suficiencia mas abajo.
        sugerencia = f"Mientras tanto, como precaución, considere {_unir(acciones[:2])}."
    elif preliminar:
        sugerencia = "Mantenga la rutina habitual y observe durante el día."
    elif nivel == "bajo" and not top:
        sugerencia = ("Mantenga la rutina habitual y los apoyos que están "
                      "funcionando.")
    elif acciones:
        sugerencia = f"Considere {_unir(acciones)}."
    else:
        sugerencia = f"Considere {DEFAULT_RECOMMENDATION}."

    # --- Frase 4: salvedad --------------------------------------------------
    salvedad = None
    if preliminar:
        faltan = len(getattr(result, "missing_relevant", []) or [])
        if result.n_history_days < 7:
            salvedad = (f"{nombre} tiene {result.n_history_days} días de historial: "
                        f"por ahora se le aplica el promedio de la población "
                        f"(arranque en frío) y la estimación se personalizará en "
                        f"los próximos días.")
        elif etiqueta_pendiente:
            salvedad = (f"Faltan {faltan} variables por registrar hoy; con una sola "
                        f"respuesta más el sistema tendrá con qué afinarla.")
        else:
            salvedad = (f"Quedan {faltan} variables sin registrar. No es necesario "
                        f"completarlas, pero hacerlo daría más base a la "
                        f"predicción de mañana.")
    elif result.sufficiency_level == "moderada":
        salvedad = ("La información disponible para esta predicción es moderada: "
                    "se afinará a medida que se completen los registros.")

    partes = [observacion, veredicto, sugerencia] + ([salvedad] if salvedad else [])
    texto = " ".join(partes)

    ficha = None
    if audiencia == "profesional":
        modelo = ("Random Forest" if result.model_used == "random_forest"
                  else "score heurístico ponderado")
        base = (f" · línea base bayesiana {result.base_rate_bayes:.0%}"
                if result.base_rate_bayes is not None else "")
        # La ficha para profesionales lleva los TRES numeros por separado: es
        # la audiencia que puede y debe distinguirlos.
        inc = result.uncertainty or {}
        estab = (f" · estabilidad {1 - inc['dispersion']:.0%} ({inc['nivel']})"
                 if inc.get("disponible") else "")
        ficha = (f"{modelo} · riesgo {result.risk:.0%} · información "
                 f"{result.sufficiency:.0%} ({result.sufficiency_level})"
                 f"{estab}{base} · "
                 f"{result.n_history_days} días de historial · "
                 f"{len(result.missing_relevant)} variables sin registrar hoy.")

    return Narrative(
        texto=texto, observacion=observacion, veredicto=veredicto,
        sugerencia=sugerencia, salvedad=salvedad, nivel=nivel,
        senales=senales, protectores=protectores, ventana=n_dias,
        sin_causa_aparente=sin_causa, preliminar=preliminar, ficha_tecnica=ficha,
    )
