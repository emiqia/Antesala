"""
Generador de datos sinteticos para Antesala, alineado al ESQUEMA REAL de Bluba.

A diferencia de la primera version (que usaba nombres de variable inventados
a partir de la lista abstracta de la Seccion 4.1 del documento tecnico), este
generador usa los nombres de campo y las categorias EXACTAS observadas en:
  - docs/Datos base/4_seguimiento_diario_tutor.csv  (bitacora diaria del tutor)
  - docs/Datos base/5_eventos_desregulacion_tutor.csv (registro de episodios)
  - las capturas de pantalla de la app movil de Bluba (docs/5. Presentacion BLUBA.pdf)

Bitacora diaria ("Mi Dia - Estado Basal" en la app):
  - calidad_sueno: Reparador / Interrumpido / Dificultad de Conciliacion
  - modo_despertar: Tranquilo/Alegre / Cansado/Con Sueno / Irritable/Llorando
  - adherencia_medicacion: Si / No / No Aplica
  - estado_gastrointestinal: Normal / Estrenimiento / Diarrea
  - nivel_regulacion_general_dia: Excelente / Estable con Apoyo / Desregulacion Frecuente

Eventos de desregulacion (registro rapido "+", 0 o mas por dia), agregados a
nivel de dia para el modelo:
  - n_eventos_desregulacion, intensidad_max_desregulacion, intensidad_sum_desregulacion
  - tipo_evento_principal: Sobrecarga Sensorial / Transicion de Actividad /
    Desregulacion Emocional / Alimentacion
  - resultado_estrategia_principal: Regulacion Exitosa / Regulacion Parcial /
    Regulacion No Exitosa

crisis_hoy se define operacionalmente como: nivel_regulacion_general_dia ==
"Desregulacion Frecuente" O hubo un evento de intensidad Severa (>=8) ese dia.
crisis_24h (variable objetivo, Seccion 4.2) es crisis_hoy desplazada un dia
hacia atras (el dato de hoy predice si MANANA hay crisis).

Se simulan los tres mecanismos de ausencia de datos de la Seccion 4.4
(MCAR/MAR/MNAR) sobre los campos de la bitacora diaria del tutor.

MECANISMO DE AUSENCIA: ES UN ESCENARIO, NO UN HECHO (revision de agosto 2026).
La version anterior daba por sentado que "un vacio de informacion suele
coincidir con los momentos de mas dificultad para la familia" y horneaba ese
supuesto MNAR en los datos. Es plausible, pero con la evidencia disponible no
esta demostrado como regla general, y un dataset que lo asume produce un
modelo que lo confirma -- razonamiento circular.

Ahora el mecanismo se elige con --missingness y la posicion declarada es la
prudente: la ausencia PUEDE ser informativa, asi que el pipeline conserva
indicadores de ausencia y antiguedad en vez de tratar el vacio como neutro;
pero cual mecanismo opera de verdad lo tendran que decir los datos reales de
Bluba. Los cuatro escenarios existen para medir la sensibilidad del sistema a
ese supuesto:

    mixto  (por defecto)  MCAR + MAR + MNAR juntos, como la version anterior.
    mcar                  ausencia puramente aleatoria: el silencio no informa.
    mar                   depende de cosas observables (fin de semana, colegio),
                          no del resultado.
    mnar                  el caso fuerte: se registra bastante menos los dias
                          dificiles.

Uso previsto: generar los cuatro y comparar el panel de metricas. Si el
sistema solo funciona bajo MNAR, eso hay que saberlo antes de prometer nada.

Uso:
    python generate_synthetic_data.py --out ../data/bitacoras.csv
"""

import argparse
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

CALIDAD_SUENO = ["Reparador", "Interrumpido", "Dificultad de Conciliacion"]
MODO_DESPERTAR = ["Tranquilo/Alegre", "Cansado/Con Sueno", "Irritable/Llorando"]
ADHERENCIA_MEDICACION = ["Si", "No", "No Aplica"]
ESTADO_GASTROINTESTINAL = ["Normal", "Estrenimiento", "Diarrea"]
NIVEL_REGULACION = ["Excelente", "Estable con Apoyo", "Desregulacion Frecuente"]
TIPO_EVENTO = ["Sobrecarga Sensorial", "Transicion de Actividad", "Desregulacion Emocional", "Alimentacion"]
RESULTADO_ESTRATEGIA = ["Regulacion Exitosa", "Regulacion Parcial", "Regulacion No Exitosa"]
SOURCES = ["familia", "escuela", "terapeuta"]

# --- Variables de la Seccion 4.1 que no estaban en la muestra de datos reales
# (esa muestra solo cubria la pantalla "Mi Dia" del tutor). Se modelan segun la
# descripcion de las bases del desafio. ESTADO_ALERTA reutiliza las categorias
# EXACTAS observadas en docs/Datos base/3_sesiones_profesionales.csv.
NIVEL_APOYO = ["Bajo", "Medio", "Alto"]
CAMBIOS_ALIMENTACION = ["Sin cambios", "Menor apetito", "Selectividad aumentada"]
CAMBIOS_RUTINA = ["No", "Si"]
COMPORTAMIENTO_OBSERVADO = ["Estable", "Inquieto", "Desregulado"]
ESTADO_ALERTA = ["Optimo (Regulado)", "Bajo (Letargico)", "Alto (Sobreexcitado)"]
PARTICIPACION_ACTIVIDADES = ["Completa", "Parcial", "No participa"]
INTERACCIONES_SOCIALES = ["Normal", "Baja", "Evitativa"]
ALIMENTACION_RECREOS = ["Normal", "Reducida", "Rechaza"]

# --- Sesiones profesionales (esporadicas, no diarias) -----------------------
# Las bases oficiales describen la informacion como procedente de "la
# bitacora diaria Y observaciones terapeuticas/escolares" -- hasta ahora solo
# se modelaba la bitacora del tutor. Categorias EXACTAS observadas en
# docs/Datos base/3_sesiones_profesionales.csv.
PROFESIONES_SESION = ["Terapeuta Ocupacional", "Fonoaudiologo", "Psicopedagogo"]
NIVEL_ALERTA_SESION = ["Optimo (Regulado)", "Calmado", "Alto (Sobreexcitado)", "Bajo (Letargico)"]
P_SESION_HOY = 0.18  # aprox. 1 sesion cada 5-6 dias por nino

# Perfiles diagnosticos reales observados en docs/Datos base/1_casos_anonimizados.csv.
DIAGNOSTICOS = ["Trastorno del Espectro Autista (TEA)"]
PERFILES_SENSORIALES = [
    "Buscador Sensorial Vestibular y Propioceptivo",
    "Hipersensible Auditivo y Evitador Tactil",
    "Hiporreactivo Propioceptivo y Visual",
    "Hipersensible Gustativo y Olfativo",
]

# Rasgos por perfil sensorial: a diferencia de la primera version (donde el
# perfil era solo una etiqueta cosmetica), estos multiplicadores hacen que
# CADA perfil tenga un patron de riesgo genuinamente distinto -- un
# "Buscador Vestibular" es mas sensible a cambios de rutina, uno
# "Hipersensible Gustativo" es mas sensible a la alimentacion, etc. Esto
# refuerza directamente la premisa del pooling jerarquico (Seccion 3): cada
# nino es distinto, no solo en "cuanto" sino en "que" lo afecta, y hace que
# el panel de variables mas predictivas cambie de verdad segun el nino.
PROFILE_TRAITS = {
    "Buscador Sensorial Vestibular y Propioceptivo": {
        "transicion_mult": 1.7,   # los cambios de rutina le pegan mucho mas fuerte
        "food_mult": 0.6,
        "tipo_evento_favorito": "Transicion de Actividad",
        "alerta_favorita": None,  # sin sesgo particular de hiper/hipoalerta
    },
    "Hipersensible Auditivo y Evitador Tactil": {
        "transicion_mult": 1.0,
        "food_mult": 0.7,
        "tipo_evento_favorito": "Sobrecarga Sensorial",
        "alerta_favorita": "Alto (Sobreexcitado)",
    },
    "Hiporreactivo Propioceptivo y Visual": {
        "transicion_mult": 0.7,
        "food_mult": 0.8,
        "tipo_evento_favorito": "Desregulacion Emocional",
        "alerta_favorita": "Bajo (Letargico)",
    },
    "Hipersensible Gustativo y Olfativo": {
        "transicion_mult": 0.9,
        "food_mult": 2.0,         # alimentacion y recreos es MUCHO mas predictivo
        "tipo_evento_favorito": "Alimentacion",
        "alerta_favorita": None,
    },
}


def make_children(n_children: int, id_prefix: str = "nino", start_index: int = 0) -> pd.DataFrame:
    """Perfil base (oculto) de cada nino: su probabilidad real de crisis, su
    tendencia de sueno y su sensibilidad a transiciones. El modelo no ve estos
    valores; son la 'verdad' que usa el generador para simular datos realistas."""
    rows = []
    for i in range(n_children):
        child_id = f"{id_prefix}_{start_index + i + 1:03d}"
        base_crisis_rate = RNG.beta(2, 8)
        sleep_quality_bias = RNG.normal(0, 1)   # + = duerme mejor que el promedio
        sensitivity_transicion = RNG.uniform(0.5, 2.5)
        on_medication = RNG.random() < 0.35
        perfil = PERFILES_SENSORIALES[i % len(PERFILES_SENSORIALES)]
        traits = PROFILE_TRAITS[perfil]
        rows.append({
            "child_id": child_id,
            "base_crisis_rate": base_crisis_rate,
            "sleep_quality_bias": sleep_quality_bias,
            # el multiplicador del perfil se aplica UNA vez aqui, sobre la
            # sensibilidad propia de cada nino (sigue habiendo variacion
            # individual dentro de un mismo perfil).
            "sensitivity_transicion": sensitivity_transicion * traits["transicion_mult"],
            "food_sensitivity": RNG.uniform(0.7, 1.3) * traits["food_mult"],
            "tipo_evento_favorito": traits["tipo_evento_favorito"],
            "alerta_favorita": traits["alerta_favorita"],
            "on_medication": on_medication,
            "diagnostico_principal": DIAGNOSTICOS[0],
            "perfil_sensorial_predominante": perfil,
        })
    return pd.DataFrame(rows)


def _sample_calidad_sueno(quality_score: float) -> str:
    """quality_score alto -> mas probable Reparador; bajo -> mas probable
    Dificultad de Conciliacion."""
    if quality_score > 0.5:
        p = [0.75, 0.20, 0.05]
    elif quality_score > -0.5:
        p = [0.40, 0.40, 0.20]
    else:
        p = [0.15, 0.35, 0.50]
    return RNG.choice(CALIDAD_SUENO, p=p)


def _sample_modo_despertar(calidad_sueno: str) -> str:
    if calidad_sueno == "Reparador":
        p = [0.70, 0.20, 0.10]
    elif calidad_sueno == "Interrumpido":
        p = [0.30, 0.45, 0.25]
    else:
        p = [0.10, 0.35, 0.55]
    return RNG.choice(MODO_DESPERTAR, p=p)


def _sample_ordinal_by_severity(options: list[str], severity: float, sharpness: float = 3.0) -> str:
    """Muestrea de una lista ORDINAL (options[0] = mejor, options[-1] = peor)
    de forma que a mayor `severity` (0..1) suba la probabilidad de los valores
    peores. Se usa para las variables de la Seccion 4.1 que no venian en la
    muestra de datos reales, manteniendolas coherentes con el resto del dia."""
    n = len(options)
    # posicion objetivo dentro de la escala ordinal, segun severidad
    target = severity * (n - 1)
    weights = np.array([np.exp(-sharpness * abs(i - target)) for i in range(n)], dtype=float)
    weights /= weights.sum()
    return RNG.choice(options, p=weights)


# Escenario de ausencia activo. Se fija desde main() y por defecto reproduce
# EXACTAMENTE el comportamiento anterior, para que el dataset del repositorio no
# cambie al introducir esta opcion.
MECANISMO = "mixto"


def _probabilidades(row: dict, is_weekend: bool) -> tuple[float, float, float]:
    """Probabilidades de ausencia segun el escenario activo.

    Devuelve (p_mcar, p_mnar, p_weekend). Cada escenario apaga las componentes
    que no le corresponden, manteniendo la ausencia TOTAL en un rango parecido
    para que la comparacion entre escenarios no confunda "mecanismo distinto"
    con "mas datos faltantes".
    """
    if MECANISMO == "mcar":
        # Todo el peso en la componente aleatoria; sin dependencia del dia ni
        # del resultado. El indicador de ausencia no deberia aportar nada.
        return 0.19, 0.0, 0.19
    if MECANISMO == "mar":
        # Depende solo de variables OBSERVADAS (fin de semana / origen escolar).
        return 0.06, 0.0, 0.45
    if MECANISMO == "mnar":
        # Caso fuerte: la brecha entre dias con y sin episodio es grande.
        return 0.06, 0.32 if row["crisis_24h"] else 0.04, 0.35
    # mixto (por defecto): los tres a la vez, como hasta ahora.
    return 0.06, 0.20 if row["crisis_24h"] else 0.10, 0.35 if is_weekend else 0.05


def simulate_missingness(row: dict, day_of_week: int) -> dict:
    """Aplica los mecanismos de ausencia de datos de la Seccion 4.4 sobre los
    campos de la bitacora diaria del tutor, segun el escenario MECANISMO."""
    is_weekend = day_of_week >= 5
    p_missing_weekend = 0.35 if is_weekend else 0.05          # MAR
    # MNAR: la familia registra menos en los dias mas dificiles (el silencio es
    # senal, Seccion 4.4). La brecha se mantiene DELIBERADAMENTE moderada: con
    # una brecha muy grande el indicador de ausencia se vuelve la senal dominante
    # y el modelo aprende "registro completo => manana esta bien", ahogando el
    # contenido clinico real de las variables. Aqui el silencio informa, pero no
    # decide.
    p_missing_mcar, p_missing_mnar, p_weekend_base = _probabilidades(row, is_weekend)
    if MECANISMO != "mixto":
        p_missing_weekend = p_weekend_base if is_weekend else min(p_weekend_base, 0.05)

    out = dict(row)
    never_null = {"child_id", "date", "crisis_24h", "crisis_hoy", "fuente_registro"}

    # --- Dia sin registro: nadie abrio la app ---------------------------------
    # Sortear la ausencia campo por campo hace que un dia COMPLETAMENTE en
    # blanco tenga probabilidad practicamente nula (con 14 sorteos
    # independientes al 10-30%, no ocurre nunca en 3.700 filas). Pero en la
    # practica ese es el caso incompleto MAS comun: la familia simplemente no
    # registro nada ese dia. Es un evento del dia entero, no 14 ausencias
    # independientes.
    #
    # Sin este mecanismo el dataset no contiene ningun dia totalmente vacio, y
    # el modelo termina EXTRAPOLANDO justo donde la interfaz mas lo consulta:
    # el estado inicial de cada dia, antes de que nadie haya registrado nada.
    # Extrapolando la pendiente MNAR, un dia en blanco daba ~0.87 de riesgo --
    # un numero inventado sobre una region del espacio de entrada que el
    # modelo nunca vio.
    #
    # La brecha MNAR se mantiene moderada por la misma razon que mas abajo: el
    # silencio informa, pero no debe decidir.
    if MECANISMO == "mcar":
        p_dia_sin_registro = 0.08          # no depende de nada
    elif MECANISMO == "mar":
        p_dia_sin_registro = 0.14 if is_weekend else 0.05
    elif MECANISMO == "mnar":
        p_dia_sin_registro = 0.16 if row["crisis_24h"] else 0.04
    else:
        p_dia_sin_registro = 0.10 if row["crisis_24h"] else 0.06
    if RNG.random() < p_dia_sin_registro:
        for field in list(out.keys()):
            if not field.startswith("_") and field not in never_null:
                out[field] = None
        return out

    tutor_fields = ["calidad_sueno", "modo_despertar", "adherencia_medicacion",
                     "estado_gastrointestinal", "nivel_regulacion_general_dia"]
    # Campos que dependen del colegio/terapia: los fines de semana casi no se
    # registran (MAR fuerte, Seccion 4.4).
    school_fields = ["participacion_actividades", "interacciones_sociales",
                      "alimentacion_recreos"]
    # El registro de un episodio de desregulacion es UNA sola accion del
    # tutor en la app (Seccion 6.3: se pregunta como campo compuesto). Si esa
    # accion no se hizo hoy, los CINCO campos quedan sin registrar JUNTOS --
    # no cada uno con su propia probabilidad independiente. Sin este bloque,
    # n_eventos_desregulacion nunca es nulo en el dataset de entrenamiento y
    # el Random Forest no puede aprender un indicador de ausencia para el, lo
    # que en la practica le hace tratar "no se registro" exactamente igual
    # que "se registro un 0" -- imputacion silenciosa de facto, violando la
    # Seccion 4.4.
    event_fields = ["n_eventos_desregulacion", "intensidad_max_desregulacion",
                     "intensidad_sum_desregulacion", "tipo_evento_principal",
                     "resultado_estrategia_principal"]
    p_missing_event = min(p_missing_mcar + p_missing_mnar, 0.9)
    if RNG.random() < p_missing_event:
        for f in event_fields:
            out[f] = None

    for field in list(out.keys()):
        if field.startswith("_") or field in never_null or field in event_fields:
            continue
        p_missing = p_missing_mcar + p_missing_mnar
        if field in tutor_fields:
            p_missing = max(p_missing, p_missing_weekend)
        if field in school_fields:
            p_missing = max(p_missing, 0.80 if is_weekend else 0.10)
        if RNG.random() < min(p_missing, 0.9):
            out[field] = None
    return out


_SUENO_BAD = {"Reparador": 0.0, "Interrumpido": 0.5, "Dificultad de Conciliacion": 1.0}
_REG_BAD = {"Excelente": 0.0, "Estable con Apoyo": 0.4, "Desregulacion Frecuente": 1.0}


def generate_logs(children: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """Genera las bitacoras dia a dia, en un solo paso secuencial por nino.

    Punto critico de diseno (asi se evito un error de la primera version): la
    severidad que se traspasa de un dia al siguiente NO es solo un promedio
    suavizado oculto -- se construye mezclando esa tendencia suave con la
    "maldad observada" de HOY (las mismas variables que ve la familia: sueno,
    salud gastrointestinal, adherencia a medicacion, nivel de regulacion,
    intensidad de eventos). Esto es lo que hace que un dia con TODAS las
    variables en su peor valor efectivamente eleve la probabilidad de crisis
    de manana -- si la severidad de manana solo heredara un promedio suave
    (sin atarse a lo observado), un pico aislado en un nino de linea base baja
    revertia hacia esa linea base y el efecto se perdia (regresion a la media).

    crisis_24h seguye la Seccion 4.2 al pie de la letra: crisis_24h del dia d
    es el valor de "crisis_hoy" registrado el dia d+1.

    `children` puede traer columnas opcionales "history_days" y "start_offset"
    (dias de historial propio y desplazamiento del primer dia respecto a
    2026-07-01) para generar ninos con MENOS dias de historial que terminan en
    la MISMA fecha final que el resto -- asi se puede demostrar el escenario
    de "nino nuevo" (arranque en frio, Seccion 3.6) con una fecha "de hoy"
    consistente para toda la plataforma. Si faltan, se usa `n_days` y offset 0
    para todos (comportamiento original).
    """
    PERSISTENCE = 0.78   # cuanto de la severidad de ayer se traspasa a la severidad de hoy
    OBS_BLEND = 0.70     # cuanto pesa lo OBSERVADO hoy (vs. la severidad suave) al alimentar manana

    logs = []
    for _, child in children.iterrows():
        cid = child["child_id"]
        base_rate = child["base_crisis_rate"]
        sleep_bias = child["sleep_quality_bias"]
        sens_transicion = child["sensitivity_transicion"]
        on_medication = child["on_medication"]
        food_sensitivity = child["food_sensitivity"]
        tipo_evento_favorito = child["tipo_evento_favorito"]
        alerta_favorita = child["alerta_favorita"]
        local_n_days = int(child["history_days"]) if "history_days" in child and pd.notna(child["history_days"]) else n_days
        start_offset = int(child["start_offset"]) if "start_offset" in child and pd.notna(child["start_offset"]) else 0

        day_rows = []
        carry = base_rate  # severidad/observado que se traspasa dia a dia

        for day in range(local_n_days + 1):
            transicion = RNG.random() < 0.15
            # Un "carry" alto (dia(s) previos malos) tambien dificulta el sueno de hoy.
            sleep_score = RNG.normal(sleep_bias - (1.2 if transicion else 0) - 1.3 * carry, 0.8)
            sleep_deficit = max(0.0, -sleep_score)
            fresh = min(0.95, base_rate + 0.11 * sleep_deficit + (0.16 * sens_transicion if transicion else 0))
            severity_today = float(np.clip(PERSISTENCE * carry + (1 - PERSISTENCE) * fresh, 0.02, 0.97))

            calidad_sueno = _sample_calidad_sueno(sleep_score)
            modo_despertar = _sample_modo_despertar(calidad_sueno)

            if on_medication:
                adherencia_medicacion = "No" if RNG.random() < (0.03 + 0.85 * severity_today) else "Si"
            else:
                adherencia_medicacion = "No Aplica"

            gi_p = 0.03 + 0.85 * severity_today
            estado_gastrointestinal = (
                "Normal" if RNG.random() > gi_p
                else RNG.choice(["Estrenimiento", "Diarrea"], p=[0.6, 0.4])
            )

            # --- Eventos de desregulacion del dia (0 o mas) ---
            lam = 0.1 + 3.5 * severity_today + (0.4 * sens_transicion if transicion else 0)
            n_eventos = int(RNG.poisson(lam))

            intensidad_max = np.nan
            intensidad_sum = 0.0
            tipo_evento_principal = None
            resultado_estrategia_principal = None
            if n_eventos > 0:
                base_intensidad = np.clip(1 + 8 * severity_today + RNG.normal(0, 1), 0, 10)
                intensidades = np.clip(RNG.normal(base_intensidad, 1.3, size=n_eventos), 0, 10)
                intensidad_max = float(intensidades.max())
                intensidad_sum = float(intensidades.sum())
                # El tipo de evento no es uniforme entre perfiles sensoriales: el
                # "favorito" de este nino (Seccion 4.1, PROFILE_TRAITS) tiene el
                # doble de probabilidad relativa de ser el tipo del episodio.
                remaining = [t for t in TIPO_EVENTO if t != "Transicion de Actividad"]
                if tipo_evento_favorito in remaining:
                    base_p = np.array([2.0 if t == tipo_evento_favorito else 1.0 for t in remaining])
                    p_tipo = base_p / base_p.sum()
                else:
                    p_tipo = [0.45, 0.35, 0.20]
                tipo_evento_principal = "Transicion de Actividad" if transicion else RNG.choice(
                    remaining, p=p_tipo)
                resultado_estrategia_principal = RNG.choice(
                    RESULTADO_ESTRATEGIA,
                    p=[0.55, 0.30, 0.15] if intensidad_max < 8 else [0.30, 0.40, 0.30])

            event_severity = (intensidad_max if not np.isnan(intensidad_max) else 0) / 10.0
            reg_p = np.clip(
                0.05 + 0.55 * severity_today + 0.35 * event_severity
                + (0.12 if calidad_sueno == "Dificultad de Conciliacion" else 0), 0, 0.95)
            if RNG.random() < reg_p:
                nivel_regulacion_general_dia = "Desregulacion Frecuente"
            elif event_severity > 0.15 or calidad_sueno != "Reparador":
                nivel_regulacion_general_dia = RNG.choice(["Estable con Apoyo", "Excelente"], p=[0.7, 0.3])
            else:
                nivel_regulacion_general_dia = RNG.choice(["Excelente", "Estable con Apoyo"], p=[0.75, 0.25])

            crisis_hoy = (nivel_regulacion_general_dia == "Desregulacion Frecuente") or (
                not np.isnan(intensidad_max) and intensidad_max >= 8)

            # --- Variables de la Seccion 4.1 que completan el listado de Bluba ---
            # Todas se muestrean desde la severidad del dia (mas la senal especifica
            # que corresponda), de modo que sean coherentes con el resto del registro.
            day_severity = float(np.clip(0.6 * severity_today + 0.4 * event_severity, 0, 1))
            nivel_apoyo_requerido = _sample_ordinal_by_severity(NIVEL_APOYO, day_severity)
            comportamiento_observado = _sample_ordinal_by_severity(COMPORTAMIENTO_OBSERVADO, day_severity)
            participacion_actividades = _sample_ordinal_by_severity(PARTICIPACION_ACTIVIDADES, day_severity)
            interacciones_sociales = _sample_ordinal_by_severity(INTERACCIONES_SOCIALES, day_severity)
            # Sensibilidad alimentaria del perfil (Seccion 4.1, PROFILE_TRAITS):
            # para un perfil "Hipersensible Gustativo" la comida es un canal
            # mucho mas predictivo que para el resto; para otros perfiles pesa
            # menos que la severidad general del dia.
            food_severity = float(np.clip(day_severity * food_sensitivity, 0, 1))
            alimentacion_recreos = _sample_ordinal_by_severity(ALIMENTACION_RECREOS, food_severity)
            cambios_alimentacion = _sample_ordinal_by_severity(
                CAMBIOS_ALIMENTACION,
                float(np.clip(food_severity + (0.25 if estado_gastrointestinal != "Normal" else 0), 0, 1)))
            # cambios_rutina es un DISPARADOR (causa), no una consecuencia: se toma
            # de la transicion del dia, no de la severidad resultante.
            cambios_rutina = "Si" if transicion else "No"
            # estado_alerta no es monotono: la desregulacion puede ir a hiper o
            # hipo. El perfil sensorial del nino (si tiene una direccion
            # favorita, PROFILE_TRAITS) determina hacia donde se desregula.
            if RNG.random() < 0.55 * day_severity:
                if alerta_favorita:
                    estado_alerta = alerta_favorita
                else:
                    estado_alerta = RNG.choice(["Alto (Sobreexcitado)", "Bajo (Letargico)"], p=[0.7, 0.3])
            else:
                estado_alerta = "Optimo (Regulado)"

            # --- lo que efectivamente se observo hoy (0..1), retroalimenta manana ---
            observed_badness = (
                0.22 * _SUENO_BAD[calidad_sueno]
                + 0.13 * (0.0 if estado_gastrointestinal == "Normal" else 1.0)
                + 0.13 * (1.0 if adherencia_medicacion == "No" else 0.0)
                + 0.22 * _REG_BAD[nivel_regulacion_general_dia]
                + 0.18 * event_severity
                + 0.06 * (COMPORTAMIENTO_OBSERVADO.index(comportamiento_observado) / 2.0)
                + 0.06 * (0.0 if estado_alerta == "Optimo (Regulado)" else 1.0)
            )
            carry = float(np.clip(OBS_BLEND * observed_badness + (1 - OBS_BLEND) * severity_today, 0.02, 0.97))

            # --- Sesion profesional del dia (esporadica) ---------------------
            # Es una observacion INDEPENDIENTE de la severidad del dia (para no
            # acoplar mas el sistema AR ya calibrado) -- el profesional reporta
            # lo que ve, pero no retroalimenta la severidad de manana como si
            # fuera otra accion del tutor.
            hubo_sesion = RNG.random() < P_SESION_HOY
            if hubo_sesion:
                if RNG.random() < 0.5 * day_severity:
                    nivel_alerta_sesion = RNG.choice(["Alto (Sobreexcitado)", "Bajo (Letargico)"], p=[0.6, 0.4])
                else:
                    nivel_alerta_sesion = RNG.choice(["Optimo (Regulado)", "Calmado"], p=[0.6, 0.4])
                profesion_sesion = RNG.choice(PROFESIONES_SESION, p=[0.5, 0.3, 0.2])
            else:
                nivel_alerta_sesion = None
                profesion_sesion = None

            day_rows.append({
                "calidad_sueno": calidad_sueno,
                "modo_despertar": modo_despertar,
                "adherencia_medicacion": adherencia_medicacion,
                "estado_gastrointestinal": estado_gastrointestinal,
                "nivel_regulacion_general_dia": nivel_regulacion_general_dia,
                "n_eventos_desregulacion": n_eventos,
                "intensidad_max_desregulacion": None if np.isnan(intensidad_max) else round(intensidad_max, 1),
                "intensidad_sum_desregulacion": round(intensidad_sum, 1),
                "tipo_evento_principal": tipo_evento_principal,
                "resultado_estrategia_principal": resultado_estrategia_principal,
                "nivel_apoyo_requerido": nivel_apoyo_requerido,
                "cambios_alimentacion": cambios_alimentacion,
                "cambios_rutina": cambios_rutina,
                "comportamiento_observado": comportamiento_observado,
                "estado_alerta": estado_alerta,
                "participacion_actividades": participacion_actividades,
                "interacciones_sociales": interacciones_sociales,
                "alimentacion_recreos": alimentacion_recreos,
                "nivel_alerta_sesion": nivel_alerta_sesion,
                "profesion_sesion": profesion_sesion,
                "fuente_registro": RNG.choice(SOURCES, p=[0.6, 0.25, 0.15]),
                "crisis_hoy": bool(crisis_hoy),
            })

        # crisis_24h del dia d = crisis_hoy registrado el dia d+1 (Seccion 4.2, al pie de la letra).
        for day in range(local_n_days):
            date = pd.Timestamp("2026-07-01") + pd.Timedelta(days=start_offset + day)
            row = dict(day_rows[day])
            row["child_id"] = cid
            row["date"] = date
            row["crisis_24h"] = day_rows[day + 1]["crisis_hoy"]
            row = simulate_missingness(row, date.dayofweek)
            logs.append(row)

    return pd.DataFrame(logs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_children", type=int, default=25)
    parser.add_argument("--n_days", type=int, default=150)
    parser.add_argument("--n_cold_start", type=int, default=3,
                         help="ninos adicionales con historial corto (2-5 dias), para "
                              "demostrar el escenario de arranque en frio (Seccion 3.6)")
    parser.add_argument("--out", type=str, default="bitacoras.csv")
    parser.add_argument("--missingness", type=str, default="mixto",
                         choices=["mixto", "mcar", "mar", "mnar"],
                         help="mecanismo de ausencia de datos a simular (Seccion 4.4). "
                              "'mixto' reproduce el dataset del repositorio; los otros "
                              "tres sirven para medir la sensibilidad del sistema a ese "
                              "supuesto, que con datos reales todavia no se conoce.")
    args = parser.parse_args()

    global MECANISMO
    MECANISMO = args.missingness

    children = make_children(args.n_children)
    children["history_days"] = args.n_days
    children["start_offset"] = 0

    if args.n_cold_start > 0:
        # Ninos "recien llegados a la plataforma": 2 a 5 dias de historial,
        # terminando en la MISMA fecha final que el resto (start_offset los
        # corre hacia el final del rango de fechas) -- asi se puede mostrar
        # el ejemplo Nino A / Nino B del documento (Seccion 3.4) con una
        # fecha "de hoy" consistente para toda la plataforma.
        cold = make_children(args.n_cold_start, id_prefix="nino_nuevo")
        hist_days = RNG.integers(2, 6, size=len(cold))  # 2..5 dias inclusive
        cold["history_days"] = hist_days
        cold["start_offset"] = args.n_days - hist_days
        children = pd.concat([children, cold], ignore_index=True)

    logs = generate_logs(children, args.n_days)

    children.to_csv(args.out.replace(".csv", "_ninos.csv"), index=False)
    logs.to_csv(args.out, index=False)
    print(f"Generados {len(logs)} registros para {len(children)} ninos "
          f"({args.n_children} con historial completo + {args.n_cold_start} de arranque "
          f"en frio) -> {args.out}")
    print(f"Mecanismo de ausencia simulado: {MECANISMO}")


if __name__ == "__main__":
    main()
