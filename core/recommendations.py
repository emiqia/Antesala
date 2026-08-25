"""
Mapeo deterministico driver -> recomendacion accionable (Seccion 7).
Es una tabla de reglas, no otro modelo: mantiene la recomendacion auditable
y explicable, en linea con el criterio de explicabilidad de las bases.

Las claves coinciden con los nombres de campo reales de la bitacora de Bluba
y con el listado completo de variables de la Seccion 4.1 (ver core/features.py).
"""

RECOMMENDATIONS = {
    "calidad_sueno": "anticipar una rutina de sueno mas temprana y reducir pantallas antes de dormir",
    "n_eventos_desregulacion": "disponer espacios de regulacion accesibles durante el dia",
    "nivel_regulacion_general_dia": "priorizar actividades de baja demanda sensorial hoy",
    "modo_despertar": "dar mas tiempo y menos exigencia en la transicion de la manana",
    "estado_gastrointestinal": "evaluar malestar fisico antes de interpretar la conducta como emocional",
    "adherencia_medicacion": "confirmar con la familia la administracion de medicamentos de hoy",
    "nivel_apoyo_requerido": "reforzar el acompanamiento en las transiciones del dia",
    "cambios_alimentacion": "ofrecer alimentos conocidos y evitar presionar la ingesta",
    "cambios_rutina": "anticipar el cambio con apoyos visuales antes de que ocurra",
    "comportamiento_observado": "reducir demandas y ofrecer pausas reguladoras frecuentes",
    "estado_alerta": "ajustar el nivel de estimulacion: reducir ruido y luz si esta sobreexcitado, "
                      "o proponer actividad propioceptiva si esta letargico",
    "participacion_actividades": "acortar y estructurar las actividades en pasos mas breves",
    "interacciones_sociales": "permitir espacios de retiro y no forzar la interaccion social",
    "alimentacion_recreos": "acompanar el momento de comida en un entorno de menor estimulacion",
}

DEFAULT_RECOMMENDATION = "mantener la rutina habitual y observar durante el dia"


def build_recommendation_text(drivers: list[str]) -> str:
    if not drivers:
        return DEFAULT_RECOMMENDATION
    actions = [RECOMMENDATIONS.get(d) for d in drivers if RECOMMENDATIONS.get(d)]
    if not actions:
        return DEFAULT_RECOMMENDATION
    return "; ".join(actions[:2])


VARIABLE_LABELS = {
    "calidad_sueno": "calidad del sueño",
    "n_eventos_desregulacion": "eventos de desregulación",
    "nivel_regulacion_general_dia": "nivel de regulación general del día",
    "modo_despertar": "modo al despertar",
    "estado_gastrointestinal": "salud gastrointestinal",
    "adherencia_medicacion": "adherencia a la medicación",
    "nivel_apoyo_requerido": "nivel de apoyo requerido",
    "cambios_alimentacion": "cambios en la alimentación",
    "cambios_rutina": "cambios en la rutina",
    "comportamiento_observado": "comportamiento observado",
    "estado_alerta": "estado de alerta",
    "participacion_actividades": "participación en actividades",
    "interacciones_sociales": "interacciones sociales",
    "alimentacion_recreos": "alimentación y recreos",
}
