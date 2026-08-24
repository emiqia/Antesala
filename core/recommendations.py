"""
Mapeo deterministico driver -> recomendacion accionable (Seccion 7).
Es una tabla de reglas, no otro modelo: mantiene la recomendacion auditable
y explicable, en linea con el criterio de explicabilidad de las bases.
"""

RECOMMENDATIONS = {
    "horas_sueno": "anticipar una rutina de sueno mas temprana y reducir pantallas antes de dormir",
    "regulaciones_desregulaciones": "disponer espacios de regulacion accesibles durante el dia",
    "cambios_rutina": "anticipar el cambio con apoyos visuales antes de que ocurra",
    "estado_basal_despertar": "dar mas tiempo y menos exigencia en la transicion de la manana",
    "salud_gastrointestinal": "evaluar malestar fisico antes de interpretar la conducta como emocional",
    "nivel_apoyo_requerido": "reforzar el acompanamiento en el inicio del dia",
    "estado_alerta": "reducir estimulos ambientales (ruido, luz) en las proximas horas",
    "comportamiento_observado": "priorizar actividades de baja demanda sensorial hoy",
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
    "horas_sueno": "sueño",
    "regulaciones_desregulaciones": "eventos de desregulación",
    "cambios_rutina": "cambio de rutina",
    "estado_basal_despertar": "estado al despertar",
    "salud_gastrointestinal": "salud gastrointestinal",
    "nivel_apoyo_requerido": "nivel de apoyo requerido",
    "estado_alerta": "estado de alerta",
    "comportamiento_observado": "comportamiento observado",
}
