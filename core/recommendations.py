"""
Biblioteca de recomendaciones -- Seccion 12 del documento tecnico.

    "Las recomendaciones se generaran mediante una biblioteca explicita y
     auditable. No se utilizara un LLM para inventar estrategias
     automaticamente durante el MVP. Cada recomendacion tendra: condicion de
     activacion; accion sugerida; contexto; fuente o revision responsable;
     posibilidad de exclusion si no es adecuada para ese nino."

Eso es exactamente lo que hay aqui: una tabla de reglas, no un generador de
texto. Cada entrada declara los cinco campos, de modo que cualquiera pueda
tomar una sugerencia de la pantalla y rastrear por que aparecio, con que dato
se activo y quien responde por ella.

SOBRE EL CAMPO `estado_revision`
Ninguna de estas acciones ha sido revisada todavia por el equipo clinico de
Bluba: las redacto el equipo de desarrollo a partir de la descripcion del
desafio. Marcarlas todas como "pendiente" no es un descuido, es el punto: una
biblioteca auditable tiene que poder decir que NO esta validado. Inventar
citas a literatura clinica para llenar el campo seria fabricar evidencia, que
es peor que declarar el vacio.

Cuando el equipo clinico revise una entrada, cambia su `estado_revision` a
"revisada" y firma en `revisada_por`. El sistema no necesita ningun otro
cambio.

EXCLUSION POR NINO (Seccion 19)
Una estrategia puede ser inadecuada para un nino concreto: puede que ya se
haya probado y no funcione, que choque con una indicacion del equipo tratante
o que la familia simplemente no la quiera. Las entradas `excluible=True` se
pueden apagar por nino sin tocar la biblioteca ni afectar a los demas. Es
parte de tratar el sistema como apoyo a la decision y no como una fuente de
ordenes.

Dos entradas son `excluible=False` a proposito: verificar la administracion de
un medicamento y derivar al equipo tratante no son preferencias de estilo, y
apagarlas convertiria un canal de seguridad en una opcion de configuracion.

NOTA DE ESTILO: los comentarios y docstrings van sin tildes, como en todo el
repositorio; los textos que ve una persona en pantalla (condicion, accion,
contexto, fuente) van acentuados.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Recomendacion:
    """Una entrada de la biblioteca, con todo lo que la Seccion 12 exige."""
    id: str
    driver: str            # variable de la bitacora que la activa
    condicion: str         # cuando se activa, en lenguaje verificable
    accion: str            # que se sugiere hacer
    contexto: str          # cuando aplica y cuando NO
    fuente: str            # de donde sale la sugerencia
    estado_revision: str = "pendiente"   # "pendiente" | "revisada"
    revisada_por: str | None = None
    excluible: bool = True


_FUENTE_EQUIPO = ("Redactada por el equipo de desarrollo a partir de la descripción "
                  "del desafío Bluba. Sin revisión clínica.")

BIBLIOTECA: dict[str, Recomendacion] = {
    r.driver: r for r in [
        Recomendacion(
            id="REC-01", driver="calidad_sueno",
            condicion="La calidad del sueño registrada hoy está por debajo de la línea "
                      "base del niño.",
            accion="anticipar una rutina de sueño más temprana y reducir pantallas "
                   "antes de dormir",
            contexto="Actúa sobre la noche que viene, no sobre el día de hoy. No aplicar "
                     "si el equipo tratante indicó otro horario de sueño.",
            fuente=_FUENTE_EQUIPO),
        Recomendacion(
            id="REC-02", driver="n_eventos_desregulacion",
            condicion="Se registraron episodios de desregulación por encima de lo "
                      "habitual en este niño.",
            accion="disponer espacios de regulación accesibles durante el día",
            contexto="Requiere que el espacio ya sea conocido y aceptado por el niño; "
                     "improvisar un lugar nuevo en un día difícil suele empeorarlo.",
            fuente=_FUENTE_EQUIPO),
        Recomendacion(
            id="REC-03", driver="nivel_regulacion_general_dia",
            condicion="El nivel de regulación general del día es peor que su patrón "
                      "habitual.",
            accion="priorizar actividades de baja demanda sensorial hoy",
            contexto="Reducir la demanda no es suspender toda actividad: quitarle la "
                     "estructura al día puede aumentar la desregulación.",
            fuente=_FUENTE_EQUIPO),
        Recomendacion(
            id="REC-04", driver="modo_despertar",
            condicion="El modo al despertar fue irritable o más cansado que lo habitual.",
            accion="dar más tiempo y menos exigencia en la transición de la mañana",
            contexto="Útil cuando la mañana tiene horarios rígidos (colegio, terapia). "
                     "Requiere margen real de tiempo, no solo intención.",
            fuente=_FUENTE_EQUIPO),
        Recomendacion(
            id="REC-05", driver="estado_gastrointestinal",
            condicion="Se registró malestar gastrointestinal.",
            accion="evaluar malestar físico antes de interpretar la conducta como emocional",
            contexto="Es una advertencia de interpretación, no una intervención. Si el "
                     "malestar persiste corresponde consulta médica, no manejo conductual.",
            fuente=_FUENTE_EQUIPO),
        Recomendacion(
            id="REC-06", driver="adherencia_medicacion",
            condicion="La adherencia a la medicación se registró como no cumplida.",
            accion="confirmar con la familia la administración de medicamentos de hoy",
            contexto="SOLO verificar y comunicar. El sistema no sugiere iniciar, suspender "
                     "ni ajustar ninguna medicación: eso es decisión médica.",
            fuente=_FUENTE_EQUIPO, excluible=False),
        Recomendacion(
            id="REC-07", driver="nivel_apoyo_requerido",
            condicion="El nivel de apoyo requerido hoy es mayor que el habitual.",
            accion="reforzar el acompañamiento en las transiciones del día",
            contexto="Acompañar no es sustituir: conviene mantener las tareas que el niño "
                     "ya hace solo y reforzar los cambios de actividad.",
            fuente=_FUENTE_EQUIPO),
        Recomendacion(
            id="REC-08", driver="cambios_alimentacion",
            condicion="Se registraron cambios en la alimentación (menor apetito o "
                      "selectividad aumentada).",
            accion="ofrecer alimentos conocidos y evitar presionar la ingesta",
            contexto="No aplica si hay un plan de alimentación indicado por un "
                     "profesional; en ese caso, consultar antes de modificar.",
            fuente=_FUENTE_EQUIPO),
        Recomendacion(
            id="REC-09", driver="cambios_rutina",
            condicion="Se registró un cambio de rutina.",
            accion="anticipar el cambio con apoyos visuales antes de que ocurra",
            contexto="El valor está en anticipar. Si el cambio ya ocurrió, ayuda más "
                     "explicar lo que viene después que justificar lo que pasó.",
            fuente=_FUENTE_EQUIPO),
        Recomendacion(
            id="REC-10", driver="comportamiento_observado",
            condicion="El comportamiento observado es inquieto o desregulado.",
            accion="reducir demandas y ofrecer pausas reguladoras frecuentes",
            contexto="Las pausas funcionan si son previsibles; interrumpir al azar puede "
                     "sumar una transición más.",
            fuente=_FUENTE_EQUIPO),
        Recomendacion(
            id="REC-11", driver="estado_alerta",
            condicion="El estado de alerta está fuera del rango óptimo (sobreexcitado o "
                      "letárgico).",
            accion="ajustar el nivel de estimulación: reducir ruido y luz si está "
                   "sobreexcitado, o proponer actividad propioceptiva si está letárgico",
            contexto="La dirección del ajuste depende del sentido de la desviación: es la "
                     "única regla de la biblioteca que no es monótona.",
            fuente=_FUENTE_EQUIPO),
        Recomendacion(
            id="REC-12", driver="participacion_actividades",
            condicion="La participación en actividades fue parcial o nula.",
            accion="acortar y estructurar las actividades en pasos más breves",
            contexto="Dato de origen escolar: conviene coordinarlo con el colegio y no "
                     "aplicarlo solo en casa.",
            fuente=_FUENTE_EQUIPO),
        Recomendacion(
            id="REC-13", driver="interacciones_sociales",
            condicion="Las interacciones sociales fueron evitativas o menores que lo "
                      "habitual.",
            accion="permitir espacios de retiro y no forzar la interacción social",
            contexto="Retirarse puede ser una estrategia de regulación eficaz. No debe "
                     "leerse por defecto como un problema que haya que corregir.",
            fuente=_FUENTE_EQUIPO),
        Recomendacion(
            id="REC-14", driver="alimentacion_recreos",
            condicion="La alimentación en recreos fue reducida o rechazada.",
            accion="acompañar el momento de comida en un entorno de menor estimulación",
            contexto="Dato de origen escolar. Requiere que el colegio pueda ofrecer un "
                     "espacio alternativo.",
            fuente=_FUENTE_EQUIPO),
        Recomendacion(
            id="REC-15", driver="nivel_alerta_sesion",
            condicion="El equipo profesional registró una alerta en la última sesión.",
            accion="coordinar con el equipo tratante los apoyos acordados en la última sesión",
            contexto="Deriva a la decisión profesional en vez de sustituirla.",
            fuente=_FUENTE_EQUIPO, excluible=False),
    ]
}

DEFAULT_RECOMMENDATION = "mantener la rutina habitual y observar durante el día"

# Compatibilidad: el resto del codigo (core/narrative.py) consume el mapa plano
# driver -> accion. Se DERIVA de la biblioteca para que no puedan divergir.
RECOMMENDATIONS = {driver: r.accion for driver, r in BIBLIOTECA.items()}


def build_recommendation_text(drivers: list[str],
                              excluidas: set[str] | None = None) -> str:
    """Texto de apoyo para los drivers detectados, saltando las excluidas.

    `excluidas` son ids de recomendacion apagadas para este nino (Seccion 19).
    Las marcadas `excluible=False` no se pueden apagar.
    """
    if not drivers:
        return DEFAULT_RECOMMENDATION
    acciones = [r.accion for r in recomendaciones_activadas(drivers, excluidas)]
    if not acciones:
        return DEFAULT_RECOMMENDATION
    return "; ".join(acciones[:2])


def recomendaciones_activadas(drivers: list[str],
                              excluidas: set[str] | None = None) -> list[Recomendacion]:
    """Las entradas que se activaron hoy, para poder mostrar la trazabilidad
    completa (id, condicion, contexto, fuente, estado de revision) junto a la
    sugerencia -- Seccion 19, "trazabilidad de recomendaciones"."""
    excluidas = excluidas or set()
    salida = []
    for d in drivers:
        r = BIBLIOTECA.get(d)
        if r is not None and not (r.excluible and r.id in excluidas):
            salida.append(r)
    return salida


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
    "nivel_alerta_sesion": "alerta en la sesión profesional",
}
