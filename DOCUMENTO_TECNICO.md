# ANTESALA

**Sistema personalizado de apoyo para anticipar episodios de desregulación y facilitar apoyos preventivos**

NeuroHack 2026 · Desafío Bluba SpA
Facultad de Ingeniería y Ciencias · Universidad de La Frontera
Agosto de 2026

---

> **Nota sobre esta versión.** Este documento describe el sistema **tal como está
> construido**, no como se planea construirlo. Donde antes decía "se evaluará",
> "se podrá" o "se utilizará", ahora dice qué se hizo y qué número dio. Las
> cifras provienen de ejecutar `scripts/benchmark.py` y
> `scripts/sensibilidad_ausencia.py` sobre el repositorio, y son reproducibles.
>
> Lo que **no** se implementó también está declarado, con el motivo.

---

## 1. Resumen ejecutivo

Antesala es un prototipo de sistema de apoyo a decisiones para el desafío de
Bluba SpA en NeuroHack 2026. Usa la información cotidiana que ya registran
familias, terapeutas y equipos educativos para estimar el riesgo de un episodio
de desregulación en las siguientes 24 horas y, al mismo tiempo, **reducir la
carga de registro** necesaria para obtener una recomendación útil.

La premisa: el patrón relevante no es cuánto se desvía un niño del promedio de
otros niños, sino **cuánto se desvía de su propio patrón habitual**.

Cuatro componentes:

1. **Línea base individual con partial pooling.** Con historial suficiente, la
   referencia es el propio niño; con historial corto, se combina prudentemente
   con la referencia poblacional.
2. **Modelo de riesgo a 24 horas.** Random Forest calibrado sobre variables
   diarias y derivadas temporales.
3. **Separación explícita entre riesgo, suficiencia de información e
   incertidumbre predictiva.** Un riesgo alto no es lo mismo que un riesgo
   confiable, y el sistema no los confunde.
4. **Adquisición activa de información: "la pregunta del día".** En vez de
   pedir siempre el mismo formulario, el sistema identifica qué dato faltante
   aportaría más, **descontando lo que cuesta pedirlo** — y puede decidir no
   preguntar nada.

**Alcance.** El MVP corre sobre datos sintéticos. Demuestra viabilidad técnica y
lógica de funcionamiento; **no** demuestra capacidad predictiva clínicamente
validada. Eso requiere datos longitudinales reales de Bluba y evaluación
prospectiva posterior.

---

## 2. Problema y oportunidad

Bluba dispone de registros cotidianos de familias, profesionales y equipos
educativos: sueño, rutina, estado basal, salud gastrointestinal, alimentación,
comportamiento observado, nivel de apoyo, interacciones sociales y episodios de
regulación o desregulación.

El desafío no es solo construir un clasificador. Hay cinco dificultades
adicionales:

- cada niño presenta patrones diferentes;
- el historial puede ser muy corto;
- los registros diarios pueden estar incompletos;
- no todos los datos faltantes tienen la misma importancia;
- **un sistema que exige demasiada información aumenta la carga de las familias
  y destruye la adherencia.**

La oportunidad es transformar la bitácora existente en un sistema adaptativo
capaz de responder no solo *"¿cuál es el riesgo de mañana?"*, sino también
*"¿qué sabemos realmente hoy?"*, *"¿qué dato falta y vale la pena preguntar?"*,
*"¿qué cambió respecto del patrón habitual del niño?"* y *"¿qué apoyo preventivo
podría considerarse?"*.

---

## 3. Alcance del sistema

Antesala es un **sistema de apoyo a decisiones**, no una herramienta diagnóstica
ni un sustituto del juicio de familias, terapeutas o profesionales.

El sistema estima riesgo, identifica cambios respecto de una línea base, muestra
la suficiencia de los datos disponibles, prioriza información faltante, genera
explicaciones breves y vincula señales con recomendaciones previamente definidas
y auditables.

El sistema **no** interpreta emociones, intenciones ni estados internos como
hechos, ni pretende determinar por sí solo por qué ocurre una desregulación.

---

## 4. Definición de la variable objetivo

Una definición operacional del evento es un requisito previo, no un detalle: un
modelo no puede ser más consistente que la variable con la que se entrena.

**Lo que está implementado.** El generador sintético define:

```
crisis_hoy = 1  si  nivel_regulacion_general_dia == "Desregulación Frecuente"
                O   hubo un evento de intensidad ≥ 8 (severa) ese día

crisis_24h(día d) = crisis_hoy(día d+1)
```

Es una definición **verificable y auditable** — está en el código, no en una
frase. Pero es provisional: la eligió el equipo de desarrollo, no Bluba.

**Lo que falta acordar con Bluba** antes de entrenar con datos reales: qué
conductas constituyen el evento, el nivel mínimo de severidad, el momento de
inicio y término, cómo se registran episodios múltiples, cómo se resuelven
discrepancias entre informantes y qué fuente es el *ground truth*.

Se evita deliberadamente tratar como equivalentes *crisis*, *meltdown*,
*desregulación*, *estrés* y *agitación*: el estado del arte mezcla objetivos
distintos y reconoce problemas de etiquetado.

---

## 5. Variables de entrada

Antesala usa **únicamente** variables que Bluba ya registra, y derivadas de
ellas. No asume sensores, wearables ni fuentes adicionales.

### 5.1 Variables originales (14)

| Dominio | Variables |
|---|---|
| Sueño | calidad del sueño |
| Inicio del día | modo al despertar, nivel de apoyo requerido |
| Salud física | estado gastrointestinal, cambios en la alimentación |
| Medicación | adherencia a la medicación |
| Rutina | cambios de rutina |
| Conducta | comportamiento observado |
| Activación | estado de alerta |
| Regulación | nivel de regulación general del día, eventos de desregulación |
| Contexto escolar | participación en actividades, alimentación en recreos |
| Interacción | interacciones sociales |

Los nombres y categorías coinciden **exactamente** con los observados en los
datos anonimizados de Bluba y en las capturas de la app móvil, no con nombres
inventados.

### 5.2 Variables derivadas

Promedios móviles de 3 y 7 días; cambio respecto de la línea base individual;
días desde el último episodio; conteo de desregulaciones en ventanas de 3 y 7
días; cambio reciente de rutina; **antigüedad de cada una de las 14 variables**;
indicadores explícitos de dato faltante; fuente del dato; día escolar/no escolar.

La matriz final tiene **35 variables numéricas y 18 categóricas**. Está
verificada sin fuga temporal y con paridad exacta entre entrenamiento e
inferencia (`scripts/validate_features.py`, `tests/test_features.py`).

---

## 6. Personalización mediante partial pooling

### 6.1 Motivación

Un modelo completamente individual es inestable con pocos registros. Uno
completamente poblacional ignora diferencias relevantes. Antesala usa **partial
pooling**: la estimación individual se encoge hacia la referencia poblacional en
proporción a cuánta evidencia propia existe.

### 6.2 La fórmula, y qué es exactamente

```
θᵢ = wᵢ · ȳᵢ + (1 − wᵢ) · μ        con    wᵢ = nᵢ / (nᵢ + k)
```

**Esto no es un modelo jerárquico bayesiano completamente estimado.** No hay
MCMC, ni distribuciones posteriores completas, ni estimación conjunta de
hiperparámetros por verosimilitud marginal. Esa denominación se reserva para una
versión con PyMC.

Dicho eso, la fórmula **no es una heurística inventada**. Para una proporción
—que es el caso de la tasa de desregulación, la variable central— es exactamente
la media posterior de un modelo Beta-Binomial:

```
prior:       θ ~ Beta(α, β)     con  α = k·μ,  β = k·(1−μ)
verosim.:    y | θ ~ Binomial(n, θ)
posterior:   E[θ | y] = (k·μ + n·ȳ) / (k + n) = w·ȳ + (1−w)·μ
```

Es decir: **k es la concentración del prior**, medida en observaciones
equivalentes. Y como μ se estima de la propia población en vez de fijarse a
priori, esto es *empirical Bayes* por definición.

Para variables continuas la misma fórmula es la media posterior del modelo
Normal-Normal jerárquico con `k = σ²/τ²`. Para variables **ordinales** (calidad
de sueño, nivel de apoyo, estado de alerta) se codifican a escala numérica y se
aplica la versión continua: es una aproximación, no el modelo ordinal correcto,
y queda declarada como tal.

### 6.3 El valor de k, estimado en vez de elegido

Un k elegido a mano era el punto más débil de la formulación. Ahora se **estima
de los datos** por el método de los momentos (estimador clásico de Kleinman para
Beta-Binomial, `core/bayesian.estimate_prior_strength`):

```
Var(θ) = μ(1−μ) / (k+1)   →   k = μ(1−μ)/Var(θ) − 1
```

descontando de la varianza observada entre niños el ruido de muestreo esperado.

**Resultado sobre los datos del proyecto:** μ = 0,436, varianza real entre niños
= 0,0116, **k ≈ 20,2** (27 niños con historial suficiente). Es decir: los datos
piden encoger *más* hacia la población de lo que encoge k = 5.

**Y sin embargo casi no importa.** Se reentrenó el sistema completo con ambos:

| k | AUROC | AUPRC | Brier | PPV | FA/niño/sem |
|---|---|---|---|---|---|
| 5 (valor de trabajo) | 0,754 | 0,695 | 0,203 | 0,615 | 1,61 |
| 20 (estimado) | 0,756 | 0,688 | 0,202 | 0,621 | 1,57 |

La diferencia es de ±0,002 en AUROC. **El valor de k no es load-bearing**: la
formulación se sostiene sola. Se mantiene k = 5 en el pipeline por continuidad
con el ejemplo numérico validado en `tests/test_bayesian.py`, y se reporta el k
estimado como justificación empírica.

### 6.4 Evolución posterior

Con datos reales: Beta-Binomial para proporciones, modelos normales o robustos
para continuas, **modelos ordinales propios** para escalas, y logísticos
jerárquicos para el evento. PyMC para posteriores completas.

---

## 7. Tratamiento de datos faltantes

Los datos ausentes **no** se consideran automáticamente normales ni anormales.
El sistema registra explícitamente si el dato está presente, cuántos días han
pasado desde el último registro real, y quién lo registró.

### 7.1 La afirmación MNAR era circular — y ahora está medida

La propuesta original sostenía que *"en la práctica clínica un vacío de
información suele coincidir con los momentos de mayor dificultad para la
familia"*. Es plausible, pero no está demostrado como regla general. Y había un
problema peor: **el generador sintético horneaba ese supuesto en los datos**, el
modelo lo aprendía, y el resultado parecía confirmarlo. Se estaba midiendo el
supuesto, no el mundo.

La posición corregida es la prudente:

> La ausencia de registro **puede** ser informativa, por lo que Antesala
> conserva explícitamente indicadores de ausencia y antigüedad en lugar de
> asumir que los datos faltantes son neutrales. Cuál mecanismo opera en la
> realidad lo dirán los datos de Bluba.

`scripts/sensibilidad_ausencia.py` genera el mismo mundo bajo los cuatro
mecanismos y entrena el mismo modelo en cada uno:

| Mecanismo | % faltante | Peso de los indicadores de ausencia | AUROC | AUPRC | PPV | FA/niño/sem |
|---|---|---|---|---|---|---|
| **MCAR** (el silencio no informa) | 29,1 % | **0,083** | **0,718** | 0,659 | 0,561 | 1,98 |
| MAR (depende de lo observable) | 20,4 % | 0,061 | 0,703 | 0,639 | 0,552 | 2,00 |
| **MNAR fuerte** | 34,0 % | **0,363** | **0,925** | 0,887 | 0,875 | 0,36 |
| Mixto (dataset del repositorio) | 30,7 % | 0,132 | 0,754 | 0,695 | 0,615 | 1,61 |

**Cómo leerlo.** Bajo MNAR fuerte el AUROC sube a 0,925 — pero los indicadores
de "nadie registró nada" se llevan el **36 %** de la importancia del modelo. El
sistema deja de leer al niño y pasa a leer si alguien abrió la app. Presentar
esa cifra sería presentar el supuesto como resultado.

**La cifra honesta es el piso: AUROC 0,718 bajo MCAR**, el escenario en que el
silencio no aporta absolutamente nada. Todo lo que el sistema logre por encima
de eso es un extra que solo se puede reclamar tras comprobar el mecanismo con
datos reales.

---

## 8. Modelo de riesgo

### 8.1 Modelo interpretable de referencia — implementado y desplegado

Una **regresión logística regularizada** se entrena, se guarda en el mismo
bundle que el modelo principal y se muestra junto a él en la interfaz. No es
solo una fila de una tabla comparativa: está desplegada.

| Modelo | AUROC | AUPRC | Brier | PPV | FA/niño/sem |
|---|---|---|---|---|---|
| Random Forest (principal) | **0,754** | 0,695 | 0,203 | 0,615 | 1,61 |
| Logística (referencia) | 0,719 | 0,637 | 0,217 | 0,586 | 1,82 |

Si las dos cifras coinciden, la del bosque es creíble. Si divergen mucho, hay
algo que mirar antes de confiar. Además, los coeficientes tienen **signo**, lo
que permite discutirlos variable por variable con el equipo clínico — algo que
el *feature importance* del Random Forest no permite:

| Variable | Coeficiente |
|---|---|
| tipo de evento: transición de actividad | −0,456 |
| alerta en sesión: alto (sobreexcitado) | +0,436 |
| cambio de rutina: sí | +0,352 |
| participación: no participa | +0,347 |
| nivel de apoyo: *sin registrar* | +0,282 |
| comportamiento: desregulado | +0,275 |

### 8.2 Modelo principal

**Random Forest** de 400 árboles (`min_samples_leaf=5`, `max_features="sqrt"`,
`class_weight="balanced_subsample"`), con imputación por mediana **más indicador
de ausencia explícito** para numéricas y categoría `__missing__` para
categóricas.

**LightGBM no se implementó.** Habría exigido agregar una dependencia el día de
la entrega; el comparador con la logística ya responde la pregunta que
justificaba evaluarlo (*¿la complejidad del modelo se paga?*). Queda como
extensión.

La selección no se basó en accuracy. Ver §17.

---

## 9. Riesgo, suficiencia e incertidumbre: tres números, no uno

Este es el cambio conceptual más importante respecto de la versión anterior del
prototipo, que fundía las tres cosas bajo la palabra "confianza".

### 9.1 Riesgo estimado

La probabilidad calibrada de un episodio en las próximas 24 horas.

### 9.2 Índice de suficiencia de información

```
suficiencia = completitud × factor_historial
```

donde la completitud pondera cada variable por su importancia **según el modelo
entrenado** (no por pesos clínicos manuales), y `factor_historial = n/(n+k)`.

**No es la probabilidad de que la predicción sea correcta.** Es cuánta
información hay. En la interfaz se condensa en alta / moderada / baja porque es
lo legible para una familia, pero se llama *información disponible*, no
*confianza*.

### 9.3 Incertidumbre predictiva

Se mide **por separado**: la dispersión entre los árboles del ensamble,
normalizada por la dispersión máxima posible para esa media (`std / √(p(1−p))`).
Sin normalizar, toda predicción extrema parecería estable solo por ser extrema.

Que son medidas distintas está verificado, no supuesto: sobre una rejilla de
niño × grado de completitud, suficiencia y estabilidad **correlacionan −0,76**.
Se puede tener el registro completo y una predicción inestable, y al revés.

**Qué es y qué no es.** Es un *proxy computacional de inestabilidad*, no una
incertidumbre calibrada: mide desacuerdo entre modelos que comparten datos y
sesgo, así que subestima sistemáticamente la incertidumbre real. Se usa para
priorizar preguntas —un uso relativo— y se muestra como banda cualitativa, nunca
como intervalo de confianza formal.

### 9.4 Calibración

El Random Forest promedia árboles, así que sus probabilidades se comprimen hacia
el centro. Mientras la cifra se muestre como porcentaje, eso importa.

Se probaron **las dos** opciones estándar, siempre fuera de muestra y con el
calibrador sin ver las filas que evalúa:

| | ECE | Brier | Valores distintos |
|---|---|---|---|
| Sin calibrar | 0,0516 | 0,2115 | 2.732 |
| Isotónica | 0,0210 | 0,2101 | **159** |
| **Platt (elegida)** | **0,0157** | **0,2090** | 2.945 |

Platt gana en calibración *y* en Brier. Y hay una razón decisiva adicional: la
isotónica es una **función escalonada** que colapsaba 2.732 probabilidades en
159 escalones, lo que dejaba **muda la explicabilidad local** — todas las
contribuciones salían exactamente 0,000 porque el contrafactual caía en el mismo
escalón. La pendiente del calibrador ajustado es 1,390 (separa más de lo que el
bosque separaba por sí solo).

> **Nota metodológica.** Una primera versión reportaba ECE = 0,0000 tras
> calibrar. Era un artefacto: se ajustaba el calibrador sobre las mismas
> predicciones con que se lo evaluaba, y la isotónica reproduce la frecuencia
> observada por construcción. La medición actual usa un calibrador que no vio
> las filas que evalúa.

---

## 10. La pregunta del día: adquisición activa de información

Este es el principal diferenciador de Antesala.

Un formulario tradicional pregunta lo mismo todos los días. Antesala pregunta:
*"de lo que todavía no sabemos hoy, ¿qué dato aporta más antes de pedirlo?"*

### 10.1 Ganancia informativa

Para cada variable faltante se simulan sus 2-3 valores más probables (según el
historial propio del niño, o el poblacional si es nuevo), se recalcula la
dispersión del ensamble en cada escenario y se promedia ponderando por la
probabilidad de cada uno. La ganancia es la reducción esperada de esa
dispersión.

**Nomenclatura.** Antes esto se presentaba como *Expected Value of Information*.
Estrictamente no lo es: el valor de información se define respecto de una
decisión y su función de pérdida, y aquí se optimiza la estabilidad de una
predicción, no una decisión. Es **adquisición activa de información**, aproximada
por reducción esperada de varianza — el objetivo estándar de *active learning*.
Es novedoso sin sobreprometer.

### 10.2 Carga de registro — el término que faltaba

La variable más informativa no siempre es la que vale la pena preguntar.

```
U_j = g_j − λ · c_j
```

donde `g_j` es la ganancia relativa a la mejor candidata del día (0-1) y `c_j` la
carga normalizada. **λ = 0,35**: es el único parámetro subjetivo del mecanismo, y
está aislado en una constante para poder discutirlo con el equipo clínico.

La carga **no es un número puesto a dedo**. Se descompone en tres factores
observables en la app real:

```
costo = n_campos × peso_informante + carga_emocional
```

| Pregunta | Campos | Informante | Carga emocional | Costo | Normalizado |
|---|---|---|---|---|---|
| Registro de un episodio | 4 | familia (1,0) | 0,80 (revivir el episodio) | **4,80** | 1,00 |
| Participación / interacción / recreos | 1 | **colegio (2,2)** | 0,00 | 2,20 | 0,46 |
| Regulación del día, comportamiento, alerta, apoyo | 1 | familia | 0,30 (juicio del día) | 1,30 | 0,27 |
| Estado gastrointestinal | 1 | familia | 0,15 (dato íntimo) | 1,15 | 0,24 |
| Sueño, despertar, medicación, rutina, alimentación | 1 | familia | 0,00 | 1,00 | 0,21 |

El peso 2,2 del informante externo refleja que la respuesta es **asíncrona** y
puede no llegar hoy.

### 10.3 El sistema puede decidir no preguntar

Si ninguna variable faltante tiene ganancia esperada positiva, la respuesta
correcta no es "pregunta lo más barato": es **no preguntar**. Un sistema que
existe para reducir la carga de registro tiene que poder quedarse callado.

Ocurre de verdad: sobre las bitácoras sintéticas, en un día sin ningún registro
hay **7 de 28 niños** para los que ninguna pregunta ayuda; con tres campos ya
anotados baja a **2 de 28**.

---

## 11. Explicabilidad local

> "La explicación no se basará simplemente en *feature importance* global. Para
> cada predicción se identificará qué factores observados contribuyeron a alejar
> el estado actual de la línea base."

Implementado en `core/explanation.py`. Para cada variable **registrada hoy**:

```
contribución_j = riesgo(hoy) − riesgo(hoy con la variable j en su valor habitual)
```

El "valor habitual" sale del historial propio del niño, o de la población si
tiene poco historial — el mismo criterio de partial pooling aplicado a *"qué es
normal para este niño"*. Positiva = ese dato empujó el riesgo hacia arriba
respecto de lo normal en él; negativa = lo empujó hacia abajo (factor protector).

Es una atribución **contrafactual contra la línea base individual**, que es
literalmente lo que pide la sección. El *feature importance* global sigue
mostrándose, pero al lado y respondiendo lo que responde: qué importa en
promedio, no qué pasó hoy.

**Límites, declarados en la interfaz.** Se mide una variable a la vez, así que
las contribuciones **no suman** el riesgo total y nunca se presentan como
descomposición exacta; si dos variables solo importan juntas, el método reparte
mal ese efecto (un método aditivo tipo Shapley lo haría mejor y queda en el
*roadmap*). Y describe lo que el modelo usa, **no una causa**: nunca se emite
"la crisis ocurrirá porque durmió mal".

---

## 12. Recomendaciones accionables: biblioteca auditable

No se usa un LLM para inventar estrategias. `core/recommendations.py` es una
**biblioteca explícita de 15 entradas**, cada una con los cinco campos exigidos:

| Campo | Contenido |
|---|---|
| `id` | REC-01 … REC-15 |
| `condicion` | cuándo se activa, en lenguaje verificable |
| `accion` | qué se sugiere hacer |
| `contexto` | cuándo aplica y **cuándo no** |
| `fuente` | de dónde sale |
| `estado_revision` | `pendiente` / `revisada` + firma |
| `excluible` | si se puede apagar para un niño |

**Las 15 están marcadas `pendiente` de revisión clínica.** Eso no es un
descuido: es el punto. Una biblioteca auditable tiene que poder decir qué **no**
está validado. Inventar citas a literatura clínica para llenar ese campo sería
fabricar evidencia, que es peor que declarar el vacío. Cuando el equipo de Bluba
revise una entrada, cambia su estado y firma; el sistema no necesita ningún otro
cambio.

**Dos entradas no son excluibles** a propósito: verificar la administración de un
medicamento (REC-06) y derivar al equipo tratante (REC-15) son canales de
seguridad, no preferencias de estilo.

El contexto es tan importante como la acción. Ejemplos reales de la biblioteca:

- *Reducir la demanda no es suspender toda actividad: quitarle la estructura al
  día puede aumentar la desregulación.*
- *Retirarse puede ser una estrategia de regulación eficaz. No debe leerse por
  defecto como un problema que haya que corregir.*
- *SOLO verificar y comunicar. El sistema no sugiere iniciar, suspender ni
  ajustar ninguna medicación: eso es decisión médica.*

---

## 13. Registro del resultado de la intervención

Implementado en `core/intervention_log.py` sobre **SQLite**.

Tras una recomendación, el sistema registra: si apareció o no una desregulación,
qué apoyo se utilizó, si fue aceptado, si pareció útil y si generó dificultades.

**Por qué importa más de lo que parece.** Sin esto, Antesala es un sistema de una
sola dirección: predice, sugiere, y nunca se entera de si sirvió. Es además el
único mecanismo por el que el prototipo genera un dato que **hoy no existe en
ninguna bitácora**: el par `aviso → apoyo → resultado`. La bitácora de Bluba
registra lo que le pasa al niño, no qué hizo el adulto tras recibir un aviso ni
si funcionó. Ese pareo es exactamente lo que hará falta para las fases 3 y 4 de
la validación, y no se puede reconstruir después.

También cierra el círculo ético: una recomendación que nadie evalúa es una
instrucción; una que se evalúa y se puede apagar es apoyo a la decisión.

**Advertencia que el sistema muestra en pantalla.** El agregado *"qué apoyo
parece funcionar"* es una **tabulación descriptiva, no evidencia de
efectividad**. Los apoyos no se asignan al azar: se eligen justamente en los días
que pintan peor, así que el apoyo que aparezca con más desregulaciones puede ser
el que se usa en los días más difíciles, no el que funciona peor. Separar una
cosa de otra requiere diseño prospectivo.

---

## 14. Arquitectura

```
1. Captura de registros
2. Normalización al esquema real de Bluba
3. Control de calidad
4. Indicadores de ausencia y antigüedad        ← nada se imputa en silencio
5. Variables temporales derivadas
6. Línea base individual con partial pooling
7. Predicción de riesgo + calibración + incertidumbre
8. Pregunta activa · explicación local · recomendación · seguimiento
```

La interfaz llama directamente a `core/`. **FastAPI no se implementó**: es una
capa de transporte que no cambia ninguna de las respuestas del sistema, y
agregarla el día de la entrega solo añadía superficie de fallo. Queda como
extensión para la integración real con la plataforma de Bluba.

---

## 15. Interfaz

Tres vistas, separadas **por audiencia**, porque las bases piden explicaciones
"para las familias y profesionales" y no son la misma persona:

**Hoy · familia** — replica la app móvil dentro de un marco de teléfono. Una
sola pregunta al día, el riesgo, dos barras separadas (*información disponible* y
*estabilidad de la predicción*), la explicación en lenguaje cotidiano, el apoyo
sugerido y el registro de *"¿qué ocurrió después?"*.

**Panel del equipo** — triage de la cohorte por riesgo, con las alertas de
información insuficiente **suprimidas en gris** en vez de presentadas como
certeza. Incluye la trazabilidad de cada recomendación (id, condición, contexto,
fuente, estado de revisión), la exclusión por niño, la comparación con el modelo
de referencia y el agregado de seguimientos.

**Bitácora completa** — las 14 variables, opcional.

Al lado del teléfono, un panel *"detrás de la pantalla"* muestra el mecanismo:
señales detectadas con su fuente, la explicación local, el **ranking completo**
de preguntas candidatas por utilidad neta (se ve que la elegida gana de verdad, y
que algunas variables incluso aumentarían la incertidumbre), y las métricas.

---

## 16. Datos sintéticos: qué validan y qué no

No se dispone de un dataset real de Bluba. El dataset sintético
(**3.759 registros, 28 niños**: 25 con 150 días + 3 de arranque en frío con 2, 3
y 4 días; tasa base de `crisis_24h` = 0,438) **sirve para probar**:

- que el pipeline funciona y no tiene fuga temporal;
- que los indicadores temporales se calculan correctamente;
- que el partial pooling se comporta como se espera;
- que la pregunta del día cambia según el niño y el día;
- que la interfaz responde a distintos escenarios;
- que el sistema detecta cuándo dispone de poca información.

**No sirve para afirmar** sensibilidad o especificidad clínica real, reducción
real de crisis, eficacia de las recomendaciones ni generalización a niños
reales. Los datos los genera el mismo equipo que escribe el modelo, así que la
relación que el modelo encuentra es, por construcción, la que se programó.

Esta distinción está declarada en el bundle del modelo, en los tres scripts de
evaluación y **visible en la propia interfaz**.

---

## 17. Estrategia de validación

### Fase 1 — Validación técnica (hecha)

**114 tests automatizados**: partial pooling contra el ejemplo numérico del
documento, ingeniería de variables sin fuga, motor de riesgo, selector de
pregunta con carga, separación riesgo/suficiencia/incertidumbre, calibración,
explicabilidad local, biblioteca de recomendaciones, registro de intervención,
métricas y particiones, y renderizado de las tres vistas de la interfaz.

### Los dos regímenes de generalización

Un solo split no responde la pregunta. Son dos preguntas distintas:

| Régimen | Partición | Pregunta que responde |
|---|---|---|
| Niños no vistos | por niño | *cuando llega un niño nuevo, ¿sirve?* |
| Días futuros | por tiempo | *entrenado con lo que ya pasó, ¿acierta mañana?* |

Mezclar filas del mismo niño y periodo entre entrenamiento y test infla los
resultados. Hay un test que verifica explícitamente que ningún día de test
precede a un día de entrenamiento del mismo niño.

**Resultados** (umbral fijado para sensibilidad 80 %):

| Régimen | AUROC | AUPRC | Brier | Sens. | PPV | FA/niño/sem | No detectados |
|---|---|---|---|---|---|---|---|
| Niños no vistos | 0,754 | 0,695 | 0,203 | 0,800 | 0,615 | **1,61** | 83/415 |
| Días futuros | 0,739 | 0,659 | 0,208 | 0,799 | 0,567 | **1,85** | 83/413 |

Tasas base: 0,460 y 0,433 (es el piso del AUPRC).

**No se reporta accuracy a propósito**: con esta tasa base premia al modelo que
nunca avisa.

**Las 1,6 falsas alertas por niño y por semana son un número incómodo, y por eso
se reportan.** Es demasiado para uso diario sostenido. El umbral de alerta es una
decisión clínica sobre cuánta carga tolera una familia, no una constante del
modelo — por eso se reporta siempre junto a la sensibilidad.

### Fases 2 a 4 (pendientes, requieren datos reales)

**Fase 2 — retrospectiva**: división temporal y por niño, prevención de fuga,
comparación contra baselines. Métricas: sensibilidad, PPV, AUROC, AUPRC, Brier,
calibración, falsas alertas por niño/semana, crisis no detectadas, y **desempeño
por niño y en niños nuevos** por separado.

**Fase 3 — prospectiva**: anticipación real a 24 h, estabilidad, calidad de la
calibración, utilidad de la pregunta del día.

**Fase 4 — piloto de uso**: carga de registro, frecuencia de alertas, tasa de
respuesta, falsas alarmas, utilidad percibida, aceptabilidad para niños y
familias, *alert fatigue*.

---

## 18. Comparadores: ¿la complejidad se paga?

Decir "Random Forest con partial pooling y variables temporales" no significa
nada si un promedio de dos líneas hace lo mismo. `scripts/benchmark.py` lo mide.

**Régimen: niños no vistos** (tasa base 0,460)

| Modelo | AUROC | AUPRC | Brier | Sens. | PPV | FA/niño/sem |
|---|---|---|---|---|---|---|
| A · tasa global | 0,500 | 0,460 | 0,249 | 1,000 | 0,460 | 3,78 |
| B · tasa individual cruda | 0,634 | 0,558 | 0,238 | 0,800 | 0,524 | 2,34 |
| C · partial pooling (θ) solo | 0,645 | 0,564 | 0,230 | 0,800 | 0,526 | 2,32 |
| D · logística sin personalizar | 0,719 | 0,637 | 0,216 | 0,800 | 0,598 | 1,73 |
| E · RF sin personalizar | 0,749 | 0,699 | 0,205 | 0,800 | 0,608 | 1,66 |
| **F · ANTESALA completo** | **0,754** | 0,695 | **0,203** | 0,800 | **0,615** | **1,61** |

**Qué se lee aquí, sin adornos.** El salto grande está entre C y D/E: lo que
paga es **mirar el registro de hoy con un modelo**, no la línea base sola. Y F
apenas mejora sobre E: **en el agregado, la personalización no se paga**.

Las ablaciones lo confirman: quitar el bloque de personalización mueve el AUROC
de 0,754 a 0,753; quitar las variables temporales lo deja en 0,756.

### 18.1 Dónde sí paga la personalización

El promedio está dominado por días normales de niños con 150 días de historial,
donde el registro de hoy ya lo dice casi todo. Si la personalización sirve, tiene
que ser donde el registro de hoy **no alcanza**:

| Segmento | n | Sin personalización | ANTESALA completo |
|---|---|---|---|
| Todos los días | 903 | 0,753 / 0,690 | 0,754 / 0,695 |
| **Día SIN ningún registro** | 69 | 0,666 / 0,622 | **0,685 / 0,681** |
| Arranque en frío (≤ 14 días) | 93 | 0,730 / 0,692 | 0,700 / 0,656 |
| Día normal, historial largo | 750 | 0,766 / 0,708 | 0,767 / 0,707 |

*(AUROC / AUPRC)*

**La personalización paga exactamente en el día sin ningún registro** — que es
el estado en que la app abre cada mañana, antes de que nadie anote nada: AUPRC
0,622 → 0,681. Es un reclamo más pequeño y más preciso que el original, y es
defendible.

En arranque en frío el resultado va en contra (0,730 → 0,700), pero con n = 93
está dentro del ruido. Se reporta igual.

**Advertencia de alcance.** Esta tabla compara arquitecturas entre sí sobre el
mismo problema simulado. No mide capacidad predictiva clínica.

---

## 19. Seguridad, privacidad y derechos

No es una nota final: es parte de la arquitectura.

**Implementado en el prototipo:**

- **Minimización de datos.** El registro de intervenciones guarda el
  identificador seudónimo del niño, la fecha, las cifras que el sistema mostró y
  las respuestas del adulto. Nada que permita reidentificar. La interfaz advierte
  explícitamente de no escribir datos personales en el campo de notas.
- **Trazabilidad de recomendaciones.** Cada sugerencia es rastreable hasta la
  regla que la activó, con su condición, contexto, fuente y estado de revisión.
- **Opción de no aplicar una estrategia.** Exclusión por niño, persistida, con
  motivo. El filtro se aplica en el motor, no en la pantalla, para que una
  estrategia descartada no reaparezca en el párrafo generado.
- **Separación de roles.** Vistas distintas para familia y equipo profesional,
  con distinto nivel de detalle y distinto lenguaje.
- **El sistema declara cuándo no sabe** en vez de emitir un número que nadie
  debería usar.

**Requerido para una implementación real:** control de acceso, cifrado,
retención limitada, consentimiento del adulto responsable, **asentimiento del
niño** cuando sea posible y apropiado, posibilidad de no participar, y revisión
de usos secundarios.

El sistema debe apoyar a la persona, no convertirse en una herramienta de
vigilancia o de control de conducta.

---

## 20. Stack tecnológico

| Componente | Tecnología | Estado |
|---|---|---|
| Procesamiento | Python 3.10 | ✅ |
| Datos | pandas / numpy | ✅ |
| Modelado | scikit-learn | ✅ |
| Persistencia de seguimientos | SQLite | ✅ |
| Interfaz | Streamlit | ✅ |
| Tests | pytest (114) | ✅ |
| Versionamiento | Git / GitHub | ✅ |
| API | FastAPI | ❌ extensión |
| Modelo alternativo | LightGBM | ❌ extensión |
| Jerárquico completo | PyMC | ❌ Nivel 2 |
| Atribución aditiva | SHAP | ❌ *roadmap* |

Las cuatro ausencias son deliberadas y están justificadas en sus secciones. No
se consideran necesarias para demostrar el núcleo del producto.

---

## 21. Roadmap

**Hecho (MVP de hackatón).** Dataset sintético con mecanismos de ausencia
configurables · pipeline temporal sin fuga · partial pooling con k estimado ·
Random Forest calibrado · modelo interpretable de referencia · riesgo +
suficiencia + incertidumbre separados · pregunta del día con carga · explicación
local · biblioteca auditable · registro de intervención · interfaz de tres vistas
· comparadores y análisis de sensibilidad.

**Versión posterior.** Datos reales de Bluba · definición operacional acordada ·
modelo jerárquico formal (PyMC) · evaluación prospectiva · integración por API ·
métodos conformales para intervalos con garantía · atribución aditiva (SHAP).

**Futuro opcional.** La literatura muestra potencial en fuentes multimodales
(HR/HRV, actividad, conductancia, contexto ambiental) pero también problemas de
comodidad, generalización y desempeño fuera de laboratorio. Los sensores son una
**extensión**, no una carencia del prototipo:

> Mientras gran parte de la literatura se orienta a nuevos *wearables*, Antesala
> empieza aprovechando información que Bluba **ya registra**, reduciendo costo,
> intrusión y barreras de adopción.

---

## 22. Riesgos y mitigaciones

| Riesgo | Mitigación | Estado |
|---|---|---|
| No hay datos reales | Presentar la demo como validación funcional | ✅ declarado en la interfaz |
| Evento objetivo mal definido | Acordar definición con Bluba | ⚠️ definición provisional en código |
| Pocos datos por niño | Partial pooling | ✅ |
| Datos incompletos | Indicadores de ausencia + antigüedad | ✅ 14/14 variables |
| **Depender del supuesto MNAR** | Medir los cuatro mecanismos y citar el piso | ✅ AUROC 0,718 bajo MCAR |
| Falsas alarmas | Reportar FA/niño/semana junto a la sensibilidad | ✅ 1,61 |
| Overfitting | Validación por sujeto **y** temporal | ✅ |
| Predicciones opacas | Baseline interpretable + explicación local | ✅ |
| Exceso de preguntas | Utilidad = información − carga; puede no preguntar | ✅ |
| Recomendaciones inapropiadas | Biblioteca cerrada, revisable y excluible | ✅ |
| Probabilidades mal calibradas | Platt, medido fuera de muestra | ✅ ECE 0,016 |
| Sobrepromesa clínica | Declarar alcance en modelo, scripts e interfaz | ✅ |

---

## 23. Diferenciadores

Antesala no compite por tener el algoritmo más complejo. Su propuesta de valor
es integrar cuatro problemas que suelen tratarse por separado:

1. **Personalización progresiva** — aprende el patrón individual sin exigir
   meses de historial.
2. **Incertidumbre explícita** — distingue "riesgo alto" de "riesgo alto con
   información insuficiente", y ambos de "riesgo alto con predicción inestable".
3. **Registro adaptativo** — pregunta solo lo que probablemente genere valor,
   descontando la carga, y **puede no preguntar nada**.
4. **Recomendación accionable y auditable** — cada sugerencia es rastreable,
   excluible y evaluable después.

---

## 24. Propuesta de valor

> **Antesala no intenta recopilar más datos. Intenta descubrir cuál es el mínimo
> dato que necesitamos hoy para anticiparnos mejor y apoyar antes.**

Versión técnica: *una capa de inteligencia personalizada que combina línea base
individual, predicción de riesgo calibrada, suficiencia de información y
adquisición activa de datos para transformar registros cotidianos en apoyo
preventivo accionable.*

---

## 25. Ejemplo completo

Niño con 45 días de historial. Hoy: sueño 2 niveles bajo su línea base, cambio de
rutina, dos episodios en los últimos 3 días, salud gastrointestinal sin
registrar, alimentación sin registrar.

```
Riesgo estimado (24 h)      72 %
Información disponible      moderada
Estabilidad                 estable
Línea base del niño         38 %

Qué del día movió la cifra
  cambio de rutina            ▲ 9 %   (hoy Sí · habitual No)
  calidad del sueño           ▲ 6 %   (hoy Dificultad · habitual Reparador)
  estado de alerta            = igual que de costumbre

Pregunta del día            ¿Presentó molestias gastrointestinales hoy?
  ganancia relativa           1,00
  carga                       1 toque · lo responde la familia · dato íntimo
  utilidad neta               +0,92   (gana a "participación": más informativa
                                       pero hay que pedirla al colegio)

Apoyo sugerido              anticipar el cambio con apoyos visuales antes de que
                            ocurra; anticipar una rutina de sueño más temprana
  REC-09 · sin revisión clínica · excluible
  REC-01 · sin revisión clínica · excluible
```

Si la familia responde, el sistema recalcula. Al día siguiente pregunta *"¿qué
ocurrió después?"*.

---

## 26. Escenarios de demo

**A — Historial largo.** Personalización: el dato se compara con su propio
patrón, y la explicación local muestra qué se apartó de él.

**B — Niño nuevo.** Partial pooling: el sistema evita conclusiones extremas con
dos días de datos y lo dice en pantalla. El dataset incluye tres niños con 2, 3 y
4 días exactamente para esto, etiquetados como *"ingreso reciente"*.

**C — Información incompleta.** La pregunta del día. Debe ocupar la mayor parte
de la demo: es el elemento más distintivo. Incluye el caso en que el sistema
**decide no preguntar nada**.

---

## 27. Criterios de éxito del MVP

| Criterio | Estado |
|---|---|
| Recibe registros | ✅ |
| Calcula variables temporales sin fuga | ✅ verificado por test |
| Produce una línea base personalizada | ✅ |
| Genera una estimación de riesgo calibrada | ✅ ECE 0,016 |
| Informa suficiencia de datos | ✅ separada de la incertidumbre |
| Identifica una pregunta prioritaria | ✅ con descuento por carga |
| Actualiza el resultado tras recibirla | ✅ |
| Genera una recomendación legible y trazable | ✅ |
| Registra qué ocurrió después | ✅ |
| Se demuestra en menos de dos minutos | ✅ |

**No se usa como criterio de éxito una supuesta "accuracy clínica" obtenida
exclusivamente con datos sintéticos.**

---

## 28. Conclusión

Antesala propone pasar de un modelo centrado en *"predecir una crisis"* a un
sistema centrado en *"detectar cambios relevantes, reconocer cuánto sabemos,
preguntar solamente lo necesario y facilitar apoyos preventivos oportunos"*.

Este documento reporta tres resultados que van en contra del propio interés
comercial del proyecto, y los reporta igual porque son lo que hace defendible al
resto:

1. **La personalización no se paga en el agregado** (0,754 vs 0,753). Paga en el
   día sin registro, que es donde fue diseñada para pagar.
2. **Un supuesto MNAR fuerte inflaría el AUROC a 0,925**, pero haría que el
   sistema leyera si alguien abrió la app en vez de leer al niño. La cifra que
   citamos es el piso: **0,718**.
3. **1,6 falsas alertas por niño y por semana** al 80 % de sensibilidad es
   demasiado para uso diario sostenido.

Antesala no pretende demostrar eficacia clínica durante una hackatón. Pretende
demostrar algo más concreto y verificable: que es técnicamente posible
transformar los registros existentes de Bluba en un flujo personalizado,
transparente y adaptativo que reduzca la carga de registro y prepare una base
sólida para una validación posterior con datos reales.

> **No preguntar todo. No tratar a todos igual. No esconder la incertidumbre.
> Preguntar lo mínimo que realmente ayuda a decidir mejor.**

---

## Reproducibilidad

Todas las cifras de este documento salen de:

```bash
python data/generate_synthetic_data.py --out data/bitacoras.csv
python core/train_model.py                 # panel de evaluación y calibración
python scripts/benchmark.py                # §18: comparadores, ablaciones, k
python scripts/sensibilidad_ausencia.py    # §7: MCAR / MAR / MNAR / mixto
pytest                                     # §17 fase 1: 114 tests
```
