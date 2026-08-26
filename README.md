# Antesala

**Anticipar episodios de desregulación 24 horas antes — preguntando lo mínimo.**

Prototipo para el desafío de **Bluba SpA** en NeuroHack 2026.
Facultad de Ingeniería y Ciencias · Universidad de La Frontera.

---

## Qué hace

Antesala toma las bitácoras diarias que Bluba **ya registra** (sueño, rutina,
regulación, episodios, contexto escolar) y responde cuatro preguntas:

| Pregunta | Respuesta del sistema |
|---|---|
| ¿Cuál es el riesgo de mañana? | Probabilidad calibrada de episodio en 24 h |
| ¿Cuánto sabemos realmente hoy? | Índice de suficiencia de información |
| ¿Qué dato falta y **vale la pena** preguntar? | Una sola pregunta al día — o ninguna |
| ¿Qué apoyo preventivo considerar? | Recomendación trazable y excluible |

La idea central: **lo relevante no es cuánto se desvía un niño del promedio de
otros niños, sino cuánto se desvía de su propio patrón.**

> **Antesala no intenta recopilar más datos. Intenta descubrir cuál es el mínimo
> dato que necesitamos hoy para anticiparnos mejor y apoyar antes.**

---

## ▶ Cómo ejecutarlo

**Requisitos:** Python 3.10 o superior. Nada más — sin base de datos que
instalar, sin servicios externos, sin claves de API.

```bash
# 1. Dependencias
pip install -r requirements.txt

# 2. Generar las bitácoras sintéticas  (~10 segundos)
python data/generate_synthetic_data.py --out data/bitacoras.csv

# 3. Entrenar el modelo                (~2 minutos)
python core/train_model.py

# 4. Abrir la aplicación
streamlit run app.py
```

Se abre solo en el navegador. Si no, entra a **http://localhost:8501**.

> Los pasos 2 y 3 se corren **una sola vez**. Después basta con `streamlit run
> app.py`.

### Si algo falla

| Síntoma | Causa y solución |
|---|---|
| `ModuleNotFoundError` | Falta el paso 1: `pip install -r requirements.txt` |
| La app dice "entrena el modelo" | Falta el paso 3: `python core/train_model.py` |
| `FileNotFoundError: data/bitacoras.csv` | Falta el paso 2 |
| El puerto 8501 está ocupado | `streamlit run app.py --server.port 8502` |

---

## 🖥 Qué vas a ver

La aplicación tiene **tres vistas**, separadas por audiencia — porque una familia
y un terapeuta no necesitan la misma pantalla.

### 1 · Hoy · familia

Lo que vería una madre o un padre en su teléfono, dentro de un marco de móvil:

- **Una sola pregunta al día.** No un formulario de 14 campos.
- El **riesgo** de las próximas 24 horas.
- Dos barras **separadas**: *información disponible* y *estabilidad de la
  predicción*. Son cosas distintas y el sistema no las mezcla.
- La explicación en lenguaje cotidiano y el apoyo sugerido.
- **"¿Qué ocurrió después?"** — para cerrar el círculo al día siguiente.

Al lado, un panel **"detrás de la pantalla"** muestra el mecanismo completo: qué
señales se detectaron y de qué fuente vienen, qué del día de hoy movió la cifra,
y el **ranking completo** de preguntas candidatas — se ve que la elegida gana de
verdad.

### 2 · Panel del equipo

Triage de toda la cohorte ordenada por riesgo. Las alertas con información
insuficiente aparecen **en gris y marcadas como no accionables**, en vez de
presentarse como certeza.

Incluye la trazabilidad de cada recomendación (de qué regla salió, con qué
condición, quién responde por ella), la posibilidad de **excluir una estrategia
para un niño concreto**, y la comparación con el modelo interpretable de
referencia.

### 3 · Bitácora completa

Las 14 variables, por si alguien quiere registrarlas todas. Es **opcional**: el
modelo funciona con datos incompletos y guarda cada ausencia como tal, con su
antigüedad, en vez de imputarla en silencio.

---

## 🎬 Guion de demostración (2 minutos)

**Escenario A — Personalización.** Selecciona un niño con 150 días. Mira *"qué
del día de hoy movió la cifra"*: cada dato se compara con **su propio** valor
habitual, no con un promedio general.

**Escenario B — Niño nuevo.** Selecciona uno marcado 🆕 *"Ingreso reciente"* (2 a
4 días). El sistema no saca conclusiones extremas con dos días de datos y lo dice
en pantalla.

**Escenario C — Información incompleta.** ⭐ *El más importante.* Mira la pregunta
del día y el ranking a su derecha. Fíjate en que una variable muy informativa
puede **perder** contra una más barata de responder. Y prueba con **Josefa P.**:
ahí el sistema decide **no preguntar nada**, porque ningún dato faltante mejoraría
la estimación de hoy.

---

## ⚠️ Alcance — importante

El prototipo corre sobre **datos sintéticos**, generados por el propio equipo.

**Lo que demuestra:** que el pipeline funciona, que reacciona correctamente a los
datos que faltan, que la pregunta cambia según el niño y el día, que el sistema
declara cuándo no sabe, y que la interfaz sostiene el flujo completo.

**Lo que NO demuestra:** capacidad predictiva clínica, sensibilidad o
especificidad reales, reducción real de crisis, ni eficacia de las
recomendaciones. Eso requiere datos longitudinales reales de Bluba y validación
prospectiva posterior.

Esta advertencia está también dentro de la aplicación, a la vista.

---

## 📊 Resultados medidos

Sobre datos sintéticos, evaluando en **niños que el modelo nunca vio**:

| Métrica | Valor | Lectura |
|---|---|---|
| AUROC | 0,754 | Discriminación (0,5 = azar) |
| AUPRC | 0,695 | Contra una tasa base de 0,460 |
| Brier | 0,203 | Error de la probabilidad |
| Sensibilidad | 0,80 | Umbral fijado ahí a propósito |
| Falsas alertas | **1,6 por niño y semana** | Demasiado para uso diario |

**No se reporta accuracy a propósito**: con esta tasa base, premia al modelo que
nunca avisa.

Tres resultados que van contra el propio interés del proyecto, y que se reportan
igual:

- **La personalización no se paga en el agregado** (0,754 con ella, 0,753 sin
  ella). Sí paga donde fue diseñada para pagar: en el **día sin ningún registro**
  —el estado en que la app abre cada mañana— donde el AUPRC sube de 0,622 a
  **0,681**.
- **Asumir que "se registra menos los días difíciles" inflaría el AUROC a
  0,925**, pero haría que el 36 % de la importancia del modelo se fuera a *"nadie
  abrió la app"*. La cifra que citamos es el piso, sin ese supuesto: **0,718**.
- **1,6 falsas alertas por niño y semana** es demasiado. El umbral de alerta es
  una decisión clínica sobre cuánta carga tolera una familia, no una constante
  del modelo.

Para reproducirlas:

```bash
python scripts/benchmark.py               # comparadores contra alternativas simples
python scripts/sensibilidad_ausencia.py   # MCAR / MAR / MNAR / mixto
```

---

## 🗂 Estructura del proyecto

```
app.py                          Interfaz Streamlit (tres vistas)
DOCUMENTO_TECNICO.md            Documento técnico completo

core/
  bayesian.py                   Partial pooling; estima k de los datos
  features.py                   Ingeniería de variables, sin fuga temporal
  train_model.py                Entrena Random Forest + logística de referencia
  risk_model.py                 Motor de riesgo + índice de suficiencia
  calibration.py                Calibración de Platt
  uncertainty.py                Incertidumbre predictiva (separada)
  question_selector.py          La pregunta del día: información − carga
  explanation.py                Explicabilidad local contra la línea base
  recommendations.py            Biblioteca auditable (15 entradas)
  intervention_log.py           "¿Qué ocurrió después?" (SQLite)
  narrative.py                  Explicación en lenguaje natural
  evaluation.py                 Panel de métricas y particiones

data/generate_synthetic_data.py Generador de bitácoras
scripts/benchmark.py            Comparadores y ablaciones
scripts/sensibilidad_ausencia.py Sensibilidad al mecanismo de ausencia
tests/                          114 tests
```

`data/antesala.db` se crea sola al usar la app (guarda los seguimientos) y no se
versiona: es estado local de la demo.

---

## 🧪 Tests

```bash
pytest
```

**114 tests.** Cubren el partial pooling contra un ejemplo numérico verificado a
mano, la ausencia de fuga temporal, el motor de riesgo, el selector de preguntas
con su descuento por carga, la separación entre riesgo / suficiencia /
incertidumbre, la calibración, la explicabilidad local, la biblioteca de
recomendaciones, el registro de intervenciones, las métricas y particiones, y el
**renderizado completo de las tres vistas** de la interfaz.

Los tests que necesitan el modelo entrenado se saltan solos si aún no existe.

Para inspeccionar la ingeniería de variables paso a paso:

```bash
python scripts/validate_features.py
```

---

## 📄 Documentación técnica

**[`DOCUMENTO_TECNICO.md`](DOCUMENTO_TECNICO.md)** — el documento completo:
formulación del partial pooling y por qué es empirical Bayes, tratamiento de
datos faltantes con el análisis de sensibilidad, arquitectura del modelo,
separación de riesgo/suficiencia/incertidumbre, adquisición activa de
información, explicabilidad, estrategia de validación en cuatro fases,
comparadores, y consideraciones de privacidad y derechos.
