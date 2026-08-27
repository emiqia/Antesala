"""
Diagramas de arquitectura de Antesala, para la presentacion y el documento.

Genera DOS diagramas distintos porque tienen dos audiencias distintas:

  assets/arquitectura_pipeline.{png,svg}
      Tecnico. Las 8 etapas de la Seccion 14 del documento, cada una con el
      archivo de codigo que la implementa. Para el documento y para el jurado
      que pregunta por dentro.

  assets/arquitectura_simple.{png,svg}
      Amigable. Cuatro pasos, sin jerga ni formulas. Para la diapositiva de
      "como funciona": si el jurado tiene que leer notacion matematica en una
      pantalla a cinco metros, el diagrama fallo.

Ninguno es aspiracional: todas las etapas estan implementadas y cubiertas por
tests. Escribe en assets/ y no en docs/, que contiene el material entregado por
Bluba y no se versiona.

Uso:
    python scripts/generate_architecture_diagram.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets"

# Paleta identica a la de la app (app.py), para que el diagrama y el producto
# se lean como la misma cosa.
PURPLE = "#6b4ec7"
PURPLE_DARK = "#4a3aa7"
PURPLE_DEEP = "#2f2470"
PURPLE_SOFT = "#f5f2fe"
LAVENDER = "#f6f4fc"
BORDER = "#e7e3f5"
GREEN = "#0ca30c"
INK = "#171423"
INK_SOFT = "#5c5875"
MUTED = "#8b89a3"
WHITE = "#ffffff"

plt.rcParams["font.family"] = ["DejaVu Sans"]


# ══════════════════════════════════════════════════ DIAGRAMA TECNICO ═══════
# Las 8 etapas de la Seccion 14. El texto de cada caja dice QUE hace, no como
# se llama la tecnica: "reconoce cuanto sabe" antes que "indice de suficiencia".
ETAPAS = [
    dict(n=1, titulo="Captura", sec="§14 · etapa 1", code="app.py",
         desc="La familia registra la\nbitácora del día"),
    dict(n=2, titulo="Normalización", sec="§5.1 · etapa 2", code="data/",
         desc="Esquema real de Bluba:\n14 variables, sus categorías"),
    dict(n=3, titulo="Ausencia\nexplícita", sec="§7 · etapa 4", code="features.py",
         desc="Qué falta, desde cuándo\ny quién lo registró"),
    dict(n=4, titulo="Ventanas\ntemporales", sec="§5.2 · etapa 5", code="features.py",
         desc="Medias móviles 3/7 días,\nconteos, tendencias"),
    dict(n=5, titulo="Línea base\nindividual", sec="§6 · etapa 6", code="bayesian.py",
         desc="Partial pooling:\nel niño ↔ la población"),
    dict(n=6, titulo="Riesgo\ncalibrado", sec="§8 · §9.4", code="risk_model.py\ncalibration.py",
         desc="Random Forest + Platt.\nEl % significa lo que dice"),
    dict(n=7, titulo="Cuánto sabe,\ncuánto no", sec="§9.2 · §9.3", code="risk_model.py\nuncertainty.py",
         desc="Suficiencia e incertidumbre,\nmedidas por separado"),
    dict(n=8, titulo="Pregunta,\nexplica, aprende", sec="§10 · §11 · §12 · §13",
         code="question_selector.py\nexplanation.py\nintervention_log.py",
         desc="1 pregunta (o ninguna),\nexplicación y seguimiento"),
]

BOX_W, BOX_H = 2.85, 3.05
GAP = 0.35
ROW1_Y, ROW2_Y = 5.95, 2.10
START_X = 1.15

# Canales de ruteo. Las dos flechas largas (salto de fila y ciclo de
# realimentacion) van por carriles reservados en vez de en diagonal: una
# diagonal que cruza el diagrama entero se lee como un error de dibujo, no
# como un flujo.
CANAL_Y = 5.55       # hueco entre las dos filas
RETORNO_Y = 1.15     # carril inferior, por debajo de la fila 2
MARGEN_X = 0.55      # columna libre a la izquierda, para subir el retorno
FIG_W, FIG_H = 14.2, 10.2


def _centro(idx: int, y: float) -> tuple[float, float]:
    return START_X + idx * (BOX_W + GAP) + BOX_W / 2, y + BOX_H / 2


def _caja(ax, etapa: dict, cx: float, cy: float):
    x0, y0 = cx - BOX_W / 2, cy - BOX_H / 2
    ax.add_patch(FancyBboxPatch(
        (x0, y0), BOX_W, BOX_H, boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=1.4, edgecolor=PURPLE_DARK, facecolor=WHITE, zorder=2))

    ax.add_patch(Circle((x0 + 0.32, y0 + BOX_H - 0.32), 0.26,
                        facecolor=PURPLE, edgecolor="none", zorder=3))
    ax.text(x0 + 0.32, y0 + BOX_H - 0.32, str(etapa["n"]), ha="center", va="center",
            fontsize=13, fontweight="bold", color=WHITE, zorder=4)

    # Marca de implementado: ninguna etapa del diagrama es una promesa.
    ax.text(x0 + BOX_W - 0.30, y0 + BOX_H - 0.32, "✓", ha="center", va="center",
            fontsize=13, fontweight="bold", color=GREEN, zorder=4)

    ax.text(cx, y0 + BOX_H - 0.92, etapa["titulo"], ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=INK, linespacing=1.35, zorder=4)
    ax.text(cx, y0 + BOX_H - 1.72, etapa["desc"], ha="center", va="center",
            fontsize=9.3, color=INK_SOFT, linespacing=1.45, zorder=4)

    ax.add_patch(FancyBboxPatch(
        (x0 + 0.22, y0 + 0.26), BOX_W - 0.44, 0.72,
        boxstyle="round,pad=0.01,rounding_size=0.10",
        linewidth=0, facecolor=PURPLE_SOFT, zorder=3))
    ax.text(cx, y0 + 0.62, etapa["code"], ha="center", va="center",
            fontsize=7.8, color=PURPLE_DARK, family="monospace",
            linespacing=1.3, zorder=4)
    ax.text(cx, y0 + 0.08, etapa["sec"], ha="center", va="center",
            fontsize=7.5, color=MUTED, zorder=4)


def _flecha(ax, p0, p1, curva=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=17, linewidth=1.6,
        color=PURPLE, connectionstyle=f"arc3,rad={curva}", zorder=1))


def diagrama_tecnico():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(WHITE)
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    ax.text(MARGEN_X, FIG_H - 0.38, "ANTESALA · arquitectura del sistema",
            fontsize=20, fontweight="bold", color=INK)
    ax.text(MARGEN_X, FIG_H - 0.82, "De la bitácora diaria a una pregunta y un apoyo — 8 etapas, "
            "todas implementadas y con tests",
            fontsize=10.5, color=INK_SOFT)

    centros = {}
    for i, n in enumerate([1, 2, 3, 4]):
        etapa = next(e for e in ETAPAS if e["n"] == n)
        cx, cy = _centro(i, ROW1_Y)
        _caja(ax, etapa, cx, cy)
        centros[n] = (cx, cy)
    for i, n in enumerate([5, 6, 7, 8]):
        etapa = next(e for e in ETAPAS if e["n"] == n)
        cx, cy = _centro(i, ROW2_Y)
        _caja(ax, etapa, cx, cy)
        centros[n] = (cx, cy)

    for a, b in [(1, 2), (2, 3), (3, 4)]:
        _flecha(ax, (centros[a][0] + BOX_W / 2, centros[a][1]),
                (centros[b][0] - BOX_W / 2, centros[b][1]))
    for a, b in [(5, 6), (6, 7), (7, 8)]:
        _flecha(ax, (centros[a][0] + BOX_W / 2, centros[a][1]),
                (centros[b][0] - BOX_W / 2, centros[b][1]))

    # Salto de fila: baja de la etapa 4, cruza por el canal entre filas y entra
    # a la etapa 5 desde arriba. Tres tramos rectos, ningun cruce.
    x4, y4 = centros[4]
    x5, y5 = centros[5]
    ax.plot([x4, x4], [y4 - BOX_H / 2, CANAL_Y], color=PURPLE, linewidth=1.6, zorder=1)
    ax.plot([x4, x5], [CANAL_Y, CANAL_Y], color=PURPLE, linewidth=1.6, zorder=1)
    ax.add_patch(FancyArrowPatch(
        (x5, CANAL_Y), (x5, y5 + BOX_H / 2), arrowstyle="-|>",
        mutation_scale=17, linewidth=1.6, color=PURPLE, zorder=1))

    # El ciclo que cierra el sistema: lo que ocurrio despues vuelve a entrar.
    # Baja al carril inferior, va a la izquierda por fuera de las cajas, sube
    # por el margen y entra a la etapa 1 por su lado izquierdo.
    x8, y8 = centros[8]
    x1, y1 = centros[1]
    gris = dict(color=MUTED, linewidth=1.3, linestyle=(0, (5, 4)), zorder=1)
    ax.plot([x8, x8], [y8 - BOX_H / 2, RETORNO_Y], **gris)
    ax.plot([x8, MARGEN_X], [RETORNO_Y, RETORNO_Y], **gris)
    ax.plot([MARGEN_X, MARGEN_X], [RETORNO_Y, y1], **gris)
    ax.add_patch(FancyArrowPatch(
        (MARGEN_X, y1), (x1 - BOX_W / 2, y1), arrowstyle="-|>",
        mutation_scale=15, linewidth=1.3, color=MUTED, zorder=1))
    ax.text((x8 + MARGEN_X) / 2, RETORNO_Y + 0.22,
            "«¿qué ocurrió después?» realimenta el sistema  ·  §13",
            ha="center", va="bottom", fontsize=8.8, color=MUTED, style="italic")

    ax.text(MARGEN_X, 0.32,
            "116 tests automatizados  ·  sin fuga temporal, verificada por test  ·  "
            "datos sintéticos: valida el funcionamiento, no la eficacia clínica",
            fontsize=8.8, color=MUTED)
    return fig


# ══════════════════════════════════════════════════ DIAGRAMA AMIGABLE ══════
# Cuatro pasos, sin una sola formula. Es el que va en la diapositiva.
# Sin emoji: DejaVu Sans no los tiene y matplotlib los dibujaria como cuadros
# vacios. Circulos numerados, que ademas repiten el lenguaje visual de las
# insignias del diagrama tecnico.
PASOS = [
    dict(titulo="Lo que ya se\nregistra",
         desc="Sueño, rutina, ánimo,\nepisodios del día"),
    dict(titulo="Comparado con\nSU patrón",
         desc="No con el promedio\nde otros niños"),
    dict(titulo="Riesgo de\nmañana",
         desc="Y qué tan seguro\nestá el sistema"),
    dict(titulo="Una pregunta\ny un apoyo",
         desc="Solo una. A veces,\nninguna"),
]


def diagrama_simple():
    fig, ax = plt.subplots(figsize=(13.6, 5.0))
    fig.patch.set_facecolor(LAVENDER)
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    ax.text(6.8, 4.52, "Cómo funciona Antesala", ha="center",
            fontsize=21, fontweight="bold", color=INK)

    w, h, gap = 2.7, 2.45, 0.72
    total = 4 * w + 3 * gap
    x0 = (13.6 - total) / 2
    y0 = 1.15

    for i, paso in enumerate(PASOS):
        bx = x0 + i * (w + gap)
        cx = bx + w / 2
        ax.add_patch(FancyBboxPatch(
            (bx, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.20",
            linewidth=0, facecolor=WHITE, zorder=2))
        ax.add_patch(Circle((cx, y0 + h - 0.56), 0.34, facecolor=PURPLE,
                            edgecolor="none", zorder=3))
        ax.text(cx, y0 + h - 0.56, str(i + 1), ha="center", va="center",
                fontsize=17, fontweight="bold", color=WHITE, zorder=4)
        ax.text(cx, y0 + h - 1.20, paso["titulo"], ha="center", va="center",
                fontsize=13, fontweight="bold", color=INK, linespacing=1.3, zorder=3)
        ax.text(cx, y0 + 0.52, paso["desc"], ha="center", va="center",
                fontsize=10, color=INK_SOFT, linespacing=1.45, zorder=3)

        if i < 3:
            _flecha(ax, (bx + w + 0.10, y0 + h / 2), (bx + w + gap - 0.10, y0 + h / 2))

    # El umbral: la linea que separa el hoy del mañana, motivo de la identidad.
    ax.plot([x0, x0 + total], [0.72, 0.72], color=BORDER, linewidth=2, zorder=1)
    ax.text(6.8, 0.34, "Sin wearables. Sin dispositivos. Solo lo que la familia ya anota.",
            ha="center", fontsize=10.5, color=PURPLE_DARK, fontweight="bold")
    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for nombre, constructor in [("arquitectura_pipeline", diagrama_tecnico),
                                 ("arquitectura_simple", diagrama_simple)]:
        fig = constructor()
        for ext in ("png", "svg"):
            destino = OUT_DIR / f"{nombre}.{ext}"
            fig.savefig(destino, facecolor=fig.get_facecolor(),
                        bbox_inches="tight", dpi=200 if ext == "png" else None)
            print(f"Guardado: {destino.relative_to(ROOT)}")
        plt.close(fig)


if __name__ == "__main__":
    main()
