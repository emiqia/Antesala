"""
Genera el diagrama de arquitectura del pipeline de 7 etapas (Seccion 5 del
documento tecnico) como imagen -- para incluir en la presentacion de la
entrega (bases oficiales, Seccion 5.2: "diagramas arquitectonicos").

Cada caja indica la seccion del documento tecnico que la especifica y el
archivo de codigo que la implementa hoy (no es un diagrama aspiracional: las
7 etapas ya estan implementadas y con tests).

Uso:
    python scripts/generate_architecture_diagram.py

Genera docs/arquitectura_pipeline.png (para insertar en slides) y
docs/arquitectura_pipeline.svg (vectorial, editable).
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs"

PURPLE = "#6b4ec7"
PURPLE_DARK = "#4a3aa7"
PURPLE_SOFT = "#f4f2fb"
GREEN = "#0ca30c"
GREEN_SOFT = "#e6f6e6"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
MUTED = "#8b89a3"
WHITE = "#ffffff"

STAGES = [
    dict(n=1, title="Captura del dato", sec="Sección 5 · etapa 1",
         code="app.py",
         desc="Familia registra la bitácora\n(1 sola pregunta, Sección 6.3)"),
    dict(n=2, title="Normalización", sec="Sección 5 · etapa 2",
         code="generate_synthetic_data.py",
         desc="Formato estándar, fuente\ny momento del registro"),
    dict(n=3, title="Indicadores de\nausencia", sec="Sección 4.4 · 5.3",
         code="features.py",
         desc="Valor imputado + indicador +\nantigüedad + fuente, por variable"),
    dict(n=4, title="Ventanas\ntemporales", sec="Sección 4.3 · 5.4",
         code="features.py",
         desc="Promedios móviles 3/7 días,\nconteos, antigüedad de registro"),
    dict(n=5, title="Ajuste bayesiano\njerárquico", sec="Sección 3 · 5.5",
         code="bayesian.py",
         desc="θᵢ = wᵢ·ȳᵢ + (1−wᵢ)·μ\npooling niño ↔ población"),
    dict(n=6, title="Modelo de riesgo", sec="Sección 6.1 · 5.6",
         code="risk_model.py + train_model.py",
         desc="Random Forest (principal) +\nheurístico (respaldo)"),
    dict(n=7, title="Confianza + pregunta\n+ recomendación", sec="Sección 6.2 · 6.3 · 7",
         code="risk_model.py + question_selector.py\n+ recommendations.py",
         desc="Reducción de varianza del\nensamble → 1 sola pregunta"),
]

BOX_W, BOX_H = 2.85, 3.05
GAP = 0.35
ROW1_Y = 5.15
ROW2_Y = 1.30
START_X = 0.55


def box_center(idx_in_row: int, y: float) -> tuple[float, float]:
    x = START_X + idx_in_row * (BOX_W + GAP) + BOX_W / 2
    return x, y + BOX_H / 2


def draw_stage(ax, stage: dict, cx: float, cy: float):
    x0, y0 = cx - BOX_W / 2, cy - BOX_H / 2
    box = FancyBboxPatch(
        (x0, y0), BOX_W, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=1.4, edgecolor=PURPLE_DARK, facecolor=WHITE, zorder=2,
    )
    ax.add_patch(box)

    # Numero en circulo, esquina superior izquierda
    badge = Circle((x0 + 0.32, y0 + BOX_H - 0.32), 0.26,
                    facecolor=PURPLE, edgecolor="none", zorder=3)
    ax.add_patch(badge)
    ax.text(x0 + 0.32, y0 + BOX_H - 0.32, str(stage["n"]),
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=WHITE, zorder=4)

    # Check verde: las 7 etapas ya estan implementadas
    check = Circle((x0 + BOX_W - 0.28, y0 + BOX_H - 0.28), 0.19,
                    facecolor=GREEN_SOFT, edgecolor=GREEN, linewidth=1.3, zorder=3)
    ax.add_patch(check)
    ax.text(x0 + BOX_W - 0.28, y0 + BOX_H - 0.30, "✓",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=GREEN, zorder=4)

    # Offsets ABSOLUTOS desde el techo de la caja (y0+BOX_H), no fracciones:
    # el circulo del numero y el check tienen tamano fijo (0.32/0.26 y
    # 0.28/0.19) sin importar BOX_H, asi que el titulo debe empezar
    # claramente POR DEBAJO de esa franja fija para no pisarlos -- con
    # fracciones de BOX_H, una caja mas alta simplemente empuja el titulo
    # mas abajo en TERMINOS ABSOLUTOS tambien, pero el badge no se mueve,
    # y en cajas bajas quedaban a la misma altura.
    top = y0 + BOX_H
    ax.text(cx, top - 0.78, stage["title"],
            ha="center", va="top", fontsize=11.3, fontweight="bold",
            color=INK, zorder=4, linespacing=1.3)

    ax.text(cx, top - 1.62, stage["sec"],
            ha="center", va="top", fontsize=8.3, color=PURPLE_DARK,
            style="italic", zorder=4)

    ax.text(cx, top - 1.98, stage["desc"],
            ha="center", va="top", fontsize=7.6, color=INK_SOFT,
            linespacing=1.45, zorder=4)

    ax.text(cx, y0 + 0.14, stage["code"],
            ha="center", va="bottom", fontsize=7.0, color=MUTED,
            family="monospace", linespacing=1.3, zorder=4)


def straight_arrow(ax, p0, p1, **kw):
    arr = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16,
                           linewidth=1.6, color=PURPLE_DARK, zorder=1, **kw)
    ax.add_patch(arr)


def main():
    fig, ax = plt.subplots(figsize=(14.2, 7.6), dpi=200)
    fig.patch.set_facecolor(PURPLE_SOFT)
    ax.set_facecolor(PURPLE_SOFT)
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 9.15)
    ax.set_aspect("equal")  # las cajas y los circulos numerados no se deforman
    ax.axis("off")

    # --- Titulo (con margen claro por encima de la fila 1) ---
    ax.text(0.55, 8.95, "Antesala — Arquitectura del pipeline", fontsize=21,
            fontweight="bold", color=INK, ha="left", va="top")
    ax.text(0.55, 8.52, "7 etapas, de la bitácora diaria a la recomendación · Sección 5 del documento técnico"
                          "  ·  NeuroHack 2026, desafío Bluba",
            fontsize=10.5, color=INK_SOFT, ha="left", va="top")

    # --- Trazado en "S": fila 1 se lee izquierda->derecha, fila 2 se lee
    # derecha->izquierda, alineada por columna bajo la fila 1. Evita un
    # conector diagonal largo cruzando todo el diagrama -- la etapa 4 queda
    # justo ARRIBA de la etapa 5, misma columna, conector vertical limpio.
    row1_order = [1, 2, 3, 4]        # columnas 0,1,2,3
    row2_col_by_stage = {5: 3, 6: 2, 7: 1}  # columnas 3,2,1 (columna 0 = salida)

    centers = {}
    for col, n in enumerate(row1_order):
        stage = next(s for s in STAGES if s["n"] == n)
        cx, cy = box_center(col, ROW1_Y)
        centers[n] = (cx, cy)
        draw_stage(ax, stage, cx, cy)
        if col > 0:
            prev_cx, prev_cy = box_center(col - 1, ROW1_Y)
            straight_arrow(ax, (prev_cx + BOX_W / 2, prev_cy), (cx - BOX_W / 2, cy))

    for n, col in row2_col_by_stage.items():
        stage = next(s for s in STAGES if s["n"] == n)
        cx, cy = box_center(col, ROW2_Y)
        centers[n] = (cx, cy)
        draw_stage(ax, stage, cx, cy)

    # Fila 2 se lee de derecha a izquierda: 5 -> 6 -> 7.
    straight_arrow(ax, (centers[5][0] - BOX_W / 2, centers[5][1]), (centers[6][0] + BOX_W / 2, centers[6][1]))
    straight_arrow(ax, (centers[6][0] - BOX_W / 2, centers[6][1]), (centers[7][0] + BOX_W / 2, centers[7][1]))

    # Conector 4 -> 5: mismo eje x, una sola flecha vertical limpia.
    c4, c5 = centers[4], centers[5]
    straight_arrow(ax, (c4[0], c4[1] - BOX_H / 2), (c5[0], c5[1] + BOX_H / 2))

    # --- Caja de salida: continua la fila 2 hacia la izquierda, columna 0 ---
    out_cx, out_cy = box_center(0, ROW2_Y)
    out_box = FancyBboxPatch(
        (out_cx - BOX_W / 2, out_cy - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=1.6, edgecolor=GREEN, facecolor=GREEN_SOFT, zorder=2,
    )
    ax.add_patch(out_box)
    ax.text(out_cx, out_cy + BOX_H * 0.28, "Resultado\nen la app",
            ha="center", va="center", fontsize=11.3, fontweight="bold",
            color=INK, zorder=4, linespacing=1.25)
    ax.text(out_cx, out_cy - BOX_H * 0.20,
            "Riesgo · Confianza\nPregunta del día\nRecomendación",
            ha="center", va="center", fontsize=8.2, color=INK_SOFT, linespacing=1.5, zorder=4)
    c7 = centers[7]
    straight_arrow(ax, (c7[0] - BOX_W / 2, c7[1]), (out_cx + BOX_W / 2, out_cy))

    # --- Pie ---
    ax.text(0.55, 0.55,
            "Las 7 etapas están implementadas y cubiertas por tests (30/30) — ver tests/ y "
            "scripts/validate_features.py.",
            fontsize=8.3, color=INK_SOFT, ha="left", va="bottom")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "arquitectura_pipeline.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(OUT_DIR / "arquitectura_pipeline.svg", facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Guardado: {OUT_DIR / 'arquitectura_pipeline.png'}")
    print(f"Guardado: {OUT_DIR / 'arquitectura_pipeline.svg'}")


if __name__ == "__main__":
    main()
