"""
Valida core/bayesian.py contra el ejemplo numerico EXACTO de la Seccion 3.4
del documento tecnico (Nino A / Nino B, mu=0.20) -- es la pieza que mas
necesita quedar demostrablemente correcta para el jurado.
"""
import math

import pandas as pd

from core.bayesian import K_DEFAULT, compute_baseline, shrinkage_weight


def test_k_default_matches_doc():
    # Seccion 3.4: "en la version simplificada de Antesala se usa k = 5".
    assert K_DEFAULT == 5


def test_shrinkage_weight_matches_doc_example():
    # Nino A: 30 dias de historial -> w = 30/35 (el documento redondea a 0.86).
    w_a = shrinkage_weight(30, K_DEFAULT)
    assert math.isclose(w_a, 30 / 35, rel_tol=1e-9)
    assert round(w_a, 2) == 0.86

    # Nino B: 2 dias de historial -> w = 2/7 (el documento redondea a 0.29).
    w_b = shrinkage_weight(2, K_DEFAULT)
    assert math.isclose(w_b, 2 / 7, rel_tol=1e-9)
    assert round(w_b, 2) == 0.29


def test_theta_matches_doc_worked_example():
    mu = 0.20  # tasa general de dias con desregulacion (Seccion 3.4)

    # Nino A: historial largo, ybar=0.40 -> theta ~= 37%.
    w_a = shrinkage_weight(30, K_DEFAULT)
    theta_a = w_a * 0.40 + (1 - w_a) * mu
    assert round(theta_a, 2) == 0.37

    # Nino B: recien llegado, ybar=1.00 (2 de 2 dias con crisis) -> theta ~= 43%.
    w_b = shrinkage_weight(2, K_DEFAULT)
    theta_b = w_b * 1.00 + (1 - w_b) * mu
    assert round(theta_b, 2) == 0.43


def test_shrinkage_weight_edge_cases():
    # Sin historial propio, el peso del dato individual es 0.
    assert shrinkage_weight(0, K_DEFAULT) == 0.0
    # Con mucho historial (n >> k), el peso se acerca a 1.
    assert shrinkage_weight(1000, K_DEFAULT) > 0.99


def test_compute_baseline_converges_to_own_average_with_long_history():
    """Con historial muy largo, theta converge al promedio propio (ybar),
    no al poblacional -- el otro extremo del ejemplo del documento."""
    rows = [{"child_id": "nino_x", "valor": 0.9} for _ in range(200)]
    rows += [{"child_id": "otro", "valor": 0.1} for _ in range(50)]
    logs = pd.DataFrame(rows)

    base = compute_baseline(logs, "nino_x", "valor")
    assert base.n == 200
    assert math.isclose(base.ybar, 0.9, rel_tol=1e-9)
    assert base.w > 0.97          # w = 200/205
    assert base.theta > 0.88      # cerca de ybar (0.9), lejos de mu (~0.74)


def test_compute_baseline_falls_back_to_population_mean_without_own_history():
    """Nino sin ningun dato propio: theta = mu (arranque en frio total)."""
    rows = [{"child_id": "otro", "valor": 0.5} for _ in range(10)]
    logs = pd.DataFrame(rows)

    base = compute_baseline(logs, "nino_nuevo", "valor")
    assert base.n == 0
    assert base.ybar is None
    assert base.w == 0.0
    assert math.isclose(base.theta, base.mu, rel_tol=1e-9)
