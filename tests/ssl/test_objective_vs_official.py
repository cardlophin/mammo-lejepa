from __future__ import annotations

import lejepa
import pytest
import torch

from mammo_lejepa.ssl.objective import epps_pulley_quadrature, epps_pulley_statistic

pytestmark = pytest.mark.slow

# Tolerancia: sobre las mismas proyecciones (mismos datos, sin generar
# direcciones), la comparación es exacta hasta la precisión de punto flotante
# — ambas implementaciones ejecutan la misma fórmula cerrada, no un estimador
# con su propio muestreo. 1e-5 relativo cubre el reordenamiento de la suma
# entre implementaciones sin esconder una discrepancia real.
_TOLERANCE = 1e-5


@pytest.mark.parametrize(
    ("num_points", "t_max"),
    [(17, 3.0), (9, 3.0), (25, 5.0), (17, 2.0)],
)
def test_epps_pulley_statistic_matches_official_exactly(
    num_points: float, t_max: float
) -> None:
    """T007: sobre las mismas proyecciones unidimensionales (sin pasar por el
    muestreo de direcciones aleatorias, que cada implementación hace de forma
    distinta y no tiene por qué coincidir), el estadístico de Epps-Pulley
    propio coincide con el de `lejepa.univariate.EppsPulley` dentro de la
    tolerancia declarada arriba."""
    torch.manual_seed(0)
    n_samples, n_slices = 500, 37
    projections = torch.randn(n_samples, n_slices)  # convención oficial: [N, K]

    official_test = lejepa.univariate.EppsPulley(n_points=num_points, t_max=t_max)
    official_stat = official_test(projections)  # [K]

    knots, weights = epps_pulley_quadrature(num_points=num_points, t_max=t_max)
    mine_stat = epps_pulley_statistic(
        projections.transpose(-1, -2), knots=knots, weights=weights
    )  # [K]

    torch.testing.assert_close(mine_stat, official_stat, rtol=_TOLERANCE, atol=1e-6)


def test_epps_pulley_statistic_matches_official_on_collapsed_data() -> None:
    """La coincidencia se sostiene también fuera del régimen isótropo — datos
    muy concentrados, donde el estadístico es grande — no sólo en el caso
    fácil de muestras gaussianas puras."""
    torch.manual_seed(1)
    projections = 0.001 * torch.randn(300, 20)

    official_test = lejepa.univariate.EppsPulley(n_points=17)
    official_stat = official_test(projections)

    knots, weights = epps_pulley_quadrature(num_points=17)
    mine_stat = epps_pulley_statistic(
        projections.transpose(-1, -2), knots=knots, weights=weights
    )

    torch.testing.assert_close(mine_stat, official_stat, rtol=_TOLERANCE, atol=1e-6)


def test_sigreg_full_pipeline_agrees_in_order_of_magnitude_with_official() -> None:
    """Sobre el mismo lote, con direcciones muestreadas independientemente por
    cada implementación (no hay forma de forzar las mismas direcciones sin
    tocar el paquete oficial), `sigreg` propio y
    `SlicingUnivariateTest(EppsPulley(...))` oficial deben coincidir en orden
    de magnitud: es la propiedad que importa para el entrenamiento (que ambos
    penalicen la falta de isotropía de forma comparable), no la igualdad bit
    a bit de un estimador Monte Carlo con muestreo independiente."""
    from mammo_lejepa.ssl.objective import sigreg

    torch.manual_seed(0)
    embeddings = torch.randn(1000, 32)

    official_test = lejepa.univariate.EppsPulley(n_points=17)
    official_loss_fn = lejepa.multivariate.SlicingUnivariateTest(
        univariate_test=official_test, num_slices=1024
    )
    official_value = official_loss_fn(embeddings).item()

    mine_value = sigreg(embeddings, num_slices=1024).item()

    assert mine_value == pytest.approx(official_value, rel=0.5)
