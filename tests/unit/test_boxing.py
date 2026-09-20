from __future__ import annotations

import cv2
import numpy as np

from mammo_lejepa.boxing import align_mask_to_image, box_from_mask, box_iou, resolve_box
from mammo_lejepa.models import BoxingParams, BoxSource, CajaMamaria, FallbackReason
from tests.fixtures.synthetic_mammogram import make_synthetic_mammogram

TOLERANCE_PX = 2


def test_box_from_mask_matches_known_geometry() -> None:
    sm = make_synthetic_mammogram()
    x0_expected, y0_expected, x1_expected, y1_expected = sm.geometry.bounding_box
    margin_px = 10

    assert sm.mask is not None
    box = box_from_mask(sm.mask, margin_px=margin_px)

    assert abs(box.x0 - max(0, x0_expected - margin_px)) <= TOLERANCE_PX
    assert abs(box.y0 - max(0, y0_expected - margin_px)) <= TOLERANCE_PX
    assert abs(box.x1 - (x1_expected + margin_px)) <= TOLERANCE_PX
    assert abs(box.y1 - (y1_expected + margin_px)) <= TOLERANCE_PX
    assert box.source is BoxSource.MASK


def test_mask_excludes_pectoral_and_otsu_does_not() -> None:
    """El test que justifica D-01: con pectoral, la caja de la máscara se
    queda igual (excluye el pectoral) pero la de Otsu crece para incluirlo."""
    sm = make_synthetic_mammogram(include_pectoral=True)
    config = BoxingParams(margin_px=5)

    assert sm.mask is not None
    chosen, otsu_alt, fallback_reason = resolve_box(sm.image, sm.mask, config=config)

    assert fallback_reason is None
    assert chosen.source is BoxSource.MASK
    assert otsu_alt is not None
    assert otsu_alt.source is BoxSource.OTSU

    # La caja de Otsu es sensiblemente mayor: incluye la esquina del pectoral.
    assert otsu_alt.x0 < chosen.x0
    assert otsu_alt.y0 < chosen.y0
    assert otsu_alt.area_ratio > chosen.area_ratio

    iou = box_iou(chosen, otsu_alt)
    assert iou < 0.9  # discrepancia real, no ruido de redondeo


def test_missing_mask_triggers_otsu_fallback() -> None:
    sm = make_synthetic_mammogram(mask_variant="missing")
    config = BoxingParams()

    chosen, otsu_alt, fallback_reason = resolve_box(sm.image, sm.mask, config=config)

    assert fallback_reason is FallbackReason.MISSING_MASK
    assert chosen.source is BoxSource.OTSU
    assert otsu_alt is None


def test_empty_mask_triggers_otsu_fallback() -> None:
    sm = make_synthetic_mammogram(mask_variant="empty")
    config = BoxingParams()

    chosen, otsu_alt, fallback_reason = resolve_box(sm.image, sm.mask, config=config)

    assert fallback_reason is FallbackReason.EMPTY_MASK
    assert chosen.source is BoxSource.OTSU
    assert otsu_alt is None


def test_mismatched_mask_resolution_is_realigned_not_rejected() -> None:
    """Visto en datos reales de Mammo-Bench (`ddsm`: máscara e imagen casi
    nunca comparten resolución). Una máscara más pequeña que la imagen debe
    seguir usándose —realineada—, no descartarse a favor de Otsu."""
    sm = make_synthetic_mammogram(include_pectoral=True)
    config = BoxingParams(margin_px=5)
    assert sm.mask is not None
    smaller_mask = cv2.resize(
        sm.mask,
        (sm.mask.shape[1] // 2, sm.mask.shape[0] // 2),
        interpolation=cv2.INTER_NEAREST,
    )

    chosen, otsu_alt, fallback_reason = resolve_box(
        sm.image, smaller_mask, config=config
    )

    assert fallback_reason is None
    assert chosen.source is BoxSource.MASK
    assert (chosen.image_height, chosen.image_width) == sm.image.shape
    # Sigue excluyendo el pectoral (D-01) pese a la diferencia de resolución.
    assert otsu_alt is not None
    assert otsu_alt.area_ratio > chosen.area_ratio


def test_align_mask_to_image_resizes_with_nearest_neighbor() -> None:
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 10:40] = 255

    aligned = align_mask_to_image(mask, (100, 200))

    assert aligned.shape == (100, 200)
    assert set(np.unique(aligned)) <= {0, 255}


def test_align_mask_to_image_is_a_no_op_when_shapes_already_match() -> None:
    mask = np.zeros((50, 50), dtype=np.uint8)

    aligned = align_mask_to_image(mask, (50, 50))

    assert aligned is mask


def test_saturated_mask_triggers_otsu_fallback() -> None:
    sm = make_synthetic_mammogram(mask_variant="saturated")
    config = BoxingParams()

    chosen, otsu_alt, fallback_reason = resolve_box(sm.image, sm.mask, config=config)

    assert fallback_reason is FallbackReason.SATURATED_MASK
    assert chosen.source is BoxSource.OTSU
    assert otsu_alt is None


def test_box_iou_identical_boxes_is_one() -> None:
    box = CajaMamaria(
        x0=10,
        y0=10,
        x1=110,
        y1=110,
        margin_px=0,
        source=BoxSource.OTSU,
        threshold=100.0,
        image_height=200,
        image_width=200,
        area_ratio=0.25,
    )

    assert box_iou(box, box) == 1.0


def test_box_iou_disjoint_boxes_is_zero() -> None:
    a = CajaMamaria(
        x0=0,
        y0=0,
        x1=50,
        y1=50,
        margin_px=0,
        source=BoxSource.OTSU,
        threshold=100.0,
        image_height=200,
        image_width=200,
        area_ratio=0.0625,
    )
    b = CajaMamaria(
        x0=100,
        y0=100,
        x1=150,
        y1=150,
        margin_px=0,
        source=BoxSource.OTSU,
        threshold=100.0,
        image_height=200,
        image_width=200,
        area_ratio=0.0625,
    )

    assert box_iou(a, b) == 0.0


def test_box_iou_known_overlap() -> None:
    # a: 100x100 en (0,0)-(100,100); b: 100x100 en (50,50)-(150,150).
    # Intersección: 50x50=2500. Unión: 10000+10000-2500=17500. IoU=2500/17500.
    a = CajaMamaria(
        x0=0,
        y0=0,
        x1=100,
        y1=100,
        margin_px=0,
        source=BoxSource.OTSU,
        threshold=100.0,
        image_height=200,
        image_width=200,
        area_ratio=0.25,
    )
    b = CajaMamaria(
        x0=50,
        y0=50,
        x1=150,
        y1=150,
        margin_px=0,
        source=BoxSource.OTSU,
        threshold=100.0,
        image_height=200,
        image_width=200,
        area_ratio=0.25,
    )

    assert abs(box_iou(a, b) - (2500 / 17500)) < 1e-9
