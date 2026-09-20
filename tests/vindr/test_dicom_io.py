from __future__ import annotations

from pathlib import Path

import pytest

from mammo_lejepa.errors import DecodeError
from mammo_lejepa.vindr.dicom_io import read_dicom
from tests.vindr.fixtures.synthetic_dicom import make_synthetic_dicom, save_dicom


def test_read_dicom_returns_expected_payload(tmp_path: Path) -> None:
    dataset, _geometry = make_synthetic_dicom(
        height=200,
        width=150,
        photometric_interpretation="MONOCHROME2",
        laterality="R",
        view_position="MLO",
        manufacturer="ACME",
    )
    path = save_dicom(dataset, tmp_path / "img.dicom")

    payload = read_dicom(path)

    assert payload.rows == 200
    assert payload.columns == 150
    assert payload.pixels.shape == (200, 150)
    assert payload.photometric_interpretation == "MONOCHROME2"
    assert payload.manufacturer == "ACME"
    assert payload.model_name == "SyntheticGenerator"
    assert payload.pixel_spacing == (0.1, 0.1)
    assert payload.window_center is not None
    assert payload.window_width is not None


def test_read_dicom_without_window_returns_none(tmp_path: Path) -> None:
    dataset, _geometry = make_synthetic_dicom(include_window=False)
    path = save_dicom(dataset, tmp_path / "img.dicom")

    payload = read_dicom(path)

    assert payload.window_center is None
    assert payload.window_width is None


def test_read_dicom_raises_decode_error_on_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.dicom"
    path.write_bytes(b"\x00" * 128 + b"DICM" + b"\xff" * 64)

    with pytest.raises(DecodeError):
        read_dicom(path)
