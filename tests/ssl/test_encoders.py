from __future__ import annotations

import torch

from mammo_lejepa.ssl.encoders import build_encoder, registered_encoder_names
from mammo_lejepa.ssl.projector import build_projector


def test_resnet50_is_registered() -> None:
    assert "resnet50" in registered_encoder_names()


def test_build_encoder_raises_for_unknown_name() -> None:
    import pytest

    with pytest.raises(ValueError, match="desconocido"):
        build_encoder("not-a-real-encoder")


def test_resnet50_forward_matches_declared_out_dim() -> None:
    encoder, out_dim = build_encoder("resnet50")
    encoder.eval()

    batch = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        embeddings = encoder(batch)

    assert embeddings.shape == (2, out_dim)


def test_projector_maps_encoder_output_to_projection_dim() -> None:
    _encoder, out_dim = build_encoder("resnet50")
    projector = build_projector(out_dim, out_dim=64)

    embeddings = torch.randn(4, out_dim)
    projections = projector(embeddings)

    assert projections.shape == (4, 64)
