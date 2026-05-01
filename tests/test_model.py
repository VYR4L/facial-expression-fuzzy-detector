"""
Teste unitário — YOLOv11AUDetector
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest

from core import YOLOv11AUDetector
from config.settings import NUM_AUS


@pytest.fixture
def model():
    return YOLOv11AUDetector(in_channels=3, base_channels=16, num_aus=NUM_AUS)


def test_output_shapes(model):
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert "binary_logits" in out and "intensity" in out
    assert out["binary_logits"].shape == (2, NUM_AUS)
    assert out["intensity"].shape == (2, NUM_AUS)


def test_intensity_range(model):
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    assert out["intensity"].min().item() >= 0.0
    assert out["intensity"].max().item() <= 3.0


def test_predict_binary(model):
    x = torch.randn(2, 3, 224, 224)
    pred = model.predict(x)
    assert pred["binary"].dtype == torch.bool
    assert pred["binary"].shape == (2, NUM_AUS)


def test_gradient_flow(model):
    from utils.au_loss import AULoss
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    criterion = AULoss()
    B = x.size(0)
    targets = {
        "binary": torch.randint(0, 2, (B, NUM_AUS)).float(),
        "intensity": torch.rand(B, NUM_AUS) * 3,
    }
    losses = criterion(out, targets)
    losses["loss"].backward()
    first_param = next(model.backbone.parameters())
    assert first_param.grad is not None
