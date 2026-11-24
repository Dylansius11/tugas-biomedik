import torch

from cxrseg.config import load_config
from cxrseg.model import build_model, count_params


def test_output_shape_and_params():
    m = build_model(load_config())
    m.eval()
    with torch.no_grad():
        y = m(torch.zeros(2, 1, 256, 256))
    assert y.shape == (2, 1, 256, 256)
    assert 5e6 < count_params(m) < 12e6


def test_logits_not_squashed():
    m = build_model(load_config())
    m.eval()
    with torch.no_grad():
        y = m(torch.rand(1, 1, 256, 256))
    assert y.min() < 0 or y.max() > 1  # raw logits, sigmoid is applied outside
