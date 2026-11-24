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


def test_outputs_logits_not_probabilities():
    m = build_model(load_config())
    assert not any(isinstance(x, torch.nn.Sigmoid) for x in m.modules())  # sigmoid is applied outside
    assert isinstance(m.head, torch.nn.Conv2d)
