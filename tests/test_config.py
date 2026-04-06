"""
test_config.py – Smoke test: verify config loading and accessors.

Run with:
    pytest tests/test_config.py -v
"""

from src.common.config import CFG, get_channel_order, get_sample_rate, get_stft_params
from src.common.constants import CHANNEL_ORDER, SAMPLE_RATE


def test_cfg_is_dict() -> None:
    """CFG should be a non-empty dict."""
    assert isinstance(CFG, dict)
    assert len(CFG) > 0


def test_sample_rate() -> None:
    """get_sample_rate() should return the project sample rate."""
    assert get_sample_rate() == SAMPLE_RATE
    assert get_sample_rate() == 44_100


def test_channel_order() -> None:
    """get_channel_order() should return the 4-element list."""
    order = get_channel_order()
    assert order == CHANNEL_ORDER
    assert order == ["LF", "LR", "RF", "RR"]
    assert len(order) == 4


def test_stft_params_keys() -> None:
    """get_stft_params() should contain the expected keys."""
    params = get_stft_params()
    assert "n_fft" in params
    assert "hop_length" in params
    assert "win_length" in params
    assert "window" in params


def test_stft_params_types() -> None:
    """STFT parameters should have correct types."""
    params = get_stft_params()
    assert isinstance(params["n_fft"], int)
    assert isinstance(params["hop_length"], int)
    assert isinstance(params["window"], str)
