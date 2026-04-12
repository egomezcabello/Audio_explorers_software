"""
test_imports.py – Smoke test: verify that all src packages can be imported.

Run with:
    pytest tests/test_imports.py -v
"""

import importlib

import pytest

# Every module that should be importable
MODULES = [
    "src",
    "src.common",
    "src.common.constants",
    "src.common.paths",
    "src.common.config",
    "src.common.logging_utils",
    "src.common.io_utils",
    "src.common.stft_utils",
    "src.common.json_schema",
    "src.common.plotting",
    "src.member1_doa",
    "src.member2_enhance",
    "src.member3_analysis",
    "src.member4_fusion",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_import(module_name: str) -> None:
    """Each module should import without errors."""
    mod = importlib.import_module(module_name)
    assert mod is not None
