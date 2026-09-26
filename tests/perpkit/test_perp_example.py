"""The offline perpkit example runs end to end and its rigged carry wins.

It is what the README tells a new user to run first, so it must not rot.
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("numpy")

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "perpkit_factor_example.py"


def test_the_factor_example_runs(capsys):
    spec = importlib.util.spec_from_file_location("perpkit_factor_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main() == 0
    out = capsys.readouterr().out
    assert "carry_7" in out and "CLEARS COST" in out
