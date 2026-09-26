"""Every perpkit command-line module parses and exposes `main`.

A syntax error in a command-line wrapper is invisible to a suite that only
imports the libraries underneath it, and the wrapper is the only part a
person actually types. Parsing rather than importing: an import of the order
paths would want the SDK, the network and credentials, and these tests use
none of them.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "perpkit"
ENTRYPOINTS = sorted(
    [path for path in PACKAGE.glob("*.py")
     if path.name == "record.py"
     or path.name.startswith(("record_", "plan_", "run_", "close_", "monitor_"))]
    + [path for path in (PACKAGE / "analysis").glob("*.py")
       if path.name not in {"__init__.py", "stats.py", "panel.py", "blofin_spot.py"}]
)


def test_the_entrypoints_were_found():
    names = {path.name for path in ENTRYPOINTS}
    assert {"record.py", "record_hyperliquid.py", "run_carry.py",
            "replay.py"} <= names


@pytest.mark.parametrize("path", ENTRYPOINTS, ids=lambda p: p.name)
def test_entrypoint_parses(path):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize("path", ENTRYPOINTS, ids=lambda p: p.name)
def test_entrypoint_defines_main(path):
    """Each is run as `python -m perpkit.<name>`, and the convention is
    `raise SystemExit(main())` so an exit code means something to a
    scheduler."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "main" in names, path.name


MANGLED = ((0x07, "bell", "a"), (0x08, "backspace", "b"),
           (0x0b, "vertical tab", "v"), (0x0c, "form feed", "f"))


def test_no_source_file_carries_a_stray_control_character():
    """A bell or a form feed in a source file is a botched `\\a` or `\\f`,
    usually a Windows path eaten by a shell heredoc, and invisible in a diff."""
    for path in PACKAGE.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        raw = path.read_bytes()
        for code, name, letter in MANGLED:
            assert bytes([code]) not in raw, "{} in {} (a mangled backslash-{})".format(
                name, path, letter)
