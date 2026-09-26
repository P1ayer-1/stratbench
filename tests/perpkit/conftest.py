"""perpkit tests: no network, no credentials, and no BloFin SDK required.

Modules that need the SDK (the BloFin websocket feed, the order paths)
import it lazily or are skipped with `pytest.importorskip("blofin")`; the
rest of the package imports without it. numpy-based analysis tests skip
when numpy is absent, so a plain `pip install .[dev]` still runs green.
"""
