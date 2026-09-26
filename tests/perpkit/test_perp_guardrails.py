"""The order-path guardrails: dry unless --confirm, demo unless --production,
and --production --confirm refused outright, before any key or client.

These run with no SDK, no network and no credentials, which is itself part of
what they check: the refusal must happen before any of those is touched.
"""

from decimal import Decimal

import pytest

from perpkit import close_carry, run_carry
from perpkit.guardrails import (
    DEMO_BASE_URL,
    PRODUCTION_BASE_URL,
    REFUSAL,
    order_host,
    read_host,
)
from perpkit.keys_env import ENV_NAMES, blofin_credentials
from perpkit.strategies.carry import CarryExecutor


@pytest.fixture
def no_keys(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_the_default_host_is_demo():
    assert order_host(production=False, confirm=False) == (DEMO_BASE_URL, "demo")
    assert order_host(production=False, confirm=True) == (DEMO_BASE_URL, "demo")


def test_production_without_confirm_is_a_rehearsal_host():
    assert order_host(production=True, confirm=False) == (PRODUCTION_BASE_URL, "production")
    assert read_host(production=True) == (PRODUCTION_BASE_URL, "production")


def test_production_with_confirm_is_refused_outright():
    with pytest.raises(SystemExit) as refused:
        order_host(production=True, confirm=True)
    assert str(refused.value) == REFUSAL


@pytest.mark.parametrize("tool", [run_carry, close_carry], ids=["run_carry", "close_carry"])
def test_the_order_tools_refuse_before_touching_keys(tool, no_keys):
    """No key is set, so reaching the credential check would raise a
    DIFFERENT SystemExit. Getting the refusal proves it comes first."""
    with pytest.raises(SystemExit) as refused:
        tool.main(["--instrument", "BTC-USDT", "--production", "--confirm"])
    assert str(refused.value) == REFUSAL


@pytest.mark.parametrize("tool", [run_carry, close_carry], ids=["run_carry", "close_carry"])
def test_without_keys_the_order_tools_name_what_is_missing(tool, no_keys, monkeypatch):
    # run_carry first refuses a tier with no transcribed spot fee (VIP 0 has
    # none); pretend one is known so the credential check is reached.
    if hasattr(tool, "SPOT_TAKER_FEE_BPS"):
        monkeypatch.setattr(tool, "SPOT_TAKER_FEE_BPS", Decimal("6"))
    with pytest.raises(SystemExit) as refused:
        tool.main(["--instrument", "BTC-USDT"])
    message = str(refused.value)
    for name in ENV_NAMES:
        assert name in message


def test_credentials_are_read_from_the_environment_and_never_echoed(monkeypatch, capsys):
    monkeypatch.setenv("BLOFIN_API_KEY", "k-test-value")
    monkeypatch.setenv("BLOFIN_API_SECRET", "s-test-value")
    monkeypatch.delenv("BLOFIN_API_PASSPHRASE", raising=False)
    with pytest.raises(SystemExit) as refused:
        blofin_credentials()
    out = capsys.readouterr()
    for secret in ("k-test-value", "s-test-value"):
        assert secret not in str(refused.value)
        assert secret not in out.out and secret not in out.err
    assert "BLOFIN_API_PASSPHRASE" in str(refused.value)

    monkeypatch.setenv("BLOFIN_API_PASSPHRASE", "p-test-value")
    assert blofin_credentials() == ("k-test-value", "s-test-value", "p-test-value")


def test_the_executor_is_dry_unless_told_otherwise():
    class NoBroker:
        def __getattr__(self, name):
            raise AssertionError("a dry executor must not reach the broker for " + name)

    assert CarryExecutor(NoBroker()).dry_run is True


def test_no_cli_takes_a_secret_as_an_argument():
    """A secret on the command line lands in shell history and the process
    list. No parser in the order paths may define such a flag."""
    import inspect

    from perpkit import monitor_carry, plan_carry

    for module in (run_carry, close_carry, plan_carry, monitor_carry):
        source = inspect.getsource(module)
        for flag in ("--api-key", "--secret", "--passphrase", "--key"):
            assert '"' + flag + '"' not in source, (module.__name__, flag)
