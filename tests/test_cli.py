"""The command-line entry points a new user runs first, offline."""

import runpy
from pathlib import Path

import pytest

from predkit import backtest, keys, replay
from predkit.keys import Keystore
from predkit.rawlog import RawEventLog
from predkit.record_series import write_contract
from predkit.schema import Contract, utc

ROOT = Path(__file__).resolve().parents[1]


def test_replay_prints_the_rebuilt_touch(tmp_path, capsys):
    c = Contract("kalshi", "KX-1", "q", utc(2026, 9, 13, 2, 15), "kalshi:KX", "default")
    d = tmp_path / "kalshi" / "KX-1"
    write_contract(d, c)
    base = c.resolves_at.timestamp()
    clock = [base - 120]
    log = RawEventLog(d, clock=lambda: clock[0])
    for step, (yes, no) in enumerate([(48, 50), (51, 48)]):
        clock[0] = base - 120 + step * 60
        log.write("book_poll", {"orderbook": {"yes": [[yes, 10]], "no": [[no, 10]]}})
    log.close()
    assert replay.main([str(d), "--every", "1"]) == 0
    out = capsys.readouterr().out
    # yes bid 48c, no bid 50c -> YES ask 0.50; then 51 / 48 -> ask 0.52
    assert "bid 0.48  ask 0.5\n" in out and "bid 0.51  ask 0.52" in out
    assert "2 book events" in out


def test_replay_without_a_contract_says_how_to_get_one(tmp_path):
    with pytest.raises(SystemExit, match="contract.json"):
        replay.main([str(tmp_path)])


def test_the_backtest_example_runs_and_beats_its_shuffles(capsys):
    runpy.run_path(str(ROOT / "examples" / "backtest_example.py"), run_name="not_main")["main"]()
    out = capsys.readouterr().out
    assert "real result beat 100% of shuffles" in out


def test_backtest_cli_on_rows_without_a_signal_trades_nothing(tmp_path, capsys):
    example = runpy.run_path(str(ROOT / "examples" / "backtest_example.py"), run_name="not_main")
    path = tmp_path / "rows.jsonl"
    backtest.write_rows(path, example["synthetic_rows"](n=20))
    assert backtest.main(["--rows", str(path), "--strategy", "simple_taker", "--shuffles", "0"]) == 0
    out = capsys.readouterr().out
    assert "20 rows, 0 with a signal" in out and "trades 0" in out


def test_keys_cli_reads_secrets_from_the_environment_never_argv(tmp_path, monkeypatch, capsys):
    store_path = tmp_path / "keys.json"
    monkeypatch.setenv("PREDKIT_PASSPHRASE", "correct horse")
    monkeypatch.setenv("TEST_SECRET", "not-a-real-key")
    assert keys.main(["--keystore", str(store_path), "add", "polymarket", "--secret-env", "TEST_SECRET"]) == 0
    assert Keystore(store_path).load("polymarket", "correct horse") == b"not-a-real-key"
    assert b"not-a-real-key" not in store_path.read_bytes()
    keys.main(["--keystore", str(store_path), "list"])
    assert "polymarket" in capsys.readouterr().out
    keys.main(["--keystore", str(store_path), "remove", "polymarket"])
    assert Keystore(store_path).names() == []


def test_record_saves_each_contract_and_survives_a_failed_lookup(tmp_path):
    from predkit.record import MarketRecorder, _write_contracts
    from predkit.record_series import read_contract

    good = Contract("kalshi", "KX-1", "q", utc(2026, 9, 13, 2, 15), "kalshi:KX", "default")

    class Source:
        name = "kalshi"

        def market(self, market_id):
            if market_id != "KX-1":
                raise KeyError(market_id)
            return good

    recorders = [MarketRecorder(Source(), m, tmp_path / m) for m in ("KX-1", "KX-404")]
    _write_contracts(Source(), recorders)
    assert read_contract(tmp_path / "KX-1") == good
    assert not (tmp_path / "KX-404" / "contract.json").exists()
    for recorder in recorders:
        recorder.close()
