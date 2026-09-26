import pytest

from predkit.keys import Keystore


def test_round_trip_and_wrong_passphrase(tmp_path):
    store = Keystore(tmp_path / "keys.json")
    store.store("kalshi", b"-----BEGIN PRIVATE KEY-----\nabc\n", "hunter2", meta={"key_id": "kid"})
    assert store.load("kalshi", "hunter2") == b"-----BEGIN PRIVATE KEY-----\nabc\n"
    assert store.meta("kalshi") == {"key_id": "kid"}
    with pytest.raises(PermissionError):
        store.load("kalshi", "wrong")
    assert b"BEGIN PRIVATE KEY" not in (tmp_path / "keys.json").read_bytes()


def test_an_empty_passphrase_is_refused(tmp_path):
    with pytest.raises(ValueError):
        Keystore(tmp_path / "k.json").store("x", b"secret", "")
