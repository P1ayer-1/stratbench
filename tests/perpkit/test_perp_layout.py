"""Where recorded data lives, and the boundary the tools must not cross.

`data/raw` was hardcoded in two argparse defaults, and moving the archive to
`data/<INST-ID>/raw` silently broke both of them - the tools ran, found
nothing, and said so only because the glob happened to be empty. Centralising
the layout is what stops the next move breaking the next tool; these tests are
what stop the centralised version drifting.
"""

import pytest

from perpkit.layout import INSTRUMENT_DIR, instrument_dirs, resolve_raw_dir


def make_archive(root, instrument):
    path = root / instrument / "raw" / "2026-09-09"
    path.mkdir(parents=True)
    (path / "books-00.jsonl.gz").write_bytes(b"")
    return root / instrument / "raw"


def test_an_instrument_directory_is_recognised_by_shape():
    for name in ("BTC-USDT", "ADA-USDT", "1000BONK-USDT", "BTC-USD"):
        assert INSTRUMENT_DIR.fullmatch(name), name


def test_tool_directories_are_not_instruments():
    """The reason this is a shape and not a deny-list.

    `data/replayed` was read as an instrument until this rule replaced an
    explicit exclusion of `raw`, and the next tool directory would have been
    read as one too.
    """
    for name in ("raw", "replayed", "bars", "binance", "cache", "cross",
                 "tardis", "some-thing"):
        assert not INSTRUMENT_DIR.fullmatch(name), name


def test_instrument_dirs_finds_only_instruments(tmp_path):
    for name in ("BTC-USDT", "ADA-USDT", "replayed", "bars"):
        (tmp_path / name).mkdir()
    assert sorted(instrument_dirs(tmp_path)) == ["ADA-USDT", "BTC-USDT"]


def test_a_single_archive_needs_no_naming(tmp_path):
    expected = make_archive(tmp_path, "ADA-USDT")
    assert resolve_raw_dir(tmp_path) == expected


def test_several_archives_are_refused_with_a_usable_suggestion(tmp_path):
    make_archive(tmp_path, "ADA-USDT")
    make_archive(tmp_path, "BTC-USDT")

    with pytest.raises(SystemExit) as caught:
        resolve_raw_dir(tmp_path)
    message = str(caught.value)
    assert "more than one instrument" in message
    assert "--instrument ADA-USDT" in message


def test_naming_the_instrument_picks_it_out(tmp_path):
    make_archive(tmp_path, "ADA-USDT")
    expected = make_archive(tmp_path, "BTC-USDT")
    assert resolve_raw_dir(tmp_path, "BTC-USDT") == expected


def test_naming_an_instrument_with_no_archive_lists_what_exists(tmp_path):
    make_archive(tmp_path, "ADA-USDT")
    with pytest.raises(SystemExit, match="ADA-USDT"):
        resolve_raw_dir(tmp_path, "PUMP-USDT")


def test_an_instrument_directory_without_an_archive_is_not_a_candidate(tmp_path):
    """A features-only directory must not make the choice ambiguous."""
    expected = make_archive(tmp_path, "ADA-USDT")
    (tmp_path / "BTC-USDT").mkdir()
    (tmp_path / "BTC-USDT" / "features-2026-09-09.csv").write_text("ts\n")

    assert resolve_raw_dir(tmp_path) == expected


def test_the_legacy_flat_archive_still_resolves(tmp_path):
    """Data recorded before per-instrument scoping sits in a bare data/raw."""
    legacy = tmp_path / "raw" / "2026-09-07"
    legacy.mkdir(parents=True)
    assert resolve_raw_dir(tmp_path) == tmp_path / "raw"


def test_a_legacy_archive_beside_scoped_ones_is_refused(tmp_path):
    """It records its symbol nowhere, so it cannot be placed automatically."""
    (tmp_path / "raw" / "2026-09-07").mkdir(parents=True)
    make_archive(tmp_path, "ADA-USDT")

    with pytest.raises(SystemExit, match="both an unscoped raw"):
        resolve_raw_dir(tmp_path)


def test_no_archive_at_all_says_how_to_make_one(tmp_path):
    with pytest.raises(SystemExit, match="perpkit.record"):
        resolve_raw_dir(tmp_path)
