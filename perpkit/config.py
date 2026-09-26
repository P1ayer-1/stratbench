"""perpkit settings, read from environment variables with conservative defaults.

Nothing here reads a file. Settings and API keys come from the process
environment only (export them in your shell or your service manager), so a
key can never be picked up from a stray file in the working directory. Keys
are read where they are used (`perpkit.keys_env`), never here, and never
logged.

Recording and all public market data need no key at all.
"""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path

from perpkit.fees import (  # noqa: F401 - re-exported: config is where callers look
    COST_MAKER_MAKER_BPS,
    COST_MAKER_TAKER_BPS,
    COST_TAKER_TAKER_BPS,
    MAKER_FEE_BPS,
    MAKER_FEE_RATE,
    ROUND_TRIP_COST_BPS,
    SPOT_MAKER_FEE_BPS,
    SPOT_TAKER_FEE_BPS,
    SPOT_VIP_TIERS,
    TAKER_FEE_BPS,
    TAKER_FEE_RATE,
    VIP_TIER,
    VIP_TIERS,
)


def env_bool(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


# --- Where data goes ---------------------------------------------------------
# Relative to the working directory unless set. Every tool takes --data-dir too.
DATA_DIR = Path(os.getenv("PERPKIT_DATA_DIR", "data"))

# --- BloFin instrument and hosts ---------------------------------------------
INST_ID = os.getenv("BLOFIN_INST_ID", "BTC-USDT")
# Order paths use demo unless a tool is given --production. Public market data
# is always read from production regardless, so a dataset never describes the
# demo book.
USE_DEMO = env_bool("BLOFIN_USE_DEMO", "false")

# --- Microstructure feed -----------------------------------------------------
# "books" = 200 levels with incremental updates (what you want for depth
# features). "books5" = 5 levels, full snapshot each time: lighter, and it
# cannot desync, but deep-book features become meaningless.
BOOK_DEPTH = os.getenv("BLOFIN_BOOK_DEPTH", "books")

# Rolling trade-tape window, seconds. Must exceed the longest trade-flow horizon.
TAPE_WINDOW_SECONDS = float(os.getenv("BLOFIN_TAPE_WINDOW_SECONDS", "60"))

# How long a feed may go with NO market data before it is treated as dead and
# reconnected.
#
# This guards the failure error handling cannot see: the socket stays open,
# pings and pongs keep flowing, and the subscription silently stops
# delivering. Nothing raises, so `supervise` never restarts the loop and the
# recorder writes nothing for hours. The SDK cannot catch it either: its
# receive loop swallows its own read timeout, and `listen()` then blocks on an
# empty queue.
#
# Sizing: `books` on a liquid perp updates several times a second, so total
# silence is anomalous within seconds. 30 s is about two orders of magnitude
# above the normal gap: late enough that no quiet patch trips it, early
# enough that a stall costs half a minute of data rather than a night of it.
FEED_STALL_TIMEOUT_S = float(os.getenv("BLOFIN_FEED_STALL_TIMEOUT_S", "30"))

# --- Feature recording -------------------------------------------------------
RECORD_FEATURES = env_bool("PERPKIT_RECORD_FEATURES", "true")

# Archive the raw websocket messages as gzipped JSONL. Strongly recommended:
# the feature CSV only contains features you thought of today, whereas the raw
# log lets any FUTURE feature be recomputed over all your history via
# `perpkit.analysis.replay`. Costs roughly 10-40 MB/hour compressed per
# instrument.
RECORD_RAW = env_bool("PERPKIT_RECORD_RAW", "true")

# How often to persist a feature row (ms). The engine still computes on every
# event; this only controls disk volume. At a 300 s minimum horizon, sampling
# every 250 ms gives over a thousand near-identical overlapping rows per
# independent observation. Anything finer is recoverable from the archive.
FEATURE_SAMPLE_INTERVAL_MS = int(os.getenv("PERPKIT_FEATURE_SAMPLE_MS", "1000"))

# Forward horizons (seconds) to label, and the move that counts as a signal.
#
# Why minutes rather than seconds: on BTC-USDT the standard deviation of the
# forward move is under a basis point at 1-5 s and a couple of bps at 30 s,
# against a round trip of 4 bps (maker) to 12 bps (taker) at VIP 0. At 30 s an
# oracle that knew the sign and captured a full standard deviation would still
# lose at taker fees. Price scales close to a random walk (sigma ~ sqrt(T)),
# so 300/900/1800 s bracket the range where a modest model could clear costs.
#
# The recorder cannot write a row until its forward window has closed, so
# nothing lands on disk for the first max(horizon) = 30 minutes. That is the
# look-ahead guard, not a hang.
LABEL_HORIZONS = tuple(
    float(part)
    for part in os.getenv("PERPKIT_LABEL_HORIZONS", "300,900,1800").split(",")
    if part.strip()
)
# Defaults to the round-trip cost: labelling a move smaller than fees as "up"
# trains a model to chase edges it cannot capture.
LABEL_THRESHOLD_BPS = float(
    os.getenv("PERPKIT_LABEL_THRESHOLD_BPS", str(ROUND_TRIP_COST_BPS))
)

# --- Risk limits (defaults sized for a demo account) -------------------------
MAX_POSITION_BASE = Decimal(os.getenv("PERPKIT_MAX_POSITION_BASE", "0.05"))
MAX_NOTIONAL = Decimal(os.getenv("PERPKIT_MAX_NOTIONAL", "5000"))
MAX_LEVERAGE = Decimal(os.getenv("PERPKIT_MAX_LEVERAGE", "5"))
MAX_DAILY_LOSS = Decimal(os.getenv("PERPKIT_MAX_DAILY_LOSS", "100"))

# --- Hyperliquid -------------------------------------------------------------
# Public data only; no keys. Coin names are Hyperliquid's own and
# case-sensitive: BTC, not BTC-USDT, and kPEPE, not KPEPE. The recorder checks
# them against the live universe.
HYPERLIQUID_COINS = os.getenv("HYPERLIQUID_COINS", "BTC,ETH,SOL,HYPE")
# REST weight per minute spent reading accounts. The published limit is 1,200
# per IP and an account read weighs 2, so 900 is 450 reads a minute with a
# quarter of the limit left for anything else on this IP.
HYPERLIQUID_WEIGHT_PER_MINUTE = int(os.getenv("HYPERLIQUID_WEIGHT_PER_MINUTE", "900"))
# How often a liquidation map is drawn per coin, and its band width as a
# fraction of mark (0.005 = 0.5%).
HYPERLIQUID_MAP_SECONDS = float(os.getenv("HYPERLIQUID_MAP_SECONDS", "60"))
HYPERLIQUID_BUCKET_PCT = float(os.getenv("HYPERLIQUID_BUCKET_PCT", "0.005"))
