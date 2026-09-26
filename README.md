# stratbench

A strategy-testing bench with two toolkits: **predkit** for prediction
markets (Kalshi, Polymarket) and **perpkit** for crypto perpetual futures
(BloFin, Hyperliquid, Binance/Bybit liquidation feeds), plus a team of
research-lab agents for Claude Code. Both toolkits record raw venue data
first, replay it, and backtest net of transcribed fees with honest controls.

**predkit** records, replays and backtests strategies on **Kalshi** and
**Polymarket**, and runs them on paper (or, deliberately, live) with a risk
veto in front of every order. Most of this README describes it; perpkit has
its own section, [perpkit (crypto perpetual futures)](#perpkit-crypto-perpetual-futures).

Every instrument on these venues is a binary contract priced between 0 and
1 that resolves at a known time from the venue's own source. Max loss is
the price paid; there is no leverage, liquidation or funding. That one fact
shapes the whole package.

> **Disclaimer.** This is research software, provided "as is" under the MIT
> licence, with no warranty of any kind. Nothing in this repository is
> financial, investment, legal or tax advice, and no result it produces is a
> promise of future returns. Trading prediction markets can lose all the
> money you put in. **Leveraged perpetual futures are high risk: a leveraged
> position can be liquidated and lose all of its margin in minutes, and
> funding, fees and spreads accrue whether or not a trade works.** Access to
> prediction markets and to crypto derivatives venues is restricted or
> prohibited in many jurisdictions: **you are responsible for following each
> venue's terms of service and the laws where you live.** The geofence in
> predkit is a floor, not legal clearance (see [Live trading](#live-trading)),
> and perpkit has no geofence at all. The strategies shipped in either
> toolkit are teaching examples with no known edge.

## Contents

- [Architecture: the rules that span files](#architecture-the-rules-that-span-files)
- [Layout](#layout)
- [Install](#install)
- [Quickstart](#quickstart)
- [Live trading](#live-trading)
- [Adding a strategy](#adding-a-strategy)
- [Fee tables](#fee-tables)
- [What is verified and what is not](#what-is-verified-and-what-is-not)
- [perpkit (crypto perpetual futures)](#perpkit-crypto-perpetual-futures)
- [Research lab (Claude Code agents)](#research-lab-claude-code-agents)
- [Licence](#licence)

## Architecture: the rules that span files

**One currency: the price of YES.** `schema.py` is the shape every venue is
translated into. A Kalshi NO bid at 52c is a YES ask at 0.48, and a
Polymarket NO-token book is mirrored onto the YES book, both inside
`venues/`, so nothing above the adapters ever handles a NO price. An
`OrderIntent`'s `price` is the price of the outcome named (buy NO at 0.52
pays 0.52), `notional` is always dollars paid, and `yes_delta` is the signed
exposure. Every price is a `Decimal`.

**The raw archive comes first.** `rawlog.py` writes every venue message
verbatim as `{t, n, m}` gzip JSON lines under
`data/<venue>/<market>/raw/<day>/<channel>-<HH>.jsonl.gz`. Derived data (a
rebuilt book, a fill simulation, a labelled row) can be regenerated from the
archive; the archive cannot be regenerated from anything. So the recorder
writes before it parses, and a parse failure is counted, never fatal. A
writer never appends (a restart in the same hour writes `.r001` beside the
first file), and the reader recovers every gzip member of a file torn by a
hard kill. Channels merge on `(t, n)`; a reference feed recorded by another
process joins on `t` only.

**Three verbs, one directory per strategy.** `strategies/<name>/` has
`plan.py` (computes intents and has no path to an order endpoint; it returns
*every* failing gate), `execute.py` (dry unless told, reports the same shape
either way, closes only what its own ledger opened) and `monitor.py`
(read-only, verifies against the venue rather than the plan). `decide`
applies the same thresholds to a backtest row. The lifecycle is Protocols
in `strategies/__init__.py`, satisfied structurally with no base class;
`tests/test_strategy_contract.py` checks every registered strategy.

**Fees decide every verdict.** `fees.py` transcribes the `c * p * (1 - p)`
curve per venue and tier, each row with a date and a source, and checks the
tables at import (taker >= maker everywhere, zero at the extremes,
symmetric, peaks at 50c, no unexplained duplicate tiers, builder rates
within program caps). An unknown tier is an error that names the known
ones: fees are never guessed. See [Fee tables](#fee-tables).

**Labels never see the future.** `backtest.label_row` writes a row only
after resolution and only from the contract's own `resolution_source`; spot
exchanges are refused by name, because the settlement minute is exactly
where a spot print and the venue's settlement differ. `backtest.run` holds
to resolution with non-overlapping holds, charges the venue fee for the
entry's role plus any builder fee, and reports the *mean* of shuffled-label
controls and the percentile the real result beat, never the best draw.
`replay.MakerFillModel` brackets maker fills between an optimistic bound
(alone at the level) and a pessimistic one (behind the resting queue); both
are reported and the paper runner books the pessimistic one.

Two more that keep a bot alive and honest:

- **The veto imports nothing.** `risk.py` imports nothing from the package,
  takes plain strings and Decimals, and returns every reason. Gates: max
  contracts per market, max open notional, no new risk inside the resolution
  window or after it, a daily loss stop, an open-order count. An order that
  strictly shrinks a position takes a short path, because a veto that stops
  you getting out is a deadlock.
- **Loops cannot die or go silent.** `supervise.supervise` restarts any loop
  that raises; `each_with_deadline` puts a timeout on every awaited message
  and raises `FeedStalled`, which the recorder answers by reconnecting.

## Layout

```
predkit/
  schema.py         Contract, Book, Trade, OrderIntent, Fill; every price is the price of YES, Decimal
  fees.py           c*p*(1-p) per venue and tier, dated, sourced, checked at import; unknown tiers absent
  rawlog.py         the raw archive: exclusive-create writer, torn-member-safe reader
  record.py         one recorder per market: archive first, parse second, deadline on every message
  record_series.py  rolling 5m/15m windows discovered ahead of time, one directory and contract.json each
  replay.py         rebuild YES books from the archive; maker fill model with optimistic/pessimistic bounds
  label.py          turn resolved, recorded windows into labelled backtest rows
  backtest.py       rows labelled after resolution from the venue's source; non-overlapping holds; shuffled control
  risk.py           the veto: imports nothing, Decimal, every reason; reduce-only short path
  runner.py         paper | live; one process per bot; pid lock; KILL file; geofence; keystore
  supervise.py      supervise() and each_with_deadline()
  geofence.py       declared-residency refusals (a floor from venue rules, not legal clearance)
  keys.py           Fernet + PBKDF2 keystore, decrypted in-process at signing time
  ledger.py         what this process sent and what came back; closes only its own
  venues/           kalshi.py (REST, RSA-PSS signing, WS or public polling)
                    polymarket.py (Gamma, CLOB, data API, WS; py-clob-client for orders)
                    binance_reference.py (public book ticker, a leader feed, never a label)
  strategies/       Protocols + registry; simple_maker/ and simple_taker/ examples
perpkit/            crypto perpetual futures; see its own section below
examples/
  backtest_example.py         offline, synthetic end-to-end predkit backtest
  perpkit_factor_example.py   offline, synthetic perpkit factor-panel run
tests/              hand-computed values; no network, no credentials (perpkit's in tests/perpkit/)
```

## Install

Python 3.12 or newer. Use a fresh environment (venv, conda or micromamba).
One distribution, `stratbench`, ships both packages (`import predkit`,
`import perpkit`):

```
pip install stratbench                              # predkit: httpx, websockets, cryptography, psutil
pip install "stratbench[perps]"                     # + perpkit's analysis dependencies (numpy)
pip install "stratbench[polymarket]"                # only for live Polymarket orders (py-clob-client)
```

From a clone of https://github.com/P1ayer-1/stratbench, for development:

```
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[perps,dev]"                       # both toolkits + pytest
python -m pytest -q                                 # no network, no keys needed
```

The full suite takes a few minutes because the archive tests spawn and kill
real processes; `python -m pytest --deselect tests/test_rawlog.py` is the
fast subset.

Dependencies are real packages: `httpx` for REST, `websockets` for streams,
`cryptography` for the keystore and Kalshi's RSA-PSS signatures, `psutil`
for the runner's pid lock (`os.kill(pid, 0)` terminates the process on
Windows rather than probing it), and optionally `py-clob-client` for
Polymarket's EIP-712 orders.

## Quickstart

Everything below except step 1 works offline. Recording and all public
market data need no account and no key.

### 1. Record

```
# One market (a Kalshi ticker or a Polymarket conditionId), plus a leader feed
python -m predkit.record --venue kalshi --markets <TICKER> --reference binance:BTCUSDT
python -m predkit.record --venue polymarket --markets <conditionId>

# Rolling 5- and 15-minute windows, each discovered ahead of time into its own directory
python -m predkit.record_series --series polymarket:btc-5m,polymarket:btc-15m,kalshi:KXBTC15M --reference binance:BTCUSDT
```

Each market gets `data/<venue>/<market_id>/` with a `contract.json` (token
ids, tick, resolution time and source) and a `raw/` archive. Kalshi's
websocket needs a signed handshake even for public channels, so without a
key the Kalshi recorder polls the public book once a second (channel
`book_poll`); that is coarse, and fine for a first look.

Leave it running. Stop with Ctrl+C; a restart never overwrites what is on
disk.

### 2. Replay

```
python -m predkit.replay data/polymarket/<conditionId> --every 10
```

prints the rebuilt YES touch every 10 seconds of recorded time, flags a
crossed book, and counts duplicate frames dropped. In code:

```python
from pathlib import Path
from predkit.record_series import read_contract
from predkit.replay import rebuild

d = Path("data/polymarket/<conditionId>")
for t_ms, (kind, event) in rebuild(d, read_contract(d)):
    if kind == "book":
        print(t_ms, event.best_bid, event.best_ask)   # a live, mutable state: copy what you keep
```

### 3. Label and backtest

Once windows have resolved, label them from the venue's own result:

```
python -m predkit.label --data-dir data --offsets 240,120,60,30
```

This writes `data/rows/<venue>.jsonl`: one row per window per offset, with
the touch at that many seconds before resolution. The rows carry no
`signal`: that is your model's probability of YES, which you add before
backtesting a strategy that needs one:

```
python -m predkit.backtest --rows data/rows/kalshi.jsonl --strategy simple_taker
```

To see the whole path without recording anything first, run the offline
example. It builds synthetic rows through the same `label_row` guard,
attaches a toy model's signal, and scores `simple_taker` net of the Kalshi
fee against 200 shuffled-label controls:

```
python examples/backtest_example.py
```

The toy world is rigged so the model knows something the book does not,
so the result beats its shuffles. On real data, a result that does not beat
its shuffles is not an edge. Two caveats on the control: it assumes the
book is roughly fair against the pooled base rate, so on rows spread from
5c to 95c a strategy that simply buys the cheap side can "beat" it; and a
`decide` that assumes a maker fill is the optimistic bound; replay with
`MakerFillModel` for the honest one.

### 4. Paper trade

```
python -m predkit.runner --strategy simple_maker --venue polymarket --market <conditionId>
```

Paper is the default and sends nothing: intents are logged and filled on
paper by the replay fill model from the live tape, pessimistic bound booked
into risk. A run log goes to `data/<venue>/<market>/runs/`. Create
`data/<venue>/<market>/KILL` to trip the risk engine, cancel this process's
own orders (never anyone else's) and stop.

The example strategies need a fair value, which the runner does not invent:
wire your model into `Runner(fair=...)` or `Runner(reference=...)` in code
(see `tests/test_runner.py`). Without one, each plan refuses with the reason.

## Live trading

Nothing defaults to live. Live is refused unless **all** of these hold, and
every missing piece is listed at once:

- `--live` **and** `--confirm` (asked for twice, by name);
- `--residency <ISO country code>` that `geofence.py` accepts for that venue;
- a signer built from **your own** key (`--key-name`), so the adapter can sign;
- and every order still passes the risk veto (`--max-contracts`,
  `--max-notional`, `--max-daily-loss`).

```
python -m predkit.runner --strategy simple_taker --venue kalshi --market <TICKER> \
    --residency <CC> --live --confirm --key-name kalshi
```

**The geofence** is a declared-residency check built from published venue
rules and regulation, with sources in `predkit/geofence.py`:

| Residency | Venue | Rule |
|---|---|---|
| Canada | all | CSA Multilateral Instrument 91-102 prohibits offering binary options with a term under 30 days to individuals; every market here is under 30 days |
| United States | Polymarket (global) | the global exchange geoblocks US persons (CFTC order, 2022); US residents use Kalshi or Polymarket US (no adapter here) |

It is a floor, not a whitelist. Both venues publish longer restricted
lists, and local law can be stricter than any venue's list. Passing the
check does not make trading lawful for you. It is also not an IP geofence,
and it must not be circumvented.

**Keys** stay on your machine, encrypted, and are decrypted in-process only
when the signer needs them:

```
python -m predkit.keys add kalshi --key-id <your API key id> --pem-file kalshi.pem
python -m predkit.keys add polymarket --secret-env MY_POLYMARKET_PRIVATE_KEY   # or omit for a hidden prompt
python -m predkit.keys list
```

The keystore (`~/.predkit/keys.json` by default, `--keystore` to move it)
holds a random salt and a Fernet token per key, with the key derived from
your passphrase by PBKDF2-HMAC-SHA256 at 600,000 iterations. The passphrase
comes from `PREDKIT_PASSPHRASE` or a prompt. Secrets are never taken as
command-line arguments. Use trading-only API keys; neither venue's trading
key needs withdrawal permission, and this toolkit never asks for one.

One process per (venue, market, strategy) is enforced by a pid lock in the
market's data directory.

### Environment variables

| Variable | Meaning |
|---|---|
| `PREDKIT_PASSPHRASE` | keystore passphrase (else prompted) |
| `PREDKIT_BUILDER_TAKER_BPS` / `PREDKIT_BUILDER_MAKER_BPS` | optional Polymarket builder fee charged in backtests and plans; default 0, capped at 100 / 50 bps at import |

## Adding a strategy

1. Create `predkit/strategies/<name>/` with four files, copying
   `simple_taker/` as a template:
   - `plan.py`: a `plan_*(context) -> Plan` function whose `Plan` has
     `market_id`, `reasons` (every failing gate), `warnings`, `intents` and
     an `ok` property. It must not import from `predkit.venues` or
     `predkit.strategies`, and must not call `place_order` or `cancel(`;
     the contract test greps for that.
   - `execute.py`: sends a plan's intents through `venue.place_order`, only
     when `dry_run=False`, records every order id in the `OwnLedger` before
     anything else, and returns an `ExecutionResult` (`market_id`,
     `dry_run`, `problems`, `order_ids`, `ok`).
   - `monitor.py`: reads `venue.open_orders` / `venue.fills`, reconciles
     them against the ledger, and returns a report with `alerts` and a
     `critical` bool. Read-only.
   - `__init__.py`: a dataclass with `name`, `plan`, `execute`, `monitor`
     and `decide(row) -> Optional[Decision]` (the same thresholds on a
     backtest row, with `role="maker"` or `"taker"` for the fee).
2. Register it: add `<name>` to `NAMES` and a branch in `load` in
   `predkit/strategies/__init__.py`.
3. Test it: `tests/test_strategy_contract.py` picks it up automatically;
   add hand-computed tests of its gates and arithmetic beside
   `tests/test_strategies.py` (write the fee arithmetic out in a comment).
4. Backtest it with `python -m predkit.backtest --strategy <name>`, then
   replay it with `MakerFillModel` if it rests orders, then paper trade it.

The `Context` a planner receives holds the contract, the book, the time,
your `fair` value and `reference` data, feed lag, and this process's own
position. It holds nothing that can send an order.

## Fee tables

Transcribed by hand in `predkit/fees.py`. **None has been verified against
a real fill** (`verified=False` on every row). Fees change: re-read the
venue's schedule before trusting a verdict, and add a new dated row rather
than editing an old one. Per-contract fee at price `p`:

| Venue / tier | Taker | Maker | Rounding | Transcribed | Source |
|---|---|---|---|---|---|
| kalshi / `default` | 0.07 p(1-p) | 0 | up to the cent per order | 2026-09-12 | Kalshi published fee schedule |
| kalshi / `quadratic_with_maker_fees` | 0.07 p(1-p) | 0.0175 p(1-p) | up to the cent per order | 2026-09-24 | Kalshi fee schedule PDF effective 2026-07-07; `fee_type` read from `/series` |
| kalshi / `quadratic` | 0.07 p(1-p) | 0 | up to the cent per order | 2026-09-25 | same PDF; `fee_type` read from `/series` on the "mention" series |
| polymarket / `crypto_fees_v2` | 0.07 p(1-p) | 0 (rebate 0.2 of taker fee, not modelled) | none | 2026-09-13 | Gamma `feeSchedule` on the 5/15-minute BTC markets |
| polymarket / `economics_fees` | 0.05 p(1-p) | 0 (rebate 0.25, not modelled) | none | 2026-09-13 | Gamma `feeSchedule` on a Fed market |
| polymarket / `weather_fees` | 0.05 p(1-p) | 0 (rebate 0.25, not modelled) | none | 2026-09-24 | Gamma `feeSchedule` on a temperature market |

Caveats, all stated in the source rows:

- Polymarket rows assume Gamma's `rate` field is the curve coefficient `c`.
  It matches the venue's documented 1.75% at 50c, but it is an assumption.
- Polymarket maker rebates are recorded, not credited, so maker results
  are slightly pessimistic there.
- Kalshi's cent rounding is per order, so many small orders pay more than
  one large one; the tables model that.
- Polymarket tiers are keyed on Gamma's `feeType` string, copied verbatim
  onto the contract. A market family nobody has transcribed fails at fee
  time by name, and so will yours until you add its row.
- The Kalshi adapter labels every market `default`; it does not yet read a
  series' `fee_type`, so set `fee_tier` yourself on maker-fee series.

At 50c the standard taker fee is 1.75 cents a contract, which is larger
than most apparent edges on these markets. That is why the backtester
charges it on every entry and why the planners refuse a tier they cannot
price.

## What is verified and what is not

- The parsers are tested against payload shapes read live from both
  venues' public endpoints (dates in the adapter docstrings). The order
  write paths (Kalshi's order body, Polymarket's attributed orders) were
  not exercised with a real key when written; a wrong field name is a 4xx,
  not a wrong order, but test with the smallest size first.
- Kalshi's websocket snapshot/delta shape was not observed live; both the
  dollar-string and the older cents shapes are parsed.
- Polymarket's 5-minute crypto markets resolve on a Chainlink 60-second
  TWAP (`resolutionSource` on the Gamma row). The Binance book ticker is a
  leader feed only and is refused as a label source.
- Replay of Polymarket windows drops byte-identical frames delivered twice
  by the recorder's redundant sockets, but a frame only the lagging socket
  delivered is still applied late. Filter crossed or locked touches before
  treating one as a price.
- The maker fill model is a bracket, not a truth. Only live fills narrow it.

## perpkit (crypto perpetual futures)

perpkit records, replays and backtests crypto perpetual futures. It is the
same philosophy as predkit applied to a leveraged, funded, liquidatable
instrument: archive every venue message verbatim before parsing it, rebuild
anything derived from that archive, price every verdict with transcribed
fees, and put a risk veto that imports nothing in front of any order.

What it covers:

- **Recorders**, each a supervised, reconnecting loop with a deadline on
  every message:
  - `perpkit.record`: BloFin order book (`books`, 200 levels), trades and
    funding per instrument over one websocket each, plus open interest by REST;
    optional labelled feature rows (order-book imbalance, order flow,
    microprice, volatility), written only after the label horizon has closed.
  - `perpkit.record_oi`: BloFin open interest and mark price by REST, added to
    a run that is already going.
  - `perpkit.record_hyperliquid`: Hyperliquid books, trades and asset
    contexts, plus a liquidation map read from public account positions
    (the exchange's own `liquidationPx`), under a REST weight budget.
  - `perpkit.record_liquidations`: Binance's all-market `!forceOrder@arr`
    stream or Bybit's `allLiquidation.<symbol>` topics, venue-wide.
- **The raw archive** (`perpkit/rawlog.py`): gzip JSON lines `{t, n, m}`
  per channel per hour; exclusive-create files (a restart writes `.r001`
  beside the hour's first file, never appends); a reader that recovers every
  gzip member of a file torn by a hard kill.
- **Replay and audits** (`perpkit/analysis/`): `replay` rebuilds feature
  rows from the archive through the same book, tape and feature engine as
  live, on the receive clock or the venue's own clock; `audit_raw` reports
  damaged hours; `recorder_lag` measures how far each socket ran behind the
  venue's stamps.
- **Panels and a factor harness**: `panel_blofin` and `panel_hyperliquid`
  build daily panels in one schema (`perpkit/analysis/panel.py`: rows close at
  00:00 UTC, funding for day D is what accrued during D). `factor_panel`
  scores cross-sectional factors as money: non-overlapping holds, cost on
  turnover, price and funding legs split, a one-day lag by default, and the
  MEAN of within-date shuffled controls plus the percentile the real book
  beat. `check_features` tests recorded features against forward moves net
  of cost, with purged splits.
- **Fees and risk**: `perpkit/fees.py` (the BloFin ladder, below) and
  `perpkit/risk.py` (liquidation price, sizing, hard limits, kill switch,
  a reduce-only path so the veto never blocks getting out).
- **A teaching example, no known edge**: a delta-neutral funding carry on
  BloFin (long spot, short perp) with `plan_carry` / `run_carry` /
  `monitor_carry` / `close_carry`, plus `analysis/funding_carry` (a screen)
  and `analysis/carry_backtest` (every historical entry, overlap-corrected).
  It is there to show the plan / execute / monitor split, leg ordering and
  unwinds on a real two-leg position, not because it makes money.

Hyperliquid support is **read-only**: public market data and public account
state. There is no Hyperliquid order path.

### Install

```
pip install "stratbench[perps]"          # or, from a clone: pip install -e ".[perps,dev]"
```

That is enough for the Hyperliquid and liquidation recorders, replay, the
audits, the Hyperliquid panel, the factor harness and the example. The
BloFin websocket recorder, the BloFin panel, the carry screens and the carry
order paths also need the BloFin Python SDK
([P1ayer-1/blofin-sdk-python](https://github.com/P1ayer-1/blofin-sdk-python),
Apache-2.0). It is not on PyPI, so it is not a declared dependency (PyPI
refuses packages that depend on a git URL); install it from GitHub:

```
pip install "blofin @ git+https://github.com/P1ayer-1/blofin-sdk-python@v0.1.1"
```

Nothing in `import perpkit` needs the SDK: the BloFin feed is loaded lazily,
and the tests that need it skip without it.

### Quickstart

Recording and all public market data need no account and no key. Data goes
under `./data` (or `PERPKIT_DATA_DIR`, or `--data-dir`).

**1. Record.**

```
# Hyperliquid: validate the coins live, then record them (no SDK needed)
python -m perpkit.record_hyperliquid --check --coins BTC,ETH
python -m perpkit.record_hyperliquid --coins BTC,ETH

# BloFin: disk arithmetic first, then book + trades + funding + open interest
python -m perpkit.record --list-cost --instruments BTC-USDT,ETH-USDT
python -m perpkit.record --instruments BTC-USDT,ETH-USDT

# Venue-wide liquidation prints
python -m perpkit.record_liquidations --check
python -m perpkit.record_liquidations --venue binance
```

BloFin data lands in `data/<INST-ID>/raw/<day>/<channel>-<HH>.jsonl.gz`,
Hyperliquid in `data/hyperliquid/<COIN>/`, liquidation feeds in
`data/<venue>/<channel>/`. Leave recorders running; stop with Ctrl+C. A
restart never overwrites what is on disk. Budget roughly 140 MB a day per
BloFin instrument (`--list-cost` prints the arithmetic).

**2. Replay and audit.**

```
python -m perpkit.analysis.audit_raw --data-dir data                     # which hours are damaged, if any
python -m perpkit.analysis.replay --instrument BTC-USDT --date 2026-01-01 --clock exchange
python -m perpkit.analysis.recorder_lag --date 2026-01-01 --hours      # how far behind the venue each socket ran
python -m perpkit.analysis.check_features --data-dir data --horizon 900
```

`replay` regenerates feature rows from the archive; anything you think of
later can be computed over all the history you recorded. `--clock exchange`
places events at the venue's own stamps rather than when this process
received them, which matters because a socket can run tens of seconds behind
the venue in bursts while staying open and in sequence.

**3. Backtest.**

The offline example builds a synthetic panel in a rigged world (persistent
funding per coin, random-walk prices) and scores `carry_7` and `mom_14` net
of cost with a one-day lag and 50 shuffled controls:

```
python examples/perpkit_factor_example.py
```

On real data, build a daily panel and score factors on it:

```
python -m perpkit.analysis.panel_hyperliquid --top 60
python -m perpkit.analysis.factor_panel --panel data/panel/hyperliquid-daily.csv \
    --hold-days 7 --top-frac 0.2 --lag 1 --control-seeds 50
```

Read `pct` (the share of shuffled controls the real book beat) and `alpha`
before the headline number, and expect `ctrl` to land near minus the cost.
A factor that does not beat its shuffles net of cost is not an edge. Panels
built from instruments listed today are survivorship-biased; momentum-shaped
results are flattered by the coins that are missing.

**4. The carry teaching example (demo account).** Needs the SDK and a BloFin
demo API key:

```
export BLOFIN_API_KEY=... BLOFIN_API_SECRET=... BLOFIN_API_PASSPHRASE=...
python -m perpkit.analysis.funding_carry --top 25                 # screen: does funding cover the round trip?
python -m perpkit.analysis.carry_backtest --hold-days 14 --top 25 # every historical entry
python -m perpkit.plan_carry --instrument <INST-ID> --notional 200
python -m perpkit.run_carry  --instrument <INST-ID> --notional 200             # rehearsal, sends nothing
python -m perpkit.run_carry  --instrument <INST-ID> --notional 200 --confirm   # demo orders
python -m perpkit.monitor_carry --instrument <INST-ID>
python -m perpkit.close_carry --instrument <INST-ID> --confirm
```

The carry tools need a spot fee row for your tier; only VIP 1 spot fees are
transcribed (below), so at the default VIP 0 they refuse and say so rather
than guess.

### Guardrails

Every perpkit order path goes through `perpkit/guardrails.py` before it
builds a client or reads a key:

- **Dry unless `--confirm`.** Without it, `run_carry` and `close_carry` build
  the plan from live prices and print every step they would send, and send
  nothing. The executor itself defaults to `dry_run=True`.
- **Demo unless `--production`.** Orders go to BloFin's demo-trading host.
- **`--production --confirm` is refused outright.** There is no override
  flag and no second step: perpkit ships no production order path. If you
  decide to trade live, that is a decision to make deliberately, outside this
  toolkit, after the same code has run end to end on demo. `--production`
  alone is allowed for the read-only tools (`plan_carry`, `monitor_carry`)
  and for rehearsals.
- **No default trades live.** Every recorder, replay and analysis tool is
  read-only; `plan_carry` and `monitor_carry` have no path to an order
  endpoint at all.
- **Keys come only from the environment** (`BLOFIN_API_KEY`,
  `BLOFIN_API_SECRET`, `BLOFIN_API_PASSPHRASE`), are passed straight to the
  SDK, and are never logged, written or accepted as command-line arguments.
  Use a demo key, and a trading-only key without withdrawal permission.
- **The risk veto** (`perpkit/risk.py`) imports nothing from the package,
  uses `Decimal`, returns every failing reason, trips a kill switch on
  realised AND mark-to-market losses or a position drifting toward
  liquidation, and lets a verified reduce-only order through so the veto
  never blocks getting out.
- **Tests need no network, no credentials and no SDK**
  (`tests/perpkit/test_perp_guardrails.py` pins the rules above).

### Fee tables (BloFin)

Transcribed by hand in `perpkit/fees.py`. **None has been verified against a
real fill.** An import-time check rejects a ladder that gets worse as the
tier improves or repeats a tier's rates. Tiers that could not be confirmed
are absent, not interpolated, and an unknown tier is an error. Fees change:
re-read BloFin's schedule before trusting a verdict. The default is VIP 0
(`BLOFIN_VIP_TIER` to change it, only once your account qualifies).

Perpetual futures, fraction of notional per side:

| VIP tier | Maker | Taker | Transcribed | Source |
|---|---|---|---|---|
| 0 | 0.0200% | 0.0600% | 2026-09-07 | BloFin published fee schedule (via quoted search results; the site refuses automated fetches), corroborated by an account fee display 2026-09-09 |
| 1 | 0.0060% | 0.0500% | 2026-09-07 | same |
| 2 | 0.0040% | 0.0450% | 2026-09-07 | same |
| 3 | 0.0020% | 0.0425% | 2026-09-09 | account fee display |
| 4 | not transcribed | not transcribed | | could not be confirmed; absent on purpose |
| 5 | 0.0000% | 0.0350% | 2026-09-07 | published schedule, corroborated 2026-09-09 |

Spot (for the carry example's spot leg):

| VIP tier | Maker | Taker | Transcribed | Source |
|---|---|---|---|---|
| 1 | 0.0350% | 0.0600% | 2026-09-09 | account fee display |

Round trips at VIP 0: 4 bps maker/maker, 8 bps maker/taker, 12 bps
taker/taker, before spread. There is no maker rebate at any tier, so a
passive round trip still pays adverse selection when the fee reaches zero.
The risk veto's edge gate assumes taker/taker. perpkit has no fee table for
Hyperliquid, Binance or Bybit: it records them, it does not trade them.

### What is verified and what is not (perpkit)

- `risk.liquidation_price` matched BloFin's own reported liquidation price
  to 0.057% on one small demo position (MMR 0.005, ~6 bps fee term). MMR is
  tiered by size; `margin_tiers` reads the real tier from the account's own
  host (demo and production tables differ). Re-check with
  `risk.compare_to_exchange` at your size.
- The carry order path has run end to end on BloFin's demo host. It has
  never sent a production order, and cannot.
- Parsers are tested against payload shapes read live from each venue's
  public endpoints; dates are in the module docstrings.
- The feature rows and labels are regenerable from the archive; the archive
  is not regenerable from anything. Never delete it to save disk.

## Research lab (Claude Code agents)

`.claude/agents/` holds a team of [Claude Code](https://docs.claude.com/en/docs/claude-code)
subagents that research strategies the way this toolkit backtests them: with
costs first and goalposts fixed before the data is seen. They cover binary
prediction markets and crypto perpetual futures (funding, basis, order flow;
the perps side uses perpkit and public venue data).

The pipeline, one folder per idea under `research/` (see
[research/BOARD.md](research/BOARD.md)):

1. `strategy-scout` finds ideas in papers, venue docs and practitioner
   write-ups and writes a sourced card.
2. `strategy-theorist` turns a card or a hunch into a falsifiable hypothesis:
   mechanism, counterparty, cost hurdle from the fee tables, a prediction that
   can fail, and a prior.
3. `test-planner` pre-registers the test: numeric kill criteria and the scoring
   code's hash frozen before any out-of-sample data is fetched, controls
   (shuffled-label mean), cost model, sample size, and which steps run locally
   or remotely.
4. `local-tester` builds the code through `lab-test-writer`,
   `lab-code-writer` and `lab-code-reviewer`, which work from narrow briefs
   and never see the hypothesis; the reviewer hunts look-ahead, overlapping
   holds, cost errors and tests that can't fail. It then runs the plan and
   scores it against the kill criteria.
5. `remote-tester` (optional) runs latency-sensitive or long steps on a box
   near the venue.
6. `strategy-lab` orchestrates, and records every verdict (dead, promising,
   needs-more-data) on the board, negative results included.

**Using it.** The agents ship in the repo, so they are available whenever you
run Claude Code here. Start the whole lab as the main session with
`claude --agent strategy-lab`, then say "run the lab" or give it a topic. You
can also call any agent on its own (e.g. ask for `strategy-theorist` on a
hunch). Run the lab as the main session: its `tools: Agent(...)` list is the
registry every nested agent draws from, so if you add an agent, list it there.

**Safety.** The agents never place live or real-money orders. They use paper
mode, or a venue's demo environment only where the repo supports one, and
anything touching real money, keys or credentials goes back to you. They
never read or log credentials. Code and runs need your approval; research and
planning don't. Check each venue's terms and your jurisdiction before acting
on any idea.

**Configuring.** `remote-tester` is a template: fill in `<REMOTE_HOST>`,
`<SSH_KEY_PATH>`, `<REMOTE_DIR>` and the other placeholders in
`.claude/agents/remote-tester.md` before using it; until then it refuses and
plans stay local. The `model:` (and `effort:`) line in each agent's
frontmatter is a choice, not a requirement; change it to suit your plan or
budget. On Windows you can add `PowerShell` to an agent's `tools:` list.

## Licence

MIT, see [LICENSE](LICENSE). Contributions are welcome; see
[CONTRIBUTING.md](CONTRIBUTING.md).
