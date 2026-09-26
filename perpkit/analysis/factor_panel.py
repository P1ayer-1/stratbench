"""Cross-sectional factors on a daily panel, scored as money rather than as IC.

    python -m perpkit.analysis.factor_panel --panel data/panel/blofin-daily.csv
    python -m perpkit.analysis.factor_panel --panel P.csv --hold-days 14 --top-frac 0.2
    python -m perpkit.analysis.factor_panel --panel P.csv --factors carry_7,mom_30
    python -m perpkit.analysis.factor_panel --panel P.csv --external-factor my_signal=x.csv

Build a panel first with `perpkit.analysis.panel_blofin` or
`perpkit.analysis.panel_hyperliquid` (or any builder that writes the schema in
`perpkit/analysis/panel.py`), then point `--panel` at it.

What this measures, and why the bar is lower at a multi-day hold
----------------------------------------------------------------
Strategies that forecast PRICE over seconds to a day face a round trip that
is a large fraction of the move they predict, so they need an implausibly
high information coefficient to pay for themselves.

Hold for a week instead and the arithmetic inverts. A taker round trip on
BloFin at VIP 1 is ~10 bps; the cross-sectional dispersion of 7-day returns
across a hundred perpetuals is several hundred. So a long/short book needs an
IC around 0.02-0.03 to pay for itself. That is a lower bar, not a promise:
most factors still fail it once cost, lag and controls are honest.

Three rules this harness does not bend
--------------------------------------
**Rebalances do not overlap.** A 7-day hold rebalanced daily gives seven times
as many observations that are seven-sevenths the same trade. Every number here
comes from disjoint holding periods, so the count of rebalances IS the
effective N and no overlap correction is needed or offered. Five years at 7
days is 260 independent periods, not 1,825.

**The label is market-neutral and includes funding.** A long perp pays funding
when funding is positive, so the return to holding is
`log(close_out/close_in) - funding accrued`, and the label subtracts the
cross-section's own mean on that date. Measured against zero, market drift
masquerades as an edge; a dollar-neutral book cannot
earn the drift and must not be credited with it.

**The universe is point in time.** A symbol enters on a date only if, using
bars that closed at or before that date, it has `--min-history` complete days
and clears `--min-volume` median dollar volume over the trailing month. A
filter on full-sample liquidity would quietly select the names that got big.

The control, and what it is for
-------------------------------
Every factor is run beside a control that shuffles the factor's values across
symbols WITHIN each rebalance date. That keeps the number of positions, the
holding period, the universe, the turnover and the cost identical, and
destroys only the pairing between a symbol and its score. A factor that does
not beat this is not selecting symbols, whatever its t-statistic says.

**The control is reported as a distribution, not as a single number**, and that
is a correction rather than a preference. An earlier version reported the best
of five shuffles, which is a MAX statistic: with a standard error near 25 bps
the luckiest of five draws sits about 1.2 standard deviations up, so a control
of +35 was what noise looked like and got read once as the control beating the
factor. It misleads in both directions - a lucky draw can also flatter a weak
factor by making the bar look like one it cleared. So `ctrl` is now the MEAN
over `--control-seeds` shuffles and `pct` is the share of them the real book
beat, which is an empirical one-sided p-value and the number to read.

A useful property of the control mean: it lands near minus the cost, because a
shuffled book pays the same turnover and earns nothing. If it does not, the
cost model and the turnover disagree with each other.

What this cannot fix
--------------------
**Survivorship.** A panel built from the instruments listed TODAY omits every
coin that died before today. Cross-sectional demeaning removes the level
effect but not the selection: momentum-shaped factors are flattered, because
the names that kept falling until they delisted are the ones missing. Carry
is the least exposed, since it ranks on a cash flow rather than on past
price. If you can, build the panel from every perpetual that ever traded,
delisted ones included, and re-run anything with a price leg on it.

**Spreads are a constant.** `--cost-bps` is charged per unit of notional
traded, from the taker schedule in `perpkit/fees.py` (or `--spreads` per
symbol). The panel has no historical top of book, so the cost of trading a
thin name years ago is assumed to be the cost of trading it now. That is the
smaller assumption at a weekly hold; it is not small at a daily one.

**Lag.** `--lag` defaults to 1: each rebalance is scored with the factor as it
stood one day before the entry close, because a book cannot act at the
instant the bar that defines its signal closes. `--lag 0` is the optimistic
bound, never the result.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


from perpkit import config
from perpkit.analysis.stats import spearman

from perpkit.config import DATA_DIR
DEFAULT_PANEL = DATA_DIR / "panel" / "blofin-daily.csv"
MINUTES_PER_DAY = 1440


# ---------------------------------------------------------------------------
# The panel, as arrays
# ---------------------------------------------------------------------------


@dataclass
class Panel:
    """A dense `(date, symbol)` grid. NaN means "not listed / not complete"."""

    dates: List[str]
    symbols: List[str]
    close: np.ndarray             # (dates, symbols)
    volume: np.ndarray
    funding: np.ndarray           # bps accrued during that day
    rv: np.ndarray                # intraday realised vol, bps
    high: np.ndarray
    low: np.ndarray
    taker: np.ndarray
    complete: np.ndarray          # bool: the day has ~all its minutes
    # Optional: the SAME coins' funding on another venue, aligned to this grid.
    # Only ever a FEATURE input. The label must keep using this venue's own
    # funding, because that is what a position here actually pays - swapping it
    # would price the trade on an exchange it is not being made on.
    reference_funding: Optional[np.ndarray] = None

    @property
    def shape(self) -> Tuple[int, int]:
        return self.close.shape


def load_panel(path: Path, *, min_minutes: int = MINUTES_PER_DAY - 10) -> Panel:
    """Read a daily panel CSV (schema: `perpkit/analysis/panel.py`) into arrays.

    A day whose bar count falls short of `min_minutes` is marked incomplete
    rather than dropped, because a gap has to stay visible: a return computed
    across it spans more than a day and would otherwise be indistinguishable
    from one that does not.
    """
    if not path.exists():
        raise SystemExit(
            "No panel at " + str(path) + ".\n"
            "  Build one: python -m perpkit.analysis.panel_blofin "
            "(or panel_hyperliquid), then pass --panel.")

    by_key: Dict[Tuple[str, str], Dict[str, float]] = {}
    dates: Dict[str, None] = {}
    symbols: Dict[str, None] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            date, symbol = row["date"], row["symbol"]
            dates[date] = None
            symbols[symbol] = None
            by_key[(date, symbol)] = row

    date_list = sorted(dates)
    symbol_list = sorted(symbols)
    index_of_symbol = {symbol: i for i, symbol in enumerate(symbol_list)}
    shape = (len(date_list), len(symbol_list))

    def blank() -> np.ndarray:
        return np.full(shape, np.nan)

    close, volume, funding = blank(), blank(), blank()
    rv, high, low, taker = blank(), blank(), blank(), blank()
    complete = np.zeros(shape, dtype=bool)

    for d, date in enumerate(date_list):
        for symbol, s in index_of_symbol.items():
            row = by_key.get((date, symbol))
            if row is None:
                continue
            try:
                minutes = float(row["minutes"])
                close[d, s] = float(row["close"])
                volume[d, s] = float(row["quote_volume"])
                rv[d, s] = float(row["rv_bps"])
                high[d, s] = float(row["high"])
                low[d, s] = float(row["low"])
                taker[d, s] = float(row["taker_buy_frac"])
                periods = float(row["funding_periods"])
                funding[d, s] = float(row["funding_bps"]) if periods > 0 else np.nan
                complete[d, s] = minutes >= min_minutes
            except (TypeError, ValueError):
                continue

    return Panel(date_list, symbol_list, close, volume, funding, rv, high, low,
                 taker, complete)


def attach_reference_funding(panel: Panel, other: Panel) -> int:
    """Align another venue's funding onto this panel's grid, by canonical coin.

    Funding levels differ systematically between venues, which raises a
    different question from the one `carry_7` asks. `carry_7` ranks a coin by its funding
    against the rest of THIS venue's cross-section, so a venue-wide offset
    cancels out of it. The cross-venue version asks which coin is crowded
    HERE specifically - high funding relative to the same coin elsewhere - and
    that is a different signal, not a rescaling of the same one.

    Returns the number of coins matched. Coins the other venue does not carry
    are left NaN, so `carry_rel_*` simply does not score them rather than
    scoring them against a zero that would read as "not crowded".
    """
    from perpkit.analysis.panel import canonical

    def columns(target: Panel) -> Dict[str, int]:
        seen: Dict[str, int] = {}
        clashes = set()
        for index, symbol in enumerate(target.symbols):
            base = canonical(symbol)
            if base in seen:
                clashes.add(base)
            seen[base] = index
        for base in clashes:
            seen.pop(base, None)
        return seen

    mine, theirs = columns(panel), columns(other)
    other_row = {date: index for index, date in enumerate(other.dates)}
    reference = np.full(panel.close.shape, np.nan)
    matched = 0
    for base, column in mine.items():
        if base not in theirs:
            continue
        matched += 1
        source = theirs[base]
        for row, date in enumerate(panel.dates):
            index = other_row.get(date)
            if index is not None:
                reference[row, column] = other.funding[index, source]
    panel.reference_funding = reference
    return matched


# ---------------------------------------------------------------------------
# Features. Every one of these is closed at the end of day `d`.
# ---------------------------------------------------------------------------


def load_external_factor(path: Path, panel: Panel) -> Tuple[np.ndarray, int, int]:
    """A `(date, symbol, value)` CSV aligned onto this panel's grid.

    For a factor that cannot be computed from a price panel at all (say, one
    built from Hyperliquid's per-account fills). Adding it to
    `build_features`' dict rather than giving it its own evaluation path is
    the point: it then gets the same point-in-time universe, the same
    non-overlapping holds, the same turnover cost, the same within-date
    shuffles and the same report as every built-in factor, and its number is
    comparable with theirs.

    Symbols are matched through `panel.canonical()`, because the
    factor is keyed by Hyperliquid's coin names (`kBONK`) and the panel may be
    keyed by anybody's (`1000BONKUSDT`). Returns the grid with the count of
    rows matched and the count dropped, and the caller prints both - a factor
    that silently matched a tenth of its rows would score as "mostly NaN"
    rather than as broken.
    """
    from perpkit.analysis.panel import canonical

    if not Path(path).exists():
        raise SystemExit("No external factor at " + str(path))
    index_of_date = {date: i for i, date in enumerate(panel.dates)}
    index_of_symbol: Dict[str, int] = {}
    for i, symbol in enumerate(panel.symbols):
        index_of_symbol.setdefault(canonical(symbol), i)

    grid = np.full(panel.shape, np.nan)
    matched, dropped = 0, 0
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            d = index_of_date.get(str(row.get("date")))
            s = index_of_symbol.get(canonical(str(row.get("symbol") or "")))
            try:
                value = float(row["value"])
            except (KeyError, TypeError, ValueError):
                dropped += 1
                continue
            if d is None or s is None:
                dropped += 1
                continue
            grid[d, s] = value
            matched += 1
    return grid, matched, dropped


def net_price_leg(summary: Dict[str, float], cost_bps: float) -> float:
    """`price - turnover x cost / 2`: the price leg with its own cost charged.

    `net_bps` charges cost against gross, which for a carry factor is right -
    the funding leg is the result and it pays the same fees. For a factor
    whose claim is about PRICE, the funding leg is an artefact of whichever
    venue's row the panel happened to use, so the number that decides is the
    price leg alone, minus the whole cost. The halving is the harness's own
    convention: `net_bps` and `price_bps` are per unit of GROSS notional,
    which is two units of position, while `turnover` is not halved.
    """
    return float(summary["price_bps"] - summary["turnover"] * cost_bps / 2.0)


def _trailing_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Sum of the last `window` rows ending at each row, NaNs treated as 0.

    NaN-as-zero is right for funding (a day with no settlement accrued nothing)
    and is never applied to price, which goes through `_log_return` instead.
    """
    filled = np.nan_to_num(values, nan=0.0)
    out = np.full(values.shape, np.nan)
    cumulative = np.cumsum(filled, axis=0)
    out[window - 1:] = cumulative[window - 1:]
    out[window:] -= cumulative[:-window]
    return out


def _trailing_sum_nan(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing sum that propagates missing data instead of reading it as zero.

    `_trailing_sum` treats NaN as zero, which is right for funding (a day with
    no settlement accrued nothing) and wrong for a DIFFERENCE between venues: a
    day the other venue did not quote is unknown, not a day of zero crowding,
    and zero is the middle of this signal's range rather than one end of it.
    """
    out = np.full(values.shape, np.nan)
    for d in range(window - 1, values.shape[0]):
        block = values[d - window + 1:d + 1]
        if block.shape[0]:
            usable = np.isfinite(block).sum(axis=0)
            with np.errstate(invalid="ignore"):
                total = np.nansum(block, axis=0)
            out[d] = np.where(usable >= window - 1, total, np.nan)
    return out


def _log_return(close: np.ndarray, window: int) -> np.ndarray:
    """log(close_d / close_{d-window}), in bps."""
    out = np.full(close.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = close[window:] / close[:-window]
        out[window:] = np.log(np.where(ratio > 0, ratio, np.nan)) * 10_000.0
    return out


def _trailing_std(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    for d in range(window, values.shape[0]):
        block = values[d - window + 1:d + 1]
        with np.errstate(invalid="ignore"):
            out[d] = np.nanstd(block, axis=0)
    return out


def _trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    for d in range(window - 1, values.shape[0]):
        block = values[d - window + 1:d + 1]
        with np.errstate(invalid="ignore"):
            out[d] = np.nanmean(block, axis=0)
    return out


def build_features(panel: Panel) -> Dict[str, np.ndarray]:
    """Every candidate factor, as a `(dates, symbols)` array closed at day d.

    Sign convention: the factor is stated so that HIGH means "expected to
    outperform". Carry is therefore NEGATIVE funding - a perp whose longs are
    paying is one this book wants to be short - and short-horizon reversal is
    the negative of the trailing return. Stating the sign here rather than in
    the evaluation keeps a factor from being flipped after its result is seen,
    which is the cheapest way to manufacture an edge out of a coin flip.
    """
    close = panel.close
    daily_return = _log_return(close, 1)
    features: Dict[str, np.ndarray] = {}

    # Carry: trailing funding, negated. The one textbook factor whose return
    # is a cash flow rather than a forecast.
    for window in (1, 3, 7, 14, 30, 60):
        features["carry_" + str(window)] = -_trailing_sum(panel.funding, window) / window

    # Time-series momentum over weeks to months, the horizon the literature
    # documents; at minutes-to-hours crypto tends to reverse instead.
    for window in (7, 14, 30, 90):
        features["mom_" + str(window)] = _log_return(close, window)

    # Short-horizon reversal.
    for window in (1, 3):
        features["rev_" + str(window)] = -_log_return(close, window)

    # Low volatility: the cross-sectional anomaly that needs no direction
    # forecast at all.
    vol_30 = _trailing_std(daily_return, 30)
    features["lowvol_30"] = -vol_30
    features["lowrv_30"] = -_trailing_mean(panel.rv, 30)

    # Illiquidity (Amihud): |return| per dollar of volume. Small and thin pays
    # a premium in most asset classes; here it also costs the most to trade,
    # which is exactly what the cost model is for.
    with np.errstate(divide="ignore", invalid="ignore"):
        amihud = np.abs(daily_return) / np.where(panel.volume > 0, panel.volume, np.nan)
    features["illiq_30"] = _trailing_mean(amihud, 30)
    with np.errstate(divide="ignore", invalid="ignore"):
        features["small_30"] = -np.log(np.where(panel.volume > 0, panel.volume, np.nan))
    features["small_30"] = _trailing_mean(features["small_30"], 30)

    # Flow: the share of volume that lifted the offer, averaged over a week.
    features["taker_7"] = _trailing_mean(panel.taker, 7)

    # Crowding on THIS venue specifically, rather than this coin's carry: a
    # coin's funding here against the same coin's funding on a reference
    # venue. A venue-wide offset cancels out of `carry_*` (it ranks within one
    # venue), so this is a different signal, not a rescaling. Only built when
    # a reference venue has been attached.
    if panel.reference_funding is not None:
        difference = panel.funding - panel.reference_funding
        for window in (3, 7, 30):
            features["carry_rel_" + str(window)] = (
                -_trailing_sum_nan(difference, window) / window)

    # Distribution-shaped factors from the equity literature:
    #   max_7      the lottery effect: the coin with the biggest single-day
    #              jump in the last week underperforms, so NEGATIVE max.
    #   skew_30    the same story on the distribution: negative skewness.
    #   beta_30    betting against beta: negative beta to the cross-section.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # the first row is all NaN
        market = np.nanmean(daily_return, axis=1, keepdims=True)
    max_7 = np.full(close.shape, np.nan)
    skew_30 = np.full(close.shape, np.nan)
    beta_30 = np.full(close.shape, np.nan)
    for d in range(30, close.shape[0]):
        with np.errstate(invalid="ignore", divide="ignore"):
            max_7[d] = np.nanmax(daily_return[d - 6:d + 1], axis=0)
            block = daily_return[d - 29:d + 1]
            mean = np.nanmean(block, axis=0)
            sd = np.nanstd(block, axis=0)
            skew_30[d] = np.nanmean((block - mean) ** 3, axis=0) / np.where(sd > 0, sd ** 3, np.nan)
            m = market[d - 29:d + 1]
            m_dev = m - np.nanmean(m)
            cov = np.nanmean((block - mean) * m_dev, axis=0)
            beta_30[d] = cov / np.nanmean(m_dev * m_dev)
    features["max_7"] = -max_7
    features["skew_30"] = -skew_30
    features["beta_30"] = -beta_30

    # An example of combining two factors into ONE book. Blending the RETURNS
    # of two books assumes both are run and both are paid for; rank-averaging
    # the scores runs a single book, which nets the positions a symbol would
    # hold in both and pays the turnover once. Ranks rather than z-scores
    # because funding has a fat right tail and a z-score would let one extreme
    # name set the whole combination. The equal weights are illustrative, not
    # fitted: choose yours before looking at the result.
    features["carry_mom_even"] = _rank_blend(
        (features["carry_7"], 0.5), (features["mom_14"], 0.5))

    return features


def _rank_blend(*weighted: Tuple[np.ndarray, float]) -> np.ndarray:
    """Weighted average of within-date percentile ranks.

    Ranks are taken across the symbols present on each date, so a date with 30
    names and one with 80 contribute on the same 0..1 scale. Rows where any
    input is missing are NaN rather than imputed: a blend that silently falls
    back to one of its components on the dates where the other is unavailable
    is two different strategies sharing a name.
    """
    shape = weighted[0][0].shape
    out = np.full(shape, np.nan)
    for d in range(shape[0]):
        usable = np.ones(shape[1], dtype=bool)
        for values, _ in weighted:
            usable &= np.isfinite(values[d])
        index = np.flatnonzero(usable)
        if len(index) < 2:
            continue
        total = 0.0
        blended = np.zeros(len(index))
        for values, weight in weighted:
            order = np.argsort(np.argsort(values[d][index]))
            blended += weight * (order / (len(index) - 1.0))
            total += weight
        out[d, index] = blended / total
    return out


# ---------------------------------------------------------------------------
# Universe and labels
# ---------------------------------------------------------------------------


def tradeable(panel: Panel, *, min_history: int, min_volume: float,
              volume_window: int = 30) -> np.ndarray:
    """Point-in-time eligibility: `(dates, symbols)` bool.

    A symbol is eligible on day `d` if its last `min_history` days are all
    complete and its median dollar volume over the last `volume_window` days
    clears the bar. Both use only bars closed at or before `d`. A full-sample
    liquidity filter would select the names that later got big, which is the
    version of survivorship this file can actually avoid.
    """
    n_dates, n_symbols = panel.shape
    eligible = np.zeros((n_dates, n_symbols), dtype=bool)
    complete = panel.complete
    run = np.zeros(n_symbols, dtype=int)
    for d in range(n_dates):
        run = np.where(complete[d], run + 1, 0)
        if d < volume_window:
            continue
        window = panel.volume[d - volume_window + 1:d + 1]
        with np.errstate(invalid="ignore"):
            median_volume = np.nanmedian(window, axis=0)
        eligible[d] = (run >= min_history) & (median_volume >= min_volume)
    return eligible


def holding_legs(panel: Panel, entry: int, exit_index: int,
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """`(price_bps, funding_bps)` for a LONG perp held close-to-close.

    Kept separate because for the carry family the split IS the result. A carry
    book whose return comes from the funding leg is collecting a cash flow,
    which is the one kind of return a funding factor can claim by
    construction. One whose return comes from the price leg is forecasting
    price off a funding signal, and it should be read with that much less
    confidence.

    Funding is summed over days `entry+1 .. exit_index` because day `entry`'s
    funding had already accrued when the position was opened at its close. Off
    by one day here is a free lunch of one day's carry in whichever direction
    flatters the factor being tested.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = panel.close[exit_index] / panel.close[entry]
        price = np.log(np.where(ratio > 0, ratio, np.nan)) * 10_000.0
    funding = np.nansum(np.nan_to_num(panel.funding[entry + 1:exit_index + 1], nan=0.0),
                        axis=0)
    return price, -funding


def holding_return(panel: Panel, entry: int, exit_index: int) -> np.ndarray:
    """Total return in bps to a LONG perp: price move minus funding paid."""
    price, funding = holding_legs(panel, entry, exit_index)
    return price + funding


# ---------------------------------------------------------------------------
# The portfolio
# ---------------------------------------------------------------------------


@dataclass
class Rebalance:
    date: str
    entry: int
    exit_index: int
    weights: np.ndarray           # signed, sum|w| == 2 (1 long, 1 short)
    gross_bps: float
    turnover: float
    net_bps: float
    market_bps: float             # the eligible cross-section's own mean return
    price_bps: float              # the part of gross that came from price
    funding_bps: float            # the part that came from funding collected
    n_long: int
    n_short: int


@dataclass
class Result:
    name: str
    periods: List[Rebalance] = field(default_factory=list)
    ic: List[float] = field(default_factory=list)

    @property
    def net(self) -> np.ndarray:
        return np.array([p.net_bps for p in self.periods])

    @property
    def gross(self) -> np.ndarray:
        return np.array([p.gross_bps for p in self.periods])

    @property
    def market(self) -> np.ndarray:
        return np.array([p.market_bps for p in self.periods])

    @property
    def mean_ic(self) -> float:
        return float(np.mean(self.ic)) if self.ic else float("nan")

    def summary(self, periods_per_year: float) -> Dict[str, float]:
        net = self.net
        if len(net) < 2:
            return {}
        mean = float(net.mean())
        std = float(net.std(ddof=1))
        sharpe = (mean / std * math.sqrt(periods_per_year)) if std > 0 else 0.0
        stderr = std / math.sqrt(len(net))

        # A dollar-neutral book is not a beta-neutral book. Sorting on anything
        # vol-adjacent - low vol, illiquidity, a range position - puts high-beta
        # names on one side, and over five years of crypto the market's own move
        # is far larger than any factor return. `beta` is how much market the
        # book is carrying and `alpha_bps` is what is left once it is paid for;
        # a factor whose net is large and whose alpha is not was renting the
        # market, and the shuffled control cannot see that because a shuffled
        # book has no systematic tilt to shuffle away.
        market = self.market
        variance = float(market.var(ddof=1))
        beta = (float(np.cov(net, market, ddof=1)[0, 1] / variance)
                if variance > 0 else 0.0)
        alpha = mean - beta * float(market.mean())
        residual = net - beta * market
        alpha_stderr = float(residual.std(ddof=1)) / math.sqrt(len(net))

        return {
            "periods": float(len(net)),
            "gross_bps": float(self.gross.mean()),
            "net_bps": mean,
            "stderr": stderr,
            "t": mean / stderr if stderr > 0 else 0.0,
            "lo": mean - 1.96 * stderr,
            "hi": mean + 1.96 * stderr,
            "sharpe": sharpe,
            "annual_pct": mean * periods_per_year / 100.0,
            "hit": float((net > 0).mean()),
            "turnover": float(np.mean([p.turnover for p in self.periods])),
            "ic": self.mean_ic,
            "beta": beta,
            "alpha_bps": alpha,
            "alpha_t": alpha / alpha_stderr if alpha_stderr > 0 else 0.0,
            "price_bps": float(np.mean([p.price_bps for p in self.periods])),
            "funding_bps": float(np.mean([p.funding_bps for p in self.periods])),
        }

    def legs_by_year(self) -> Dict[str, Tuple[int, float, float, float]]:
        """`{year: (periods, net, price leg, funding leg)}`.

        The split is what separates a cash flow from a forecast, and the split
        can move even when the total does not: a carry book whose funding leg
        is steady across five years and whose price leg is one good year is two
        different strategies averaged together.
        """
        buckets: Dict[str, List[Tuple[float, float, float]]] = {}
        for period in self.periods:
            buckets.setdefault(period.date[:4], []).append(
                (period.net_bps, period.price_bps, period.funding_bps))
        out = {}
        for year, values in sorted(buckets.items()):
            array = np.array(values)
            out[year] = (len(values), float(array[:, 0].mean()),
                         float(array[:, 1].mean()), float(array[:, 2].mean()))
        return out

    def by_year(self) -> Dict[str, Tuple[int, float]]:
        """`{year: (periods, mean net bps)}`.

        A factor that earned everything in one year is a regime, not a factor,
        and the mean over five years cannot say which it is. This is the cheapest
        check that separates them and the one most often left out.
        """
        buckets: Dict[str, List[float]] = {}
        for period in self.periods:
            buckets.setdefault(period.date[:4], []).append(period.net_bps)
        return {year: (len(values), float(np.mean(values)))
                for year, values in sorted(buckets.items())}


def correlation_clusters(returns: np.ndarray, *, threshold: float = 0.75,
                         ) -> np.ndarray:
    """Group columns whose trailing returns move together. Point in time.

    The worst week this strategy had was not three bad positions, it was ONE
    bad position held three times: short SHIB, PEPE and BONK in February 2024,
    when all three roughly doubled together. Inverse-volatility sizing made it
    worse rather than better, because it sizes on trailing volatility and those
    three were quiet right up until they were not.

    Clustering is the control that does not depend on having estimated the risk
    correctly. This is deliberately the crudest version that works - a
    single-linkage pass at a correlation threshold, no hierarchy, no fitted
    number of clusters - because anything with parameters to tune would need
    its own out-of-sample evidence before it could be used to produce any.

    **Returns are demeaned across the cross-section first, and without that
    step this function does nothing useful.** On raw returns a 0.75 threshold
    can put nearly every name in ONE cluster (61 of 64 on one sample), because
    every crypto correlates through its beta to the market and single linkage
    chains A-B-C through it. Weighting on that hands almost the whole book to
    whichever few names happened not to chain. Demeaning removes the common
    factor, so what is left is names that move together BEYOND their beta,
    such as a sector rallying as one.

    `returns` is `(days, symbols)` of trailing daily returns. Columns with too
    little data are left in clusters of their own, which is the conservative
    reading: unknown correlation is treated as zero rather than as one.
    """
    with np.errstate(invalid="ignore"):
        market = np.nanmean(returns, axis=1, keepdims=True)
    returns = returns - market
    n_symbols = returns.shape[1]
    labels = np.arange(n_symbols)
    usable = np.isfinite(returns).sum(axis=0) >= 20
    index = np.flatnonzero(usable)
    if len(index) < 2:
        return labels

    block = returns[:, index]
    valid = np.all(np.isfinite(block), axis=1)
    if valid.sum() < 20:
        return labels
    with np.errstate(invalid="ignore"):
        matrix = np.corrcoef(block[valid].T)
    matrix = np.nan_to_num(matrix, nan=0.0)

    # Single linkage by union-find: i and j share a cluster if they correlate
    # above the threshold, transitively.
    parent = list(range(len(index)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i in range(len(index)):
        for j in range(i + 1, len(index)):
            if matrix[i, j] >= threshold:
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[root_j] = root_i

    for position, symbol in enumerate(index):
        labels[symbol] = n_symbols + find(position)
    return labels


def _weights(scores: np.ndarray, eligible: np.ndarray, top_frac: float,
             risk: Optional[np.ndarray] = None,
             clusters: Optional[np.ndarray] = None,
             previous: Optional[np.ndarray] = None,
             band: float = 0.0,
             ) -> Tuple[np.ndarray, int, int]:
    """Dollar-neutral weights: long the top, short the bottom, `sum|w| == 2`.

    `band` is a rebalance buffer: a name the book already holds on a side is
    KEPT as long as it still ranks inside the top `top_frac + band` for that
    side, and only names inside the top `top_frac` are newly opened. Each side
    still holds `n_side` names. A name that drifts from rank 28% to 34% and
    back is otherwise sold and re-bought for nothing, and at a 3-day hold
    that churn is a fifth of the gross return (turnover 1.21 x 10 bps / 2).

    With `risk` given, each position is sized by `1/risk` and each side is then
    renormalised to one dollar. Equal weight lets the most volatile name in the
    book dominate its variance - a meme perp at 15%/day sits beside a major at
    2%/day, so a tenth of the positions carries most of the risk and the book's
    Sharpe is set by whichever handful of alts happened to be ranked. Sizing by
    inverse volatility is the standard correction and it changes the RISK of
    the book, not its direction: the same names are held, in the same sign.

    Each side is renormalised separately, so the book stays dollar-neutral
    rather than drifting long the quiet names, which is how inverse-vol sizing
    usually acquires a beta by accident.
    """
    usable = eligible & np.isfinite(scores)
    if risk is not None:
        usable = usable & np.isfinite(risk) & (risk > 0)
    index = np.flatnonzero(usable)
    weights = np.zeros_like(scores)
    if len(index) < 6:
        return weights, 0, 0
    order = index[np.argsort(scores[index])]
    n_side = max(1, int(round(len(order) * top_frac)))
    if 2 * n_side > len(order):
        n_side = len(order) // 2
    shorts, longs = order[:n_side], order[-n_side:]
    if band > 0 and previous is not None:
        n_band = min(len(order) - n_side, max(n_side, int(round(len(order) * (top_frac + band)))))
        # Longs: keep held names still inside the wider band (best first),
        # then fill from the top of the ranking with names not yet held.
        band_longs = order[-n_band:][::-1]
        kept = [i for i in band_longs if previous[i] > 0][:n_side]
        fill = [i for i in order[::-1] if i not in kept and previous[i] <= 0]
        longs = np.array(kept + fill[:n_side - len(kept)])
        band_shorts = order[:n_band]
        kept_s = [i for i in band_shorts if previous[i] < 0 and i not in longs][:n_side]
        fill_s = [i for i in order if i not in kept_s and i not in longs and previous[i] >= 0]
        shorts = np.array(kept_s + fill_s[:n_side - len(kept_s)])

    for side, sign in ((longs, 1.0), (shorts, -1.0)):
        raw = (1.0 / risk[side]) if risk is not None else np.ones(len(side))
        if clusters is not None:
            # Divide each name's weight by how many of its cluster-mates are on
            # the same side, so a cluster gets one cluster's worth of money
            # however many of its members the ranking happened to pick. Three
            # memecoins that move as one then carry the risk of one position
            # rather than three.
            counts = np.array([float(np.sum(clusters[side] == clusters[member]))
                               for member in side])
            raw = raw / np.maximum(counts, 1.0)
        weights[side] = sign * raw / raw.sum()
    return weights, len(longs), len(shorts)


def load_spreads(path: Path, symbols: Sequence[str], *, taker_fee_bps: float,
                 default_spread_bps: float) -> np.ndarray:
    """Per-symbol cost per unit of notional traded: fee plus half the spread.

    `--cost-bps` charges every instrument the same, which is the assumption the
    live planner immediately contradicted: on BloFin, BTC quotes 0.01 bps and
    NEAR 16.9, and the carry book wants to SHORT the wide ones, because wide
    spreads and crowded longs live on the same instruments. A flat cost is
    therefore not a neutral simplification - it is one that flatters this
    particular strategy.

    A symbol with no measured spread gets `default_spread_bps` rather than the
    cheap end, and the count of those is reported: filling a gap with the
    median would quietly price the unmeasured names as typical when the reason
    they are missing is usually that they are small.
    """
    measured: Dict[str, float] = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    measured[row["symbol"]] = float(row["spread_bps"])
                except (KeyError, TypeError, ValueError):
                    continue
    costs = np.empty(len(symbols))
    missing = []
    for index, symbol in enumerate(symbols):
        spread = measured.get(symbol)
        if spread is None:
            spread = default_spread_bps
            missing.append(symbol)
        costs[index] = taker_fee_bps + spread / 2.0
    print("Per-instrument cost from {}: median {:.2f} bps, max {:.2f} bps"
          .format(path.name, float(np.median(costs)), float(costs.max())))
    if missing:
        print("  {} of {} symbols had no measured spread and were charged the "
              "{:.1f} bps default: {}".format(
                  len(missing), len(symbols), default_spread_bps,
                  ", ".join(sorted(missing)[:8])
                  + (" ..." if len(missing) > 8 else "")))
    return costs


def run_factor(panel: Panel, scores: np.ndarray, eligible: np.ndarray, *,
               hold_days: int, top_frac: float, cost_bps: float,
               start: int, shuffle_seed: Optional[int] = None,
               shuffle_mode: str = "row",
               risk: Optional[np.ndarray] = None,
               cost_per_symbol: Optional[np.ndarray] = None,
               cluster_lookback: int = 0,
               cluster_threshold: float = 0.75,
               lag: int = 0,
               band: float = 0.0) -> Result:
    """Hold a dollar-neutral book on non-overlapping `hold_days` periods.

    `lag` scores each rebalance with the factor as it stood `lag` rows EARLIER
    than the entry close, which is the honest version of a book that cannot
    act at the instant a bar closes: a one-day hold rebalanced from
    yesterday's ranking has to still pay, or the result is a claim about a
    trade nobody can place.

    `shuffle_seed` permutes scores across eligible symbols. ``row`` preserves
    the historical CLI behaviour (a fresh permutation each date). ``fixed``
    uses one symbol ordering for the whole run, preserving a signal's temporal
    persistence and approximately preserving its turnover. Prefer the
    latter for persistent signals: otherwise a persistent but useless signal beats its controls just
    because independently shuffled controls churn every rebalance.
    """
    result = Result(name="")
    rng = random.Random(shuffle_seed) if shuffle_seed is not None else None
    if shuffle_mode not in ("row", "fixed"):
        raise ValueError("shuffle_mode must be 'row' or 'fixed'")
    fixed_order = list(range(panel.shape[1]))
    if rng is not None and shuffle_mode == "fixed":
        rng.shuffle(fixed_order)
    previous = np.zeros(panel.shape[1])
    n_dates = panel.shape[0]

    for entry in range(start, n_dates - hold_days, hold_days):
        exit_index = entry + hold_days
        row = scores[max(entry - lag, 0)].copy()
        usable = eligible[entry] & np.isfinite(row)
        if rng is not None:
            index = np.flatnonzero(usable)
            values = row[index].tolist()
            if shuffle_mode == "row":
                rng.shuffle(values)
                row[index] = values
            else:
                # Restrict the run-wide ordering to today's usable set. When
                # the universe is unchanged, every symbol receives the same
                # donor time series on every date.
                targets = np.array([i for i in fixed_order if usable[i]], dtype=int)
                row[targets] = values

        clusters = None
        if cluster_lookback:
            # Correlations are measured on the days BEFORE entry only, so the
            # clustering cannot see the week it is sizing for.
            window = panel.close[max(0, entry - cluster_lookback):entry + 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = window[1:] / window[:-1]
                daily = np.log(np.where(ratio > 0, ratio, np.nan))
            clusters = correlation_clusters(daily, threshold=cluster_threshold)

        weights, n_long, n_short = _weights(
            row, eligible[entry], top_frac,
            risk[entry] if risk is not None else None, clusters,
            previous=previous, band=band)
        if n_long == 0:
            previous = np.zeros(panel.shape[1])
            continue

        price_leg, funding_leg = holding_legs(panel, entry, exit_index)
        forward = price_leg + funding_leg
        held = weights != 0
        if not np.all(np.isfinite(forward[held])):
            # A name that stopped printing mid-hold is not a return of zero.
            # Drop it from the book and renormalise what is left.
            bad = held & ~np.isfinite(forward)
            weights[bad] = 0.0
            # Renormalise each side on its own. Scaling the whole vector would
            # leave the book directional by exactly the size of whatever
            # vanished, and what vanishes is not a random name.
            long_side, short_side = weights > 0, weights < 0
            if not long_side.any() or not short_side.any():
                previous = np.zeros(panel.shape[1])
                continue
            # The short weights are negative and so is their sum: dividing by
            # its negation keeps them negative and makes them sum to -1. A
            # third line here used to multiply them by -1 again, so every
            # period in which a held name vanished ran LONG ON BOTH SIDES,
            # sum(w) = +2, and `excess` hid it. Rare on a survivors' panel,
            # common on a full universe, where vanishing mid-hold is what
            # delisted coins do.
            weights[long_side] /= weights[long_side].sum()
            weights[short_side] /= -weights[short_side].sum()
            held = weights != 0

        market = float(np.nanmean(forward[eligible[entry] & np.isfinite(forward)]))
        excess = forward - market
        gross = float(np.nansum(weights[held] * excess[held]))
        # The two legs sum to `gross` up to the market term, which sum(w)==0
        # removes; each is reported on the same per-unit-of-gross basis.
        price_part = float(np.nansum(weights[held] * np.nan_to_num(
            price_leg[held], nan=0.0)))
        funding_part = float(np.nansum(weights[held] * funding_leg[held]))

        traded = np.abs(weights - previous)
        turnover = float(traded.sum())
        if cost_per_symbol is not None:
            charged = float((traded * cost_per_symbol).sum())
        else:
            charged = turnover * cost_bps
        net = gross - charged
        previous = weights

        result.periods.append(Rebalance(
            date=panel.dates[entry], entry=entry, exit_index=exit_index,
            weights=weights, gross_bps=gross / 2.0, turnover=turnover,
            net_bps=net / 2.0, market_bps=market, price_bps=price_part / 2.0,
            funding_bps=funding_part / 2.0, n_long=n_long, n_short=n_short))

        ranked = eligible[entry] & np.isfinite(row) & np.isfinite(excess)
        if ranked.sum() >= 6:
            result.ic.append(spearman(row[ranked], excess[ranked]))

    return result


def block_bootstrap(values: np.ndarray, *, block: int = 4, draws: int = 2000,
                    seed: int = 7) -> Tuple[float, float]:
    """95% interval on the mean, in blocks, because factor returns cluster.

    Adjacent non-overlapping periods are not the same trade, but they do share
    a regime: a month in which the whole cross-section trends will hit several
    consecutive rebalances the same way. Blocks of four periods keep that
    clustering inside the resample instead of averaging it away.
    """
    n = len(values)
    if n < block * 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    means = np.empty(draws)
    for i in range(draws):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        sample = np.concatenate([values[s:s + block] for s in starts])[:n]
        means[i] = sample.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(rows: Sequence[Tuple[str, Dict[str, float], Dict[str, float],
                                      Tuple[float, float]]],
                 *, hold_days: int, cost_bps: float, top_frac: float,
                 universe: str) -> None:
    print()
    print("Cross-sectional long/short, " + str(hold_days) + "-day "
          "non-overlapping holds, top/bottom " + str(int(top_frac * 100)) + "%")
    print(universe)
    print("cost " + "{:.1f}".format(cost_bps) + " bps per unit of notional traded; "
          "returns are bps per period on gross notional")
    print()
    header = ("factor", "net", "95% block", "Sharpe", "ann%", "hit", "turn",
              "beta", "alpha", "price", "fund", "netprice", "ctrl", "pct")
    print("{:<14} {:>8} {:>18} {:>7} {:>7} {:>5} {:>5} {:>6} {:>7} "
          "{:>7} {:>6} {:>9} {:>8} {:>6}".format(*header))
    print("-" * 136)
    for name, real, control, interval in rows:
        if not real:
            continue
        print("{:<14} {:>8.1f} {:>18} {:>7.2f} {:>7.1f} {:>5.0%} "
              "{:>5.2f} {:>6.2f} {:>7.1f} {:>7.1f} {:>6.1f} {:>9.1f} "
              "{:>8.1f} {:>6.0%}".format(
                  name, real["net_bps"],
                  "[{:+.1f}, {:+.1f}]".format(interval[0], interval[1]),
                  real["sharpe"], real["annual_pct"], real["hit"],
                  real["turnover"], real["beta"],
                  real["alpha_bps"], real["price_bps"], real["funding_bps"],
                  net_price_leg(real, cost_bps),
                  control.get("net_bps", float("nan")),
                  control.get("percentile", float("nan"))))
    print()
    print("`price` and `fund` split gross into the price move and the funding "
          "collected. For the carry family that split is the result: a return "
          "from `fund` is a cash flow, one from `price` is a forecast.")
    print("`netprice` is `price - turn x cost / 2`: the price leg charged the "
          "whole cost, for a factor whose claim is about price and whose "
          "funding leg belongs to whichever venue priced the row.")
    print("`ctrl` is the MEAN of the shuffled controls and `pct` the share of "
          "them the real book beat - an empirical one-sided p-value. `ctrl` "
          "should land near minus the cost, since a shuffled book pays the "
          "same turnover and earns nothing.")


def print_years(rows: Sequence[Tuple[str, Dict[str, float], Dict[str, float],
                                     Tuple[float, float]]],
                results: Dict[str, "Result"], *, best: int = 6) -> None:
    """Net bps per period, by calendar year, for the factors that scored best.

    A five-year mean cannot distinguish a factor from a regime. Crypto's last
    five years contain a bull year, a collapse and two ranges, so a factor that
    earned its whole number in one of them is a bet on that regime returning.
    """
    ranked = sorted((row for row in rows if row[1]),
                    key=lambda row: row[1]["net_bps"], reverse=True)[:best]
    if not ranked:
        return
    years = sorted({year for name, _, _, _ in ranked
                    for year in results[name].by_year()})
    print()
    print("Net bps per period by year (the " + str(len(ranked))
          + " highest-scoring factors)")
    print("{:<14}".format("factor") + "".join("{:>12}".format(y) for y in years))
    print("-" * (14 + 12 * len(years)))
    for name, _, _, _ in ranked:
        table = results[name].by_year()
        cells = []
        for year in years:
            if year in table:
                count, mean = table[year]
                cells.append("{:>8.1f}({:d})".format(mean, count))
            else:
                cells.append("{:>12}".format("-"))
        print("{:<14}".format(name) + "".join("{:>12}".format(c) for c in cells))


def sweep(panel: Panel, features: Dict[str, np.ndarray], name: str, *,
          holds: Sequence[int], fracs: Sequence[float], costs: Sequence[float],
          volumes: Sequence[float], start: int, min_history: int,
          vol_scale: bool) -> None:
    """One factor over the whole specification grid, because one tuned point is
    not a result.

    By the time a factor has been looked at from four angles it has been fitted
    to the sample whether or not anything was deliberately optimised, and the
    honest defence is not another control - it is showing the surface. A real
    effect is positive across the grid and merely varies in size. A tuned one
    has a peak, and the peak is wherever the search stopped.

    The grid is deliberately coarse and deliberately includes settings that
    should make the factor WORSE - a cost six times the fee, a universe cut to
    the largest names - because a specification that only survives at its own
    best setting has not survived.
    """
    risk = -features["lowvol_30"] if vol_scale else None
    print()
    print("Specification sweep: " + name + ("  (vol-scaled)" if vol_scale else ""))
    print("net bps per period / annualised % / Sharpe; blank where the "
          "universe was too thin")
    print()
    for min_volume in volumes:
        eligible = tradeable(panel, min_history=min_history, min_volume=min_volume)
        median_names = int(np.median(eligible[start:].sum(axis=1)))
        print("  min volume ${:,.0f}/day   median {} eligible names"
              .format(min_volume, median_names))
        print("  {:<8}{:<8}".format("hold", "top%")
              + "".join("{:>22}".format("cost " + str(c) + " bps") for c in costs))
        for hold in holds:
            periods_per_year = 365.0 / hold
            for frac in fracs:
                cells = []
                for cost in costs:
                    result = run_factor(panel, features[name], eligible,
                                        hold_days=hold, top_frac=frac,
                                        cost_bps=cost, start=start, risk=risk)
                    summary = result.summary(periods_per_year)
                    cells.append("{:+7.1f} {:+6.1f}% {:5.2f}".format(
                        summary["net_bps"], summary["annual_pct"],
                        summary["sharpe"]) if summary else " " * 20)
                print("  {:<8}{:<8}".format(hold, int(frac * 100))
                      + "".join("{:>22}".format(c) for c in cells))
        print()


def universe_split(panel: Panel, features: Dict[str, np.ndarray], name: str, *,
                   eligible: np.ndarray, hold_days: int, top_frac: float,
                   cost_bps: float, start: int, risk: Optional[np.ndarray],
                   seeds: int = 6) -> None:
    """Run the factor on two disjoint halves of the SYMBOLS, several ways.

    Splitting by time only ever gives one held-out period, and a result can
    survive one split and dissolve on a bigger sample.
    Splitting the cross-section instead gives as many independent replications
    as there are ways to cut it: the two halves share every date, every regime
    and every market move, and differ only in which symbols carry the signal.
    A factor that is a property of the market appears in both halves. One that
    is a handful of lucky names appears in the half that holds them.
    """
    periods_per_year = 365.0 / hold_days
    print()
    print("Cross-section split: " + name + ", " + str(seeds)
          + " random halves of the symbol list")
    print("  {:<8}{:>12}{:>12}{:>10}{:>10}".format(
        "seed", "half A", "half B", "A Sharpe", "B Sharpe"))
    print("  " + "-" * 52)
    both_positive = 0
    for seed in range(seeds):
        rng = np.random.default_rng(1000 + seed)
        order = rng.permutation(len(panel.symbols))
        halves = (order[:len(order) // 2], order[len(order) // 2:])
        summaries = []
        for half in halves:
            mask = np.zeros(len(panel.symbols), dtype=bool)
            mask[half] = True
            result = run_factor(panel, features[name], eligible & mask,
                                hold_days=hold_days, top_frac=top_frac,
                                cost_bps=cost_bps, start=start, risk=risk)
            summaries.append(result.summary(periods_per_year))
        if not all(summaries):
            continue
        if summaries[0]["net_bps"] > 0 and summaries[1]["net_bps"] > 0:
            both_positive += 1
        print("  {:<8}{:>12.1f}{:>12.1f}{:>10.2f}{:>10.2f}".format(
            seed, summaries[0]["net_bps"], summaries[1]["net_bps"],
            summaries[0]["sharpe"], summaries[1]["sharpe"]))
    print("  both halves positive in {} of {} splits".format(both_positive, seeds))


def capacity(panel: Panel, result: "Result", *, participation: float,
             hold_days: int) -> None:
    """How large the book can be before it is trading against itself.

    Every number in this file is a RATE - bps per unit of gross notional - and
    a rate says nothing about how many units there are. A strategy that earns
    20% a year on $5,000 and cannot hold $500,000 is a hobby, and the panel
    already carries what is needed to tell the difference.

    The binding constraint is the smallest position, not the average one. A
    book is sized by its gross, each name takes a fixed share of that gross, and
    the name with the least volume relative to its weight is the one that caps
    the whole book: at gross G a name with weight w must trade `G * w` against
    its own daily volume. So the book's ceiling in a period is
    `min_i(participation * volume_i / w_i)`.

    `participation` is an assumption, not a measurement, and the honest way to
    use it is as a dial. A weekly rebalance can be worked over hours rather than
    sent at once, which is the one genuine advantage a multi-day strategy has
    over short-horizon strategies - patience is free here. 2% of a
    day's volume is conservative for that; 10% is not.
    """
    ceilings = []
    for period in result.periods:
        held = np.flatnonzero(period.weights != 0)
        volume = panel.volume[period.entry]
        limits = []
        for index in held:
            weight = abs(period.weights[index]) / 2.0     # share of GROSS
            if weight > 0 and np.isfinite(volume[index]) and volume[index] > 0:
                limits.append(participation * volume[index] / weight)
        if limits:
            ceilings.append(min(limits))
    if not ceilings:
        print("\nCapacity: no periods with usable volume.")
        return

    ceilings = np.array(ceilings)
    print()
    print("Capacity at {:.0%} of a day's volume per name, {} names a side"
          .format(participation,
                  int(np.median([p.n_long for p in result.periods]))))
    for label, value in (("worst period", ceilings.min()),
                         ("10th percentile", np.percentile(ceilings, 10)),
                         ("median", np.median(ceilings)),
                         ("most recent", ceilings[-1])):
        print("  {:<18} ${:>14,.0f} gross".format(label, value))
    net_per_period = float(result.net.mean())
    dollars = np.percentile(ceilings, 10) * net_per_period / 10_000.0
    print("  At the 10th-percentile size, {:+.1f} bps a period is "
          "${:+,.0f} per {} days, ${:+,.0f} a year.".format(
              net_per_period, dollars, hold_days,
              dollars * 365.0 / hold_days))
    print("  The binding name is the smallest position, so raising --top-frac "
          "spreads\n  the same gross over more names and RAISES capacity, "
          "while raising the\n  liquidity floor removes the names that cap it.")


def print_detail(result: "Result", *, periods_per_year: float) -> None:
    """One factor, year by year, with the price and funding legs separated."""
    print()
    print("Detail: " + result.name)
    print("{:<8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>8}".format(
        "year", "periods", "net", "price", "funding", "std", "Sharpe"))
    print("-" * 68)
    by_year_net: Dict[str, List[float]] = {}
    for period in result.periods:
        by_year_net.setdefault(period.date[:4], []).append(period.net_bps)
    for year, (count, net, price, funding) in result.legs_by_year().items():
        values = np.array(by_year_net[year])
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        sharpe = (net / std * math.sqrt(periods_per_year)) if std > 0 else float("nan")
        print("{:<8} {:>8d} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f} {:>8.2f}".format(
            year, count, net, price, funding, std, sharpe))
    net = result.net
    print("{:<8} {:>8d} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f} {:>8.2f}".format(
        "all", len(net), float(net.mean()),
        float(np.mean([p.price_bps for p in result.periods])),
        float(np.mean([p.funding_bps for p in result.periods])),
        float(net.std(ddof=1)),
        float(net.mean() / net.std(ddof=1) * math.sqrt(periods_per_year))))

    worst = sorted(result.periods, key=lambda p: p.net_bps)[:5]
    print()
    print("worst five periods: " + ", ".join(
        "{} {:+.0f}".format(p.date, p.net_bps) for p in worst))
    equity, peak, drawdown = 0.0, 0.0, 0.0
    for period in result.periods:
        equity += period.net_bps
        peak = max(peak, equity)
        drawdown = min(drawdown, equity - peak)
    print("cumulative {:+.0f} bps, worst drawdown {:.0f} bps "
          "(on gross notional, un-compounded)".format(equity, -drawdown))


def verdict(rows: Sequence[Tuple[str, Dict[str, float], Dict[str, float],
                                 Tuple[float, float]]]) -> None:
    print()
    survivors = [
        (name, real, control, interval) for name, real, control, interval in rows
        if real and interval[0] > 0.0 and control.get("percentile", 0.0) >= 0.95
    ]
    if not survivors:
        print("VERDICT: nothing clears. No factor has a bootstrap interval above "
              "zero net of cost.")
        best = max((r for _, r, _, _ in rows if r),
                   key=lambda r: r["net_bps"], default=None)
        if best:
            print("  best net was {:+.1f} bps per period.".format(best["net_bps"]))
        return

    survivors.sort(key=lambda item: item[3][0], reverse=True)
    print("CLEARS COST, interval above zero, and beats its own shuffled control:")
    for name, real, control, interval in survivors:
        print("  {:<14} {:+.1f} bps/period  [{:+.1f}, {:+.1f}]  "
              "Sharpe {:.2f}  {:+.1f}%/yr  beta {:+.2f}  alpha {:+.1f} "
              "(t {:.2f})  beat {:.0%} of shuffles".format(
                  name, real["net_bps"], interval[0], interval[1],
                  real["sharpe"], real["annual_pct"], real["beta"],
                  real["alpha_bps"], real["alpha_t"],
                  control.get("percentile", 0.0)))
    print()
    print(str(len(survivors)) + " of " + str(len(rows)) + " factors tested "
          "cleared, so read that as " + str(len(survivors)) + " draws in "
          + str(len(rows)) + ". The columns that decide it are `alpha` (net of "
          "the market the book is carrying) and `pct`, not the interval.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--hold-days", type=int, default=7)
    parser.add_argument("--top-frac", type=float, default=0.2)
    parser.add_argument("--cost-bps", type=float, default=None,
                        help="per unit of notional traded; default = config taker fee")
    parser.add_argument("--min-history", type=int, default=90)
    parser.add_argument("--min-volume", type=float, default=5e6,
                        help="median daily quote volume over the trailing month")
    parser.add_argument("--factors", help="comma-separated; default all")
    parser.add_argument("--control-seeds", type=int, default=50,
                        help="shuffles per factor; the report gives their "
                             "mean and the percentile the real book beat")
    parser.add_argument("--detail", help="comma-separated factors to break down by year")
    parser.add_argument("--reference-panel", type=Path, default=None,
                        help="another venue's panel; enables the carry_rel_* "
                             "factors, which rank a coin by its funding here "
                             "against the same coin's funding there")
    parser.add_argument("--cluster-lookback", type=int, default=0,
                        metavar="DAYS",
                        help="group names by trailing return correlation over "
                             "this many days and give each group one group's "
                             "worth of money; 0 disables")
    parser.add_argument("--cluster-threshold", type=float, default=0.75)
    parser.add_argument("--capacity", type=float, default=None,
                        metavar="PARTICIPATION",
                        help="report the book's size ceiling at this share of a "
                             "name's daily volume, e.g. 0.02")
    parser.add_argument("--spreads", type=Path, default=None,
                        help="CSV of measured per-symbol spreads; charges each "
                             "instrument its own fee + half spread instead of --cost-bps")
    parser.add_argument("--default-spread-bps", type=float, default=10.0,
                        help="charged to symbols absent from --spreads")
    parser.add_argument("--sweep", help="one factor over the whole specification grid")
    parser.add_argument("--split", help="one factor over random halves of the universe")
    parser.add_argument("--vol-scale", action="store_true",
                        help="size positions by 1/vol_30 instead of equally")
    parser.add_argument("--band", type=float, default=0.0,
                        help="rebalance buffer: keep a held name while it ranks inside "
                             "top_frac + band; open only inside top_frac")
    parser.add_argument("--lag", type=int, default=1,
                        help="score each rebalance with the factor from this many "
                             "days earlier (default 1; 0 is the optimistic bound)")
    parser.add_argument("--start", type=int, default=120,
                        help="skip this many leading days so features are warm")
    parser.add_argument("--external-factor", action="append", default=[],
                        metavar="NAME=PATH.CSV",
                        help="a (date, symbol, value) CSV scored beside the "
                             "built-in factors; repeatable")
    args = parser.parse_args(argv)

    cost_bps = args.cost_bps
    if cost_bps is None:
        cost_bps = float(config.TAKER_FEE_BPS)

    panel = load_panel(args.panel)
    if args.reference_panel:
        matched = attach_reference_funding(panel, load_panel(args.reference_panel))
        print("Reference funding from {}: {} of {} coins matched".format(
            args.reference_panel.name, matched, len(panel.symbols)))
    features = build_features(panel)
    for spec in args.external_factor:
        if "=" not in spec:
            raise SystemExit("--external-factor wants NAME=path.csv, got " + spec)
        name, _, location = spec.partition("=")
        name = name.strip()
        if name in features:
            raise SystemExit(
                "--external-factor " + name + " would shadow a built-in "
                "factor of the same name.\n  Rename it: the report's rows "
                "would otherwise be two different things under one label.")
        grid, matched, dropped = load_external_factor(Path(location), panel)
        features[name] = grid
        print("external factor {}: {:,} rows matched, {:,} dropped, "
              "{} coins, {} dates".format(
                  name, matched, dropped,
                  int((np.isfinite(grid).any(axis=0)).sum()),
                  int((np.isfinite(grid).any(axis=1)).sum())))
    names = ([n.strip() for n in args.factors.split(",") if n.strip()]
             if args.factors else sorted(features))
    missing = [n for n in names if n not in features]
    if missing:
        raise SystemExit("Unknown factors: " + ", ".join(missing)
                         + "\n  Available: " + ", ".join(sorted(features)))

    eligible = tradeable(panel, min_history=args.min_history,
                         min_volume=args.min_volume)

    if args.sweep:
        if args.sweep not in features:
            raise SystemExit("Unknown factor: " + args.sweep)
        sweep(panel, features, args.sweep, holds=(3, 7, 14, 30),
              fracs=(0.1, 0.2, 0.3), costs=(5.0, 10.0, 20.0, 30.0),
              volumes=(5e6, 50e6), start=args.start,
              min_history=args.min_history, vol_scale=args.vol_scale)
        return 0

    if args.split:
        if args.split not in features:
            raise SystemExit("Unknown factor: " + args.split)
        universe_split(panel, features, args.split, eligible=eligible,
                       hold_days=args.hold_days, top_frac=args.top_frac,
                       cost_bps=cost_bps, start=args.start,
                       risk=-features["lowvol_30"] if args.vol_scale else None)
        return 0
    counts = eligible[args.start:].sum(axis=1)
    universe = ("universe {} symbols, {} .. {}; eligible per rebalance "
                "min {} median {} max {}".format(
                    len(panel.symbols), panel.dates[args.start], panel.dates[-1],
                    int(counts.min()), int(np.median(counts)), int(counts.max())))
    periods_per_year = 365.0 / args.hold_days
    # `lowvol_30` is the negated trailing vol, so the risk measure is its
    # negation. Taken from the same feature so the sizing cannot see a day the
    # features cannot.
    risk = -features["lowvol_30"] if args.vol_scale else None
    cost_per_symbol = None
    if args.spreads:
        cost_per_symbol = load_spreads(
            args.spreads, panel.symbols, taker_fee_bps=cost_bps,
            default_spread_bps=args.default_spread_bps)

    rows = []
    results: Dict[str, Result] = {}
    for name in names:
        real = run_factor(panel, features[name], eligible, hold_days=args.hold_days,
                          top_frac=args.top_frac, cost_bps=cost_bps,
                          start=args.start, risk=risk,
                          cost_per_symbol=cost_per_symbol,
                       cluster_lookback=args.cluster_lookback,
                       cluster_threshold=args.cluster_threshold, lag=args.lag,
                       band=args.band)
        real.name = name
        results[name] = real
        controls = [
            run_factor(panel, features[name], eligible, hold_days=args.hold_days,
                       top_frac=args.top_frac, cost_bps=cost_bps, start=args.start,
                       shuffle_seed=seed, risk=risk,
                       cost_per_symbol=cost_per_symbol,
                       cluster_lookback=args.cluster_lookback,
                       cluster_threshold=args.cluster_threshold, lag=args.lag,
                       band=args.band)
            for seed in range(args.control_seeds)
        ]
        summaries = [c.summary(periods_per_year) for c in controls]
        summary = real.summary(periods_per_year)
        control_nets = np.array([s["net_bps"] for s in summaries if s])
        best_control: Dict[str, float] = {}
        if len(control_nets):
            best_control = {
                "net_bps": float(control_nets.mean()),
                "sd": float(control_nets.std(ddof=1)) if len(control_nets) > 1 else 0.0,
                "ic": float(np.mean([s["ic"] for s in summaries if s])),
                "percentile": (float((control_nets < summary["net_bps"]).mean())
                               if summary else float("nan")),
            }
        interval = (block_bootstrap(real.net) if summary
                    else (float("nan"), float("nan")))
        rows.append((name, summary, best_control, interval))

    print_report(rows, hold_days=args.hold_days, cost_bps=cost_bps,
                 top_frac=args.top_frac, universe=universe)
    print_years(rows, results)
    if args.capacity:
        for name in (args.detail.split(",") if args.detail else names[:1]):
            if name.strip() in results:
                capacity(panel, results[name.strip()],
                         participation=args.capacity, hold_days=args.hold_days)
    if args.detail:
        for name in args.detail.split(","):
            if name.strip() in results:
                print_detail(results[name.strip()], periods_per_year=periods_per_year)
    verdict(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
