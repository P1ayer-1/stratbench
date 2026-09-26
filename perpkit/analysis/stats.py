"""Statistics for evaluating whether features predict anything.

Small, dependency-light (numpy only) implementations of the handful of things
the predictiveness check needs. They live here rather than inline so they can
be unit-tested against known answers — a bug in the evaluation code is worse
than a bug in the model, because it makes you believe a false result.

The subtle one is `effective_sample_size`. Read its docstring before trusting
any t-statistic computed from overlapping labels.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Linear correlation, NaN-safe, 0.0 for a degenerate (constant) input."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    a, b = x[mask], y[mask]
    a_std, b_std = a.std(), b.std()
    if a_std == 0 or b_std == 0:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (a_std * b_std))


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, matching scipy's default tie handling.

    Ties matter here: order-book features are full of repeated values (a book
    that didn't change), and ordinal ranking would invent an ordering between
    identical observations.
    """
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)

    sorted_values = values[order]
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation — robust to the fat tails and outliers that dominate
    high-frequency return distributions, where Pearson can be driven almost
    entirely by a handful of extreme prints."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    return pearson(rankdata(x[mask]), rankdata(y[mask]))


def effective_sample_size(
    n_rows: int, horizon_seconds: float, sample_interval_seconds: float
) -> int:
    """Independent observations, given that adjacent labels OVERLAP.

    This is the correction that separates an honest evaluation from a
    self-deluding one.

    Sampling every 250ms with a 5s forward horizon means row *i* and row *i+1*
    predict windows that share 4.75 of their 5 seconds. They are very nearly
    the same observation. Twenty consecutive rows are, for statistical
    purposes, roughly one.

    So a "significant" t-statistic computed on N=350,000 is meaningless: the
    real N is closer to 17,500, and the standard error is ~4.5x larger than
    the naive calculation suggests. Ignoring this is the single most common
    way to convince yourself that noise is signal.

    Returns at least 1.
    """
    if sample_interval_seconds <= 0 or horizon_seconds <= 0:
        return max(1, n_rows)
    overlap = max(1.0, horizon_seconds / sample_interval_seconds)
    return max(1, int(n_rows / overlap))


def correlation_tstat(correlation: float, effective_n: int) -> float:
    """t = r * sqrt((n-2) / (1-r^2)), using the EFFECTIVE sample size."""
    if effective_n <= 2:
        return 0.0
    r2 = min(correlation**2, 0.999999)
    return float(correlation * np.sqrt((effective_n - 2) / (1 - r2)))


def auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, via the Mann-Whitney rank identity.

    0.5 = coin flip. For short-horizon price prediction, 0.52 is a plausible
    real edge and 0.60+ on out-of-sample data almost always means leakage.
    """
    y_true = np.asarray(y_true).astype(bool)
    positives, negatives = y_true.sum(), (~y_true).sum()
    if positives == 0 or negatives == 0:
        return 0.5
    ranks = rankdata(scores)
    return float((ranks[y_true].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def standardize(
    train: np.ndarray, *others: np.ndarray
) -> Tuple[np.ndarray, ...]:
    """Z-score using ONLY the training mean/std, applied to every split.

    Fitting the scaler on all the data before splitting leaks test-set
    information into training. It is a small leak, but it is exactly the kind
    that quietly inflates results.
    """
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return tuple((array - mean) / std for array in (train, *others))


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    iterations: int = 25,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Ridge-penalised logistic regression by IRLS (Newton's method).

    Deliberately a *linear* model. It is the right baseline precisely because
    it can only find linear structure: if it finds nothing, the honest
    conclusion is usually that there is little to find, not that the model was
    too simple. Reaching for LightGBM first hides that distinction.

    Returns the coefficient vector, with the intercept prepended.
    """
    n_samples, n_features = X.shape
    design = np.hstack([np.ones((n_samples, 1)), X])
    weights = np.zeros(n_features + 1)

    # Do not penalise the intercept — that would bias the base rate.
    penalty = np.eye(n_features + 1) * l2
    penalty[0, 0] = 0.0

    for _ in range(iterations):
        logits = np.clip(design @ weights, -35, 35)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        # Floor the variance so a confidently-separated point cannot make the
        # Hessian singular.
        variance = np.clip(probabilities * (1 - probabilities), 1e-8, None)

        gradient = design.T @ (y - probabilities) - penalty @ weights
        hessian = (design.T * variance) @ design + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        weights += step
        if np.abs(step).max() < tolerance:
            break

    return weights


def predict_proba(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.hstack([np.ones((X.shape[0], 1)), X])
    return 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -35, 35)))


def purged_split(
    n_rows: int,
    *,
    train_fraction: float = 0.7,
    purge_rows: int = 0,
) -> Tuple[slice, slice]:
    """Time-ordered train/test split with a PURGE GAP between them.

    Two rules, both non-negotiable for time-series:

    1. Never shuffle. A random split lets the model see the future during
       training, and produces a backtest that cannot be reproduced live.
    2. Drop `purge_rows` between train and test. The last training rows carry
       labels that extend forward *into* the test period; without a gap, their
       labels overlap the test features and leak. The gap must be at least the
       label horizon.

    (See López de Prado on purging and embargo.)
    """
    train_end = int(n_rows * train_fraction)
    test_start = min(n_rows, train_end + max(0, purge_rows))
    return slice(0, train_end), slice(test_start, n_rows)


def decile_returns(
    scores: np.ndarray, forward_returns: np.ndarray, deciles: int = 10
) -> np.ndarray:
    """Mean forward return within each score decile, lowest score first.

    A usable signal is monotonic across deciles: the more bullish the model,
    the better the realised return. A high top-decile number with no monotonic
    structure underneath is usually a handful of outliers, not an edge.
    """
    order = np.argsort(scores)
    buckets = np.array_split(order, deciles)
    return np.array([forward_returns[bucket].mean() for bucket in buckets])
