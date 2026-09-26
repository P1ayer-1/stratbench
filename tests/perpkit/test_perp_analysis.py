"""Tests for the evaluation statistics.

An evaluation script that cannot fail is worthless — it will happily report
"promising" on pure noise. These tests check the primitives against known
answers, and `test_perp_check_features.py` checks the whole pipeline end to end on
data with a deliberately planted edge and on data with none.
"""

import numpy as np
import pytest

from perpkit.analysis.stats import (
    auc,
    correlation_tstat,
    decile_returns,
    effective_sample_size,
    fit_logistic,
    pearson,
    predict_proba,
    purged_split,
    rankdata,
    spearman,
    standardize,
)


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def test_pearson_perfect_correlation():
    x = np.array([1.0, 2, 3, 4, 5])
    assert pearson(x, 2 * x + 1) == pytest.approx(1.0)
    assert pearson(x, -x) == pytest.approx(-1.0)


def test_pearson_of_constant_is_zero_not_nan():
    x = np.array([1.0, 2, 3])
    assert pearson(x, np.array([5.0, 5, 5])) == 0.0


def test_pearson_ignores_non_finite_pairs():
    x = np.array([1.0, 2, 3, np.nan])
    y = np.array([2.0, 4, 6, 100])
    assert pearson(x, y) == pytest.approx(1.0)


def test_rankdata_averages_ties():
    # Two tied 2s occupy ranks 2 and 3 -> both get 2.5.
    assert list(rankdata(np.array([1.0, 2.0, 2.0, 4.0]))) == [1.0, 2.5, 2.5, 4.0]


def test_spearman_catches_monotonic_nonlinearity():
    """The reason to prefer Spearman: Pearson understates a monotonic but
    strongly curved relationship, which is common in book features."""
    x = np.linspace(0.1, 10, 200)
    y = np.exp(x)
    assert spearman(x, y) == pytest.approx(1.0, abs=1e-9)
    assert pearson(x, y) < 0.8


def test_spearman_is_robust_to_a_single_outlier():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = x + rng.normal(scale=0.1, size=500)
    x[0], y[0] = 1e6, -1e6  # one catastrophic print
    assert spearman(x, y) > 0.9
    assert pearson(x, y) < 0.5  # Pearson is wrecked by it


# ---------------------------------------------------------------------------
# Effective sample size — the correction that matters most
# ---------------------------------------------------------------------------


def test_overlapping_labels_shrink_effective_sample_size():
    # 250ms sampling, 5s horizon -> 20x overlap.
    assert effective_sample_size(100_000, horizon_seconds=5.0,
                                 sample_interval_seconds=0.25) == 5_000


def test_non_overlapping_labels_keep_full_sample_size():
    assert effective_sample_size(1000, horizon_seconds=1.0,
                                 sample_interval_seconds=1.0) == 1000
    # Sampling slower than the horizon cannot give MORE than n.
    assert effective_sample_size(1000, horizon_seconds=1.0,
                                 sample_interval_seconds=5.0) == 1000


def test_tstat_uses_effective_n_and_shrinks_accordingly():
    """The same correlation is far less significant once overlap is admitted."""
    naive = correlation_tstat(0.02, 100_000)
    honest = correlation_tstat(0.02, 5_000)
    assert naive > 6      # "highly significant"
    assert honest < 1.5   # ...and actually not significant at all
    assert honest < naive


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------


def test_auc_of_a_perfect_ranker_is_one():
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_auc_of_a_reversed_ranker_is_zero():
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)


def test_auc_of_random_scores_is_about_half():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 4000)
    assert auc(y, rng.normal(size=4000)) == pytest.approx(0.5, abs=0.03)


def test_auc_with_one_class_returns_half():
    assert auc(np.array([1, 1, 1]), np.array([0.1, 0.5, 0.9])) == 0.5


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def test_purged_split_is_ordered_and_leaves_a_gap():
    train, test = purged_split(1000, train_fraction=0.7, purge_rows=20)
    assert train == slice(0, 700)
    assert test == slice(720, 1000)
    # The gap is what stops training labels overlapping test features.
    assert test.start - train.stop == 20


def test_purged_split_never_overlaps():
    train, test = purged_split(100, train_fraction=0.9, purge_rows=50)
    assert test.start >= train.stop
    assert test.start <= 100


def test_standardize_uses_only_training_statistics():
    """Fitting the scaler on test data is a real, if subtle, leak."""
    train = np.array([[0.0], [2.0]])
    test = np.array([[100.0]])
    scaled_train, scaled_test = standardize(train, test)
    assert scaled_train.mean() == pytest.approx(0.0)
    # Test point is scaled by TRAIN stats, so it stays far away.
    assert scaled_test[0, 0] == pytest.approx(99.0)


def test_standardize_handles_constant_columns():
    train = np.array([[1.0, 5.0], [3.0, 5.0]])
    scaled, = standardize(train)
    assert np.isfinite(scaled).all()


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------


def test_logistic_recovers_a_known_relationship():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(4000, 2))
    # True model: driven by feature 0 only.
    probability = 1 / (1 + np.exp(-(1.5 * X[:, 0])))
    y = (rng.random(4000) < probability).astype(float)

    weights = fit_logistic(X, y, l2=0.01)
    assert weights[1] > 1.0          # feature 0 recovered, right sign
    assert abs(weights[2]) < 0.3     # feature 1 correctly near zero


def test_logistic_on_pure_noise_finds_nothing():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(3000, 3))
    y = rng.integers(0, 2, 3000).astype(float)
    weights = fit_logistic(X, y, l2=1.0)
    scores = predict_proba(X, weights)
    assert auc(y, scores) == pytest.approx(0.5, abs=0.05)


def test_logistic_probabilities_are_valid():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] > 0).astype(float)
    scores = predict_proba(X, fit_logistic(X, y))
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    assert np.isfinite(scores).all()


def test_logistic_survives_perfectly_separable_data():
    """Separable data sends unregularised logistic weights to infinity. The
    ridge penalty and variance floor must keep it finite."""
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    weights = fit_logistic(X, y, l2=1.0)
    assert np.isfinite(weights).all()


# ---------------------------------------------------------------------------
# Decile analysis
# ---------------------------------------------------------------------------


def test_decile_returns_are_monotonic_when_the_score_is_informative():
    scores = np.linspace(0, 1, 1000)
    forward = scores * 10  # perfectly informative
    deciles = decile_returns(scores, forward)
    assert all(a < b for a, b in zip(deciles, deciles[1:]))


def test_decile_returns_are_flat_for_an_uninformative_score():
    rng = np.random.default_rng(5)
    deciles = decile_returns(rng.normal(size=5000), rng.normal(size=5000))
    assert abs(pearson(np.arange(10, dtype=float), deciles)) < 0.7
