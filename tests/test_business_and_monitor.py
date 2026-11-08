import numpy as np
import pandas as pd

from churn.business import best_threshold, campaign_profit, campaign_summary, random_profit
from churn.monitor import drift_summary, drift_table, psi


# --- business --------------------------------------------------------------

def test_campaign_profit_counts_saved_value_minus_offer_cost():
    y = [1, 0, 1, 0]
    targeted = [True, True, False, False]
    value = [1000, 500, 800, 300]
    # one churner reached: 0.3 * 1000 saved, two offers at 25
    assert campaign_profit(y, targeted, value, offer_cost=25, save_rate=0.3) == 300 - 50


def test_random_profit_is_proportional():
    y, value = [1, 0, 0, 0], [400, 100, 100, 100]
    # contacting 2 of 4 at random reaches half the churn value on average
    assert random_profit(y, 2, value, offer_cost=10, save_rate=0.5) == 0.5 * 0.5 * 400 - 20


def test_best_threshold_prefers_targeting_likely_churners():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    proba = np.clip(y * 0.7 + rng.random(500) * 0.3, 0, 1)   # churners score high
    value = np.full(500, 1000)
    t, profit = best_threshold(y, proba, value, offer_cost=25, save_rate=0.3)
    assert 0.05 <= t <= 0.95
    assert profit > 0


def test_campaign_summary_beats_random_with_a_good_model():
    y = np.array([1] * 20 + [0] * 80)
    proba = np.array([0.9] * 20 + [0.1] * 80)
    s = campaign_summary(y, proba, np.full(100, 500), 0.5, offer_cost=25, save_rate=0.3)
    assert s["n_targeted"] == 20 and s["churners_reached"] == 20
    assert s["campaign_profit"] > s["random_profit"]


# --- drift -----------------------------------------------------------------

def test_psi_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(1)
    assert psi(rng.normal(0, 1, 2000), rng.normal(0, 1, 2000)) < 0.05


def test_psi_is_large_for_shifted_distribution():
    rng = np.random.default_rng(1)
    assert psi(rng.normal(0, 1, 2000), rng.normal(1.5, 1, 2000)) > 0.2


def test_psi_handles_categories():
    assert psi([1] * 50 + [2] * 50, [1] * 50 + [2] * 50) < 0.01
    assert psi([1] * 90 + [2] * 10, [1] * 20 + [2] * 80) > 0.2


def test_drift_table_flags_only_the_shifted_feature():
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 1000), "b": rng.integers(1, 3, 1000)})
    cur = pd.DataFrame({"a": rng.normal(2, 1, 1000), "b": rng.integers(1, 3, 1000)})
    table = drift_table(ref, cur, ["a", "b"], psi_threshold=0.2)
    assert table.set_index("feature").loc["a", "drifted"]
    assert not table.set_index("feature").loc["b", "drifted"]
    summary = drift_summary(table, share_threshold=0.5)
    assert summary["drift_share"] == 0.5 and summary["data_drift"]
    assert summary["drifted_features"] == ["a"]
