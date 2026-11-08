"""Turning predictions into a retention campaign, measured in money.

A retention offer costs `offer_cost`. If it reaches a customer who would have
churned, a share `save_rate` of them stay, keeping their `customer_value`.
Offers sent to customers who were never going to leave are wasted.
"""

import numpy as np


def campaign_profit(y_true, targeted, value, offer_cost: float, save_rate: float) -> float:
    """Net money saved by contacting the `targeted` customers."""
    y_true, targeted, value = np.asarray(y_true), np.asarray(targeted, bool), np.asarray(value, float)
    saved = save_rate * value[targeted & (y_true == 1)].sum()
    return float(saved - offer_cost * targeted.sum())


def random_profit(y_true, n_targeted: int, value, offer_cost: float, save_rate: float) -> float:
    """Expected profit of contacting the same number of customers chosen at random."""
    y_true, value = np.asarray(y_true), np.asarray(value, float)
    if len(y_true) == 0:
        return 0.0
    share = n_targeted / len(y_true)
    return float(share * save_rate * value[y_true == 1].sum() - offer_cost * n_targeted)


def best_threshold(y_true, proba, value, offer_cost: float, save_rate: float) -> tuple[float, float]:
    """Probability cut-off that maximises campaign profit."""
    proba = np.asarray(proba)
    best_t, best_p = 0.5, -np.inf
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        p = campaign_profit(y_true, proba >= t, value, offer_cost, save_rate)
        if p > best_p:
            best_t, best_p = float(t), p
    return best_t, float(best_p)


def campaign_summary(y_true, proba, value, threshold: float, offer_cost: float, save_rate: float) -> dict:
    targeted = np.asarray(proba) >= threshold
    y_true = np.asarray(y_true)
    profit = campaign_profit(y_true, targeted, value, offer_cost, save_rate)
    baseline = random_profit(y_true, int(targeted.sum()), value, offer_cost, save_rate)
    return {
        "n_targeted": int(targeted.sum()),
        "churners_reached": int((targeted & (y_true == 1)).sum()),
        "campaign_profit": round(profit, 2),
        "random_profit": round(baseline, 2),
        "profit_vs_random": round(profit - baseline, 2),
    }
