import numpy as np
from .types import StrategyInputs
from .fees import bps_to_rate
from .returns import monthly_returns_deterministic, monthly_returns_monte_carlo

def make_monthly_return_path(si: StrategyInputs, months: int, mode: str = "deterministic", seed: int = 42) -> np.ndarray:
    net_annual = si.expected_return - bps_to_rate(si.mgmt_fee_bps)
    if mode == "monte_carlo":
        return monthly_returns_monte_carlo(net_annual, si.volatility, months, seed)
    return monthly_returns_deterministic(net_annual, months)

def annual_tax_buckets(si: StrategyInputs):
    """
    Planning-grade: tax buckets as % of starting NAV each year.
    """
    fee = bps_to_rate(si.mgmt_fee_bps)
    return {
        "ordinary": max(0.0, si.ordinary_yield),
        "qdiv": max(0.0, si.qualified_div_yield),
        "stcg": max(0.0, si.realized_stcg),
        "ltcg": max(0.0, si.realized_ltcg),
        "unrealized": max(0.0, si.unrealized_appreciation - fee),
        "fee_rate": fee,
    }
