import numpy as np
from .types import StrategyInputs
from .fees import bps_to_rate

def deterministic_monthly_returns(si: StrategyInputs, months: int) -> np.ndarray:
    net_annual = si.expected_return - bps_to_rate(si.mgmt_fee_bps)
    r_m = (1 + net_annual) ** (1/12) - 1
    return np.full(months, r_m, dtype=float)

def annual_tax_buckets(si: StrategyInputs):
    fee = bps_to_rate(si.mgmt_fee_bps)
    return {
        "ordinary": max(0.0, si.ordinary_yield),
        "qdiv": max(0.0, si.qualified_div_yield),
        "stcg": max(0.0, si.realized_stcg),
        "ltcg": max(0.0, si.realized_ltcg),
        "unrealized": max(0.0, si.unrealized_appreciation - fee),
        "fee_rate": fee,
    }
