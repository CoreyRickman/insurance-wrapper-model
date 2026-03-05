def monthly_rate_from_annual(r_annual: float) -> float:
    return (1 + r_annual) ** (1/12) - 1

def bps_to_rate(bps: float) -> float:
    return bps / 10_000.0

def apply_monthly_asset_fee(value: float, total_bps: float) -> float:
    return max(0.0, value * (1 - (total_bps / 10_000.0) / 12.0))
