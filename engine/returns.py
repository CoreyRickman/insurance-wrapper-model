import numpy as np

def monthly_returns_deterministic(annual_return: float, months: int) -> np.ndarray:
    r_m = (1 + annual_return) ** (1/12) - 1
    return np.full(months, r_m, dtype=float)

def monthly_returns_monte_carlo(annual_return: float, annual_vol: float, months: int, seed: int) -> np.ndarray:
    """
    Planning-grade MC using normal simple returns per month.
    """
    rng = np.random.default_rng(seed)
    mu_m = (1 + annual_return) ** (1/12) - 1
    vol_m = annual_vol / np.sqrt(12)
    return rng.normal(loc=mu_m, scale=vol_m, size=months)
