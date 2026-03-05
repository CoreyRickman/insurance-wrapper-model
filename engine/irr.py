import numpy as np

def irr_monthly(cashflows):
    """
    cashflows: list/array of monthly cashflows starting at month 0.
      - month 0 should include initial investment as negative
      - subsequent months include withdrawals/loans as positive, taxes as negative if you model them separately,
        and terminal value as positive.
    Returns: (monthly_irr, annualized_irr)
    """
    cf = np.array(cashflows, dtype=float)
    if len(cf) < 2:
        return None, None
    # Need at least one negative and one positive
    if not (np.any(cf < 0) and np.any(cf > 0)):
        return None, None

    # Use numpy.irr-like approach via npf if available; implement robust solve:
    # Solve NPV(r)=0 with simple bracket + bisection.
    def npv(r):
        return np.sum(cf / ((1 + r) ** np.arange(len(cf))))

    # Bracket: start from -0.99 to high
    lo, hi = -0.99, 5.0
    f_lo, f_hi = npv(lo), npv(hi)
    if np.isnan(f_lo) or np.isnan(f_hi):
        return None, None

    # If no sign change, try expanding hi
    tries = 0
    while f_lo * f_hi > 0 and tries < 50:
        hi *= 1.5
        f_hi = npv(hi)
        tries += 1

    if f_lo * f_hi > 0:
        return None, None

    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < 1e-8:
            r_m = mid
            r_a = (1 + r_m) ** 12 - 1
            return r_m, r_a
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    r_m = (lo + hi) / 2
    r_a = (1 + r_m) ** 12 - 1
    return r_m, r_a
