import pandas as pd
from .types import PolicyInputs, StrategyInputs, TaxInputs, TaxableInputs
from .strategy import annual_tax_buckets
from .taxes import eff_rate_ordinary, eff_rate_qdiv, eff_rate_stcg, eff_rate_ltcg
from .fees import bps_to_rate

def run_taxable(policy: PolicyInputs,
                strategy: StrategyInputs,
                taxes: TaxInputs,
                taxable: TaxableInputs,
                years: int,
                monthly_returns) -> pd.DataFrame:
    months = years * 12
    r = monthly_returns
    buckets = annual_tax_buckets(strategy)

    value = policy.premium * (1 - policy.premium_load)
    basis = policy.premium
    rows = []

    ord_r = eff_rate_ordinary(taxes)
    qd_r = eff_rate_qdiv(taxes)
    st_r = eff_rate_stcg(taxes)
    lt_r = eff_rate_ltcg(taxes)

    tlh_rate = bps_to_rate(taxable.tlh_bps)

    death_month = None
    if taxable.step_up_at_death and taxable.death_age is not None:
        death_month = int(max(1, (taxable.death_age - policy.issue_age) * 12))

    for m in range(1, months + 1):
        age = policy.issue_age + (m-1)/12.0
        v_start = value

        # grow by total return
        value = max(0.0, value * (1 + r[m-1]))
        tax = 0.0
        value_gross_end = value  # before paying annual taxes

        if m % 12 == 0:
            start_nav = v_start

            tax += start_nav * buckets["ordinary"] * ord_r
            tax += start_nav * buckets["qdiv"] * qd_r
            tax += start_nav * buckets["stcg"] * st_r
            tax += start_nav * buckets["ltcg"] * lt_r

            tax = max(0.0, tax - (start_nav * tlh_rate))
            value = max(0.0, value - tax)

            # Basis increases by *all* realized/distributed components assumed taxed annually
            basis += start_nav * (buckets["ordinary"] + buckets["qdiv"] + buckets["stcg"] + buckets["ltcg"])

        if death_month is not None and m == death_month:
            basis = value  # step-up

        rows.append({
            "month": m,
            "age": age,
            "value_start": v_start,
            "value_gross_end": value_gross_end,
            "value_end": value,
            "basis": basis,
            "tax_paid": tax,
        })

    return pd.DataFrame(rows)
