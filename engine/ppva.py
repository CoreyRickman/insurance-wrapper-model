import pandas as pd
from .types import PolicyInputs, FeeInputs, TaxInputs, WithdrawalPlan, AnnuitizationPlan
from .fees import apply_monthly_asset_fee, monthly_rate_from_annual
from .taxes import eff_rate_ordinary

def lifo_withdraw(value: float, basis: float, amt: float):
    amt = max(0.0, min(amt, value))
    gain = max(0.0, value - basis)
    taxable = min(amt, gain)
    nontaxable = amt - taxable
    value2 = value - amt
    basis2 = max(0.0, basis - nontaxable)
    return value2, basis2, taxable, nontaxable

def run_ppva_accum(policy: PolicyInputs, fees: FeeInputs, years: int, monthly_returns) -> pd.DataFrame:
    months = years * 12
    r = monthly_returns

    value = policy.premium * (1 - policy.premium_load)
    basis = policy.premium
    admin_m = fees.admin_fee_annual / 12.0
    total_bps = fees.wrapper_fee_bps + fees.fund_er_bps

    rows = []
    for m in range(1, months + 1):
        age = policy.issue_age + (m-1)/12.0
        v_start = value
        value = max(0.0, value * (1 + r[m-1]))
        value = apply_monthly_asset_fee(value, total_bps)
        value = max(0.0, value - admin_m)

        rows.append({
            "month": m,
            "age": age,
            "value_start": v_start,
            "value_end": value,
            "basis": basis,
        })
    return pd.DataFrame(rows)

def apply_ppva_withdrawals(df: pd.DataFrame, plan: WithdrawalPlan, taxes: TaxInputs) -> pd.DataFrame:
    out = df.copy()
    out["withdraw_gross"] = 0.0
    out["withdraw_taxable"] = 0.0
    out["withdraw_nontaxable"] = 0.0
    out["tax"] = 0.0
    out["penalty"] = 0.0
    out["withdraw_net_cash"] = 0.0

    ord_r = eff_rate_ordinary(taxes)
    start_m = int(max(1, (plan.start_age - out["age"].iloc[0]) * 12 + 1))
    end_m = int(min(len(out), (plan.end_age - out["age"].iloc[0]) * 12 + 1))

    basis = float(out.loc[0, "basis"])

    for i in range(len(out)):
        m = int(out.loc[i, "month"])
        age = float(out.loc[i, "age"])
        value = float(out.loc[i, "value_end"])

        out.loc[i, "basis"] = basis

        if m >= start_m and m <= end_m:
            years_since = (m - start_m) / 12.0
            infl = (1 + plan.inflation) ** years_since
            amt = plan.amount * infl

            if plan.frequency == "annual" and (m - start_m) % 12 != 0:
                amt = 0.0

            value, basis, taxable, nontaxable = lifo_withdraw(value, basis, amt)
            tax = taxable * ord_r
            penalty = 0.0
            if plan.apply_59_5_penalty and age < 59.5:
                penalty = taxable * plan.penalty_rate

            net = amt - tax - penalty

            out.loc[i, "withdraw_gross"] = amt
            out.loc[i, "withdraw_taxable"] = taxable
            out.loc[i, "withdraw_nontaxable"] = nontaxable
            out.loc[i, "tax"] = tax
            out.loc[i, "penalty"] = penalty
            out.loc[i, "withdraw_net_cash"] = net
            out.loc[i, "value_end"] = value
            out.loc[i, "basis"] = basis

    return out

def period_certain_payment(pv: float, r_annual: float, years: int, freq: str) -> float:
    if freq == "monthly":
        n = years * 12
        r = monthly_rate_from_annual(r_annual)
    else:
        n = years
        r = r_annual
    if abs(r) < 1e-12:
        return pv / n
    return pv * (r / (1 - (1 + r) ** (-n)))

def ppva_annuitize_period_certain(account_value: float, basis: float, plan: AnnuitizationPlan, taxes: TaxInputs) -> pd.DataFrame:
    ord_r = eff_rate_ordinary(taxes)
    pmt = period_certain_payment(account_value, plan.pricing_rate, plan.payout_years, plan.payout_frequency)
    periods = plan.payout_years * (12 if plan.payout_frequency == "monthly" else 1)
    expected_total = pmt * periods
    exclusion_ratio = 0.0 if expected_total <= 0 else min(1.0, basis / expected_total)

    rows = []
    for k in range(1, periods + 1):
        tax_free = pmt * exclusion_ratio
        taxable = pmt - tax_free
        tax = taxable * ord_r
        rows.append({
            "period": k,
            "gross_payment": pmt,
            "tax_free_return_of_basis": tax_free,
            "taxable_ordinary_income": taxable,
            "tax": tax,
            "after_tax_cash": pmt - tax,
        })
    df = pd.DataFrame(rows)
    df.attrs["payment"] = pmt
    df.attrs["exclusion_ratio"] = exclusion_ratio
    df.attrs["expected_total_payout"] = expected_total
    return df
