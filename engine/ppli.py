import pandas as pd
from .types import PolicyInputs, TaxInputs, PPLIInputs
from .fees import apply_monthly_asset_fee, monthly_rate_from_annual
from .taxes import eff_rate_ordinary

def _interp_bps(age: float, curve: dict) -> float:
    if not curve:
        return 0.0
    xs = sorted(curve.keys())
    if age <= xs[0]:
        return float(curve[xs[0]])
    if age >= xs[-1]:
        return float(curve[xs[-1]])
    for i in range(len(xs)-1):
        a0, a1 = xs[i], xs[i+1]
        if a0 <= age <= a1:
            y0, y1 = float(curve[a0]), float(curve[a1])
            t = (age - a0) / (a1 - a0)
            return y0 + t * (y1 - y0)
    return float(curve[xs[-1]])

def _death_benefit(cash_value: float, face: float, corridor: float, option: str) -> float:
    if option == "B_increasing":
        return max(face + cash_value, corridor * cash_value)
    return max(face, corridor * cash_value)

def run_ppli(policy: PolicyInputs,
             taxes: TaxInputs,
             ppli: PPLIInputs,
             years: int,
             monthly_returns,
             assumed_face: float = None) -> pd.DataFrame:
    months = years * 12
    r = monthly_returns

    cash_value = policy.premium * (1 - ppli.premium_load)
    basis = policy.premium
    loan_balance = 0.0

    if assumed_face is None:
        assumed_face = ppli.corridor * policy.premium

    admin_m = ppli.admin_fee_annual / 12.0
    total_asset_bps = ppli.asset_charge_bps + ppli.fund_er_bps
    ord_r = eff_rate_ordinary(taxes)

    loan_r_m = monthly_rate_from_annual(ppli.loan_interest_rate)
    loan_cred_m = monthly_rate_from_annual(ppli.loan_crediting_rate)

    loan_start_m = None
    if ppli.loan_start_age is not None:
        loan_start_m = int(max(1, (ppli.loan_start_age - policy.issue_age) * 12 + 1))

    rows = []
    lapsed = False
    lapse_tax_event = 0.0

    for m in range(1, months + 1):
        age = policy.issue_age + (m-1)/12.0
        cv_start = cash_value

        cash_value = max(0.0, cash_value * (1 + r[m-1]))
        cash_value = apply_monthly_asset_fee(cash_value, total_asset_bps)
        cash_value = max(0.0, cash_value - admin_m)

        db = _death_benefit(cash_value, assumed_face, ppli.corridor, ppli.db_option)
        naar = max(0.0, db - cash_value)

        coi_bps = _interp_bps(age, ppli.coi_bps_by_age or {})
        coi_charge = (coi_bps / 10_000.0) / 12.0 * naar
        cash_value = max(0.0, cash_value - coi_charge)

        loan_balance = loan_balance * (1 + loan_r_m)
        loan_balance = max(0.0, loan_balance * (1 - loan_cred_m))

        loan_gross = 0.0
        loan_tax = 0.0
        if loan_start_m is not None and m >= loan_start_m and not lapsed:
            years_since = (m - loan_start_m) / 12.0
            infl = (1 + ppli.loan_inflation) ** years_since
            amt = ppli.loan_amount * infl

            if ppli.loan_frequency == "annual" and (m - loan_start_m) % 12 != 0:
                amt = 0.0

            if amt > 0:
                amt = min(amt, cash_value)
                cash_value -= amt
                loan_balance += amt
                loan_gross = amt

                if ppli.is_mec:
                    gain = max(0.0, (cash_value + loan_balance) - basis)
                    taxable = min(amt, gain)
                    loan_tax = taxable * ord_r

        if not lapsed and cash_value <= 0.0:
            lapsed = True
            if loan_balance > 0:
                deemed_value = loan_balance
                gain = max(0.0, deemed_value - basis)
                lapse_tax_event = gain * ord_r

        rows.append({
            "month": m,
            "age": age,
            "cash_value_start": cv_start,
            "cash_value_end": cash_value,
            "basis": basis,
            "death_benefit": db,
            "naar": naar,
            "coi_charge": coi_charge,
            "admin_fee": admin_m,
            "loan_gross": loan_gross,
            "loan_tax": loan_tax,
            "loan_balance": loan_balance,
            "lapsed": lapsed,
            "lapse_tax_event": lapse_tax_event if lapsed else 0.0,
        })

        if lapsed:
            break

    return pd.DataFrame(rows)
