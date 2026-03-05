import streamlit as st
import pandas as pd
import numpy as np

from engine.types import (
    PolicyInputs, StrategyInputs, TaxInputs, FeeInputs, TaxableInputs, PPLIInputs,
    WithdrawalPlan, AnnuitizationPlan
)
from engine.strategy import make_monthly_return_path
from engine.taxable import run_taxable
from engine.ppva import run_ppva_accum, apply_ppva_withdrawals, ppva_annuitize_period_certain
from engine.ppli import run_ppli
from engine.liquidation import liquidate_taxable, liquidate_ppva
from engine.formatting import fmt_money, fmt_money2
from engine.irr import irr_monthly

st.set_page_config(page_title="Taxable vs PPVA vs PPLI (Advanced)", layout="wide")
st.title("Taxable vs PPVA vs PPLI — Advanced Planning Model")

# ---------------- Session-state defaults (cross-tab wiring) ----------------
if "results" not in st.session_state:
    st.session_state["results"] = {}
if "summary" not in st.session_state:
    st.session_state["summary"] = pd.DataFrame()

# PPVA defaults used by Compare unless user changes in PPVA tab
if "ppva_cfg" not in st.session_state:
    st.session_state["ppva_cfg"] = {
        "fees": {"wrap_bps": 75.0, "er_bps": 50.0, "admin": 5000.0},
        "withdrawals": {
            "enabled": False,
            "start_age": 75,
            "end_age": 90,
            "amount": 250_000.0,
            "frequency": "annual",
            "inflation": 0.0,
            "penalty": True,
        },
        "annuitize": {
            "enabled": False,
            "ann_age": 85,
            "payout_years": 20,
            "pricing_rate": 0.055,
            "pay_freq": "monthly",
        }
    }

# PPLI defaults used by Compare unless user changes in PPLI tab
if "ppli_cfg" not in st.session_state:
    st.session_state["ppli_cfg"] = {
        "premium_load": 0.01,
        "admin_fee_annual": 1200.0,
        "asset_charge_bps": 50.0,
        "fund_er_bps": 50.0,
        "coi_text": "65:35\n75:80\n85:160\n95:300",
        "loan_start_age": 75,
        "loan_amount": 300_000.0,
        "loan_frequency": "annual",
        "loan_inflation": 0.0,
        "loan_interest_rate": 0.06,
        "loan_crediting_rate": 0.00,
        "is_mec": False,
        "death_age": 90,
    }

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Scenarios")
    do_taxable = st.checkbox("Taxable", value=True, key="scen_taxable")
    do_ppva = st.checkbox("PPVA", value=True, key="scen_ppva")
    do_ppli = st.checkbox("PPLI", value=True, key="scen_ppli")

    st.header("Horizon & Return Model")
    premium = st.number_input("Initial investment / premium ($)", value=10_000_000.0, step=100_000.0, key="base_premium")
    issue_age = st.number_input("Issue age", value=65, step=1, key="base_issue_age")
    horizon_years = st.number_input("Base projection years", value=30, step=1, key="base_horizon_years")

    return_mode = st.selectbox("Return mode", ["deterministic", "monte_carlo"], index=0, key="ret_mode")
    mc_sims = st.number_input("Monte Carlo sims", value=250, step=50, min_value=50, max_value=5000, key="mc_sims")
    mc_seed = st.number_input("MC seed", value=42, step=1, key="mc_seed")

    st.header("Strategy (total return)")
    exp_ret = st.number_input("Expected total return (%/yr)", value=8.0, key="strat_exp_ret") / 100.0
    vol = st.number_input("Volatility (%/yr)", value=12.0, key="strat_vol") / 100.0

    st.subheader("Taxable return composition (%/yr of NAV)")
    ordinary_yield = st.number_input("Ordinary yield", value=4.0, key="comp_ordinary") / 100.0
    qdiv_yield = st.number_input("Qualified dividend yield", value=1.0, key="comp_qdiv") / 100.0
    realized_stcg = st.number_input("Realized STCG", value=1.0, key="comp_stcg") / 100.0
    realized_ltcg = st.number_input("Realized LTCG", value=1.0, key="comp_ltcg") / 100.0
    unreal = st.number_input("Unrealized appreciation", value=1.0, key="comp_unreal") / 100.0
    mgmt_fee_bps = st.number_input("Underlying strategy mgmt fee (bps)", value=0.0, key="comp_mgmt_fee_bps")

    st.header("Taxes")
    fed_ord = st.number_input("Fed ordinary (%)", value=37.0, key="tax_fed_ord") / 100.0
    fed_st = st.number_input("Fed STCG (%)", value=37.0, key="tax_fed_st") / 100.0
    fed_lt = st.number_input("Fed LTCG (%)", value=20.0, key="tax_fed_lt") / 100.0
    fed_qd = st.number_input("Fed qualified div (%)", value=20.0, key="tax_fed_qd") / 100.0
    niit = st.number_input("NIIT (%)", value=3.8, key="tax_niit") / 100.0
    include_niit = st.checkbox("Include NIIT", value=True, key="tax_include_niit")

    st.subheader("State")
    st_ord = st.number_input("State ordinary (%)", value=3.0, key="tax_state_ord") / 100.0
    st_st = st.number_input("State STCG (%)", value=3.0, key="tax_state_st") / 100.0
    st_lt = st.number_input("State LTCG (%)", value=3.0, key="tax_state_lt") / 100.0
    st_qd = st.number_input("State qualified div (%)", value=3.0, key="tax_state_qd") / 100.0

# ---------------- Inputs objects ----------------
policy = PolicyInputs(premium=float(premium), issue_age=int(issue_age), premium_load=0.0)

strategy = StrategyInputs(
    expected_return=float(exp_ret),
    volatility=float(vol),
    ordinary_yield=float(ordinary_yield),
    qualified_div_yield=float(qdiv_yield),
    realized_stcg=float(realized_stcg),
    realized_ltcg=float(realized_ltcg),
    unrealized_appreciation=float(unreal),
    mgmt_fee_bps=float(mgmt_fee_bps),
)

taxes = TaxInputs(
    federal_ordinary=float(fed_ord),
    federal_ltcg=float(fed_lt),
    federal_stcg=float(fed_st),
    qualified_div=float(fed_qd),
    niit=float(niit),
    state_ordinary=float(st_ord),
    state_ltcg=float(st_lt),
    state_stcg=float(st_st),
    state_qualified_div=float(st_qd),
    include_niit=bool(include_niit),
)

tabs = st.tabs(["Compare", "Taxable", "PPVA", "PPLI", "Monte Carlo"])

def _parse_coi_text(coi_text: str) -> dict:
    curve = {}
    for line in (coi_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        a, b = line.split(":")
        curve[int(a.strip())] = float(b.strip())
    return curve

# ---------------- Helper: build one run ----------------
def run_once(seed: int):
    years = int(horizon_years)
    months = years * 12

    mode = "monte_carlo" if return_mode == "monte_carlo" else "deterministic"
    r = make_monthly_return_path(strategy, months, mode=mode, seed=int(seed))

    results = {}
    summary_rows = []

    # ---- Taxable assumptions (still default; easy to expose later) ----
    taxable_inputs = TaxableInputs(step_up_at_death=True, death_age=int(issue_age) + years, tlh_bps=0.0)

    # ---- PPVA from session_state ----
    ppva_ss = st.session_state["ppva_cfg"]
    ppva_fees = FeeInputs(
        wrapper_fee_bps=float(ppva_ss["fees"]["wrap_bps"]),
        fund_er_bps=float(ppva_ss["fees"]["er_bps"]),
        admin_fee_annual=float(ppva_ss["fees"]["admin"]),
    )

    # ---- PPLI from session_state ----
    ppli_ss = st.session_state["ppli_cfg"]
    coi_curve = _parse_coi_text(str(ppli_ss["coi_text"]))

    ppli_inputs = PPLIInputs(
        db_option="A_level",
        corridor=1.10,
        premium_load=float(ppli_ss["premium_load"]),
        admin_fee_annual=float(ppli_ss["admin_fee_annual"]),
        asset_charge_bps=float(ppli_ss["asset_charge_bps"]),
        fund_er_bps=float(ppli_ss["fund_er_bps"]),
        coi_bps_by_age=coi_curve,
        loan_start_age=int(ppli_ss["loan_start_age"]),
        loan_amount=float(ppli_ss["loan_amount"]),
        loan_frequency=str(ppli_ss["loan_frequency"]),
        loan_inflation=float(ppli_ss["loan_inflation"]),
        loan_interest_rate=float(ppli_ss["loan_interest_rate"]),
        loan_crediting_rate=float(ppli_ss["loan_crediting_rate"]),
        is_mec=bool(ppli_ss["is_mec"]),
    )

    # ---------------- Taxable ----------------
    if do_taxable:
        df_tax = run_taxable(policy, strategy, taxes, taxable_inputs, years=years, monthly_returns=r)
        results["Taxable"] = df_tax

        end_val = float(df_tax["value_end"].iloc[-1])
        basis = float(df_tax["basis"].iloc[-1])
        after_tax_exit, exit_tax = liquidate_taxable(end_val, basis, taxes)

        cf = np.zeros(months + 1)
        cf[0] = -float(premium)
        cf[-1] = float(after_tax_exit)
        _, irr_a = irr_monthly(cf)

        pre_tax_cagr = (end_val / float(premium)) ** (1 / years) - 1 if end_val > 0 else None
        after_tax_cagr = (float(after_tax_exit) / float(premium)) ** (1 / years) - 1 if float(after_tax_exit) > 0 else None
        tax_drag_dollars = end_val - float(after_tax_exit)
        tax_drag_pct = (tax_drag_dollars / end_val) if end_val > 0 else None

        summary_rows.append({
            "scenario": "Taxable",
            "ending_value": end_val,
            "ending_value_after_tax_exit": float(after_tax_exit),
            "exit_tax_at_horizon": float(exit_tax),
            "total_tax_paid": float(df_tax["tax_paid"].sum()),
            "total_cashflows": 0.0,

            "pre_tax_cagr": pre_tax_cagr,
            "after_tax_cagr": after_tax_cagr,
            "tax_drag_$": tax_drag_dollars,
            "tax_drag_%_of_end": tax_drag_pct,

            "irr_annual": irr_a,
        })

    # ---------------- PPVA ----------------
    if do_ppva:
        df_ppva = run_ppva_accum(policy, ppva_fees, years=years, monthly_returns=r)

        # optional withdrawals from PPVA tab config
        w = ppva_ss["withdrawals"]
        if bool(w.get("enabled", False)):
            plan = WithdrawalPlan(
                start_age=int(w["start_age"]),
                end_age=int(w["end_age"]),
                amount=float(w["amount"]),
                frequency=str(w["frequency"]),
                inflation=float(w["inflation"]),
                apply_59_5_penalty=bool(w["penalty"]),
            )
            df_ppva = apply_ppva_withdrawals(df_ppva, plan, taxes)

        results["PPVA"] = df_ppva

        end_val = float(df_ppva["value_end"].iloc[-1])
        basis = float(df_ppva["basis"].iloc[-1])
        after_tax_exit, exit_tax = liquidate_ppva(end_val, basis, taxes)

        cf = np.zeros(months + 1)
        cf[0] = -float(premium)
        cf[-1] = float(after_tax_exit)
        _, irr_a = irr_monthly(cf)

        pre_tax_cagr = (end_val / float(premium)) ** (1 / years) - 1 if end_val > 0 else None
        after_tax_cagr = (float(after_tax_exit) / float(premium)) ** (1 / years) - 1 if float(after_tax_exit) > 0 else None
        tax_drag_dollars = end_val - float(after_tax_exit)
        tax_drag_pct = (tax_drag_dollars / end_val) if end_val > 0 else None

        summary_rows.append({
            "scenario": "PPVA (Surrender @ horizon)",
            "ending_value": end_val,
            "ending_value_after_tax_exit": float(after_tax_exit),
            "exit_tax_at_horizon": float(exit_tax),
            "total_tax_paid": 0.0,
            "total_cashflows": float(df_ppva["withdraw_net_cash"].sum()) if "withdraw_net_cash" in df_ppva.columns else 0.0,

            "pre_tax_cagr": pre_tax_cagr,
            "after_tax_cagr": after_tax_cagr,
            "tax_drag_$": tax_drag_dollars,
            "tax_drag_%_of_end": tax_drag_pct,

            "irr_annual": irr_a,
        })

    # ---------------- PPLI ----------------
    if do_ppli:
        df_ppli = run_ppli(policy, taxes, ppli_inputs, years=years, monthly_returns=r)
        results["PPLI"] = df_ppli

        death_age = int(ppli_ss.get("death_age", int(issue_age) + years))
        death_m = int(max(1, (death_age - int(issue_age)) * 12))
        death_m = min(death_m, len(df_ppli))
        row = df_ppli.iloc[death_m - 1]

        lapsed = bool(row["lapsed"])
        if lapsed:
            net_to_heirs = 0.0
            exit_tax = float(row.get("lapse_tax_event", 0.0))
        else:
            db = float(row["death_benefit"])
            loan_bal = float(row["loan_balance"])
            net_to_heirs = max(0.0, db - loan_bal)
            exit_tax = 0.0

        loans_received = float(df_ppli["loan_gross"].sum())
        total_benefit = loans_received + float(net_to_heirs)

        months_effective = len(df_ppli)
        cf = np.zeros(months_effective + 1)
        cf[0] = -float(premium)
        for i in range(months_effective):
            cf[i + 1] += float(df_ppli["loan_gross"].iloc[i])
        cf[-1] += float(net_to_heirs)
        if exit_tax > 0:
            cf[-1] -= float(exit_tax)
        _, irr_a = irr_monthly(cf)

        cv_end = float(df_ppli["cash_value_end"].iloc[-1])
        pre_tax_cagr = (cv_end / float(premium)) ** (1 / years) - 1 if cv_end > 0 else None
        after_tax_cagr = (float(net_to_heirs) / float(premium)) ** (1 / years) - 1 if float(net_to_heirs) > 0 else None

        summary_rows.append({
            "scenario": "PPLI (Net DB @ death)",
            "ending_value": cv_end,
            "ending_value_after_tax_exit": float(net_to_heirs),  # interpret as net to heirs
            "exit_tax_at_horizon": float(exit_tax),
            "total_tax_paid": float(df_ppli["loan_tax"].sum()),
            "total_cashflows": loans_received,  # loans to client

            "ppli_loans_received": loans_received,
            "ppli_net_to_heirs": float(net_to_heirs),
            "ppli_total_benefit": float(total_benefit),

            "pre_tax_cagr": pre_tax_cagr,
            "after_tax_cagr": after_tax_cagr,
            "irr_annual": irr_a,
        })

    return results, pd.DataFrame(summary_rows)

# ---------------- Compare tab ----------------
with tabs[0]:
    st.subheader("Compare")

    if st.button("Run model", type="primary", key="btn_run_model_compare"):
        results, summ = run_once(int(mc_seed))
        st.session_state["results"] = results
        st.session_state["summary"] = summ

    if not st.session_state["summary"].empty:
        summ = st.session_state["summary"].copy()

        # ---- Growth chart (account values) ----
        growth = None
        if st.session_state["results"]:
            # pick a month index from any available df
            base_df = None
            for k in ["Taxable", "PPVA", "PPLI"]:
                if k in st.session_state["results"]:
                    base_df = st.session_state["results"][k]
                    break

            if base_df is not None and "month" in base_df.columns:
                growth = pd.DataFrame({"month": base_df["month"].astype(int)})

                if "Taxable" in st.session_state["results"]:
                    growth["Taxable"] = st.session_state["results"]["Taxable"]["value_end"].astype(float)
                if "PPVA" in st.session_state["results"]:
                    growth["PPVA (CV)"] = st.session_state["results"]["PPVA"]["value_end"].astype(float)
                if "PPLI" in st.session_state["results"]:
                    growth["PPLI (CV)"] = st.session_state["results"]["PPLI"]["cash_value_end"].astype(float)

        if growth is not None and growth.shape[1] > 1:
            st.markdown("### Growth chart (account values)")
            st.line_chart(growth.set_index("month"))

        # ---- Format summary table ----
        display = summ.copy()

        money_cols = [
            "ending_value",
            "ending_value_after_tax_exit",
            "exit_tax_at_horizon",
            "total_tax_paid",
            "total_cashflows",
            "tax_drag_$",
            "ppli_loans_received",
            "ppli_net_to_heirs",
            "ppli_total_benefit",
        ]
        pct_cols = ["pre_tax_cagr", "after_tax_cagr", "irr_annual", "tax_drag_%_of_end"]

        for c in money_cols:
            if c in display.columns:
                display[c] = display[c].map(lambda v: fmt_money(float(v)) if pd.notnull(v) else "")

        for c in pct_cols:
            if c in display.columns:
                display[c] = display[c].map(lambda v: f"{100*float(v):.2f}%" if (pd.notnull(v) and v is not None) else "")

        rename_map = {
            "ending_value": "Ending value (pre-tax)",
            "ending_value_after_tax_exit": "After-tax value (Net to heirs for PPLI)",
            "exit_tax_at_horizon": "Exit tax @ horizon",
            "total_tax_paid": "Ongoing taxes",
            "total_cashflows": "Cashflows to client (loans/withdrawals)",
            "pre_tax_cagr": "Pre-tax CAGR",
            "after_tax_cagr": "After-tax CAGR",
            "irr_annual": "After-tax IRR",
            "tax_drag_$": "Tax drag ($)",
            "tax_drag_%_of_end": "Tax drag (% of end)",
            "ppli_loans_received": "PPLI loans received",
            "ppli_net_to_heirs": "PPLI net to heirs",
            "ppli_total_benefit": "PPLI total benefit (loans + heirs)",
        }
        display = display.rename(columns={k: v for k, v in rename_map.items() if k in display.columns})

        st.dataframe(display, use_container_width=True)

        st.markdown("### After-tax comparison (Net to heirs for PPLI)")
        st.bar_chart(summ.set_index("scenario")["ending_value_after_tax_exit"])

# ---------------- Taxable tab ----------------
with tabs[1]:
    st.subheader("Taxable")
    df = st.session_state["results"].get("Taxable")
    if df is None:
        st.info("Run the model first.")
    else:
        df_show = df.tail(24).copy()
        for c in ["value_start", "value_end", "basis", "tax_paid"]:
            if c in df_show.columns:
                df_show[c] = df_show[c].map(lambda v: fmt_money(float(v)))
        st.dataframe(df_show, use_container_width=True)
        st.download_button("Download taxable CSV", df.to_csv(index=False).encode("utf-8"), "taxable.csv", "text/csv", key="dl_taxable")

# ---------------- PPVA tab ----------------
with tabs[2]:
    st.subheader("PPVA (Advanced)")

    st.markdown("### Fees")
    c1, c2, c3 = st.columns(3)
    with c1:
        ppva_wrap = st.number_input("Wrapper fee (bps)", value=float(st.session_state["ppva_cfg"]["fees"]["wrap_bps"]), step=5.0, key="ppva_wrap_bps")
    with c2:
        ppva_er = st.number_input("Fund ER (bps)", value=float(st.session_state["ppva_cfg"]["fees"]["er_bps"]), step=5.0, key="ppva_fund_er_bps")
    with c3:
        ppva_admin = st.number_input("Admin fee ($/yr)", value=float(st.session_state["ppva_cfg"]["fees"]["admin"]), step=500.0, key="ppva_admin_fee")

    st.markdown("### Withdrawals (optional)")
    use_w = st.checkbox("Apply systematic withdrawals", value=bool(st.session_state["ppva_cfg"]["withdrawals"]["enabled"]), key="ppva_use_withdrawals")
    w1, w2 = st.columns(2)
    with w1:
        w_start_age = st.number_input("Withdrawal start age", value=int(st.session_state["ppva_cfg"]["withdrawals"]["start_age"]), step=1, key="ppva_w_start_age")
    with w2:
        w_end_age = st.number_input("Withdrawal end age", value=int(st.session_state["ppva_cfg"]["withdrawals"]["end_age"]), step=1, key="ppva_w_end_age")

    c1, c2, c3 = st.columns(3)
    with c1:
        w_amt = st.number_input("Withdrawal amount ($ per period)", value=float(st.session_state["ppva_cfg"]["withdrawals"]["amount"]), step=25_000.0, key="ppva_w_amt")
    with c2:
        w_freq = st.selectbox("Withdrawal frequency", ["monthly", "annual"], index=(0 if st.session_state["ppva_cfg"]["withdrawals"]["frequency"] == "monthly" else 1), key="ppva_w_freq")
    with c3:
        w_infl = st.number_input("Withdrawal inflation (%/yr)", value=float(st.session_state["ppva_cfg"]["withdrawals"]["inflation"]) * 100.0, key="ppva_w_infl") / 100.0

    w_pen = st.checkbox("Apply <59½ 10% penalty (taxable portion)", value=bool(st.session_state["ppva_cfg"]["withdrawals"]["penalty"]), key="ppva_w_penalty")

    st.markdown("### Annuitization (optional)")
    use_ann = st.checkbox("Annuitize at a specific age", value=bool(st.session_state["ppva_cfg"]["annuitize"]["enabled"]), key="ppva_use_annuitize")
    a1, a2, a3 = st.columns(3)
    with a1:
        ann_age = st.number_input("Annuitize age", value=int(st.session_state["ppva_cfg"]["annuitize"]["ann_age"]), step=1, key="ppva_ann_age")
    with a2:
        payout_years = st.number_input("Payout years", value=int(st.session_state["ppva_cfg"]["annuitize"]["payout_years"]), step=1, key="ppva_payout_years")
    with a3:
        pricing_rate = st.number_input("Pricing rate (%/yr)", value=float(st.session_state["ppva_cfg"]["annuitize"]["pricing_rate"]) * 100.0, key="ppva_pricing_rate") / 100.0
    pay_freq = st.selectbox("Payment frequency", ["monthly", "annual"], index=(0 if st.session_state["ppva_cfg"]["annuitize"]["pay_freq"] == "monthly" else 1), key="ppva_pay_freq")

    # Save PPVA settings for Compare
    st.session_state["ppva_cfg"] = {
        "fees": {"wrap_bps": float(ppva_wrap), "er_bps": float(ppva_er), "admin": float(ppva_admin)},
        "withdrawals": {
            "enabled": bool(use_w),
            "start_age": int(w_start_age),
            "end_age": int(w_end_age),
            "amount": float(w_amt),
            "frequency": str(w_freq),
            "inflation": float(w_infl),
            "penalty": bool(w_pen),
        },
        "annuitize": {
            "enabled": bool(use_ann),
            "ann_age": int(ann_age),
            "payout_years": int(payout_years),
            "pricing_rate": float(pricing_rate),
            "pay_freq": str(pay_freq),
        }
    }

    years = int(horizon_years)
    months = years * 12
    mode = "monte_carlo" if return_mode == "monte_carlo" else "deterministic"
    r = make_monthly_return_path(strategy, months, mode=mode, seed=int(mc_seed))

    fees = FeeInputs(wrapper_fee_bps=float(ppva_wrap), fund_er_bps=float(ppva_er), admin_fee_annual=float(ppva_admin))
    df_ppva = run_ppva_accum(policy, fees, years=years, monthly_returns=r)

    if use_w:
        plan = WithdrawalPlan(
            start_age=int(w_start_age),
            end_age=int(w_end_age),
            amount=float(w_amt),
            frequency=str(w_freq),
            inflation=float(w_infl),
            apply_59_5_penalty=bool(w_pen),
        )
        df_ppva = apply_ppva_withdrawals(df_ppva, plan, taxes)

    st.markdown("### Results")
    st.metric("End value (pre-tax)", fmt_money(float(df_ppva["value_end"].iloc[-1])))
    st.metric("Basis (planning)", fmt_money(float(df_ppva["basis"].iloc[-1])))

    if use_ann:
        months_to_ann = int(max(0, (int(ann_age) - int(issue_age)) * 12))
        months_to_ann = min(months_to_ann, len(df_ppva))
        idx = max(0, months_to_ann - 1)
        acct_val = float(df_ppva["value_end"].iloc[idx])
        basis = float(df_ppva["basis"].iloc[idx])

        ann_plan = AnnuitizationPlan(
            annuitize_age=int(ann_age),
            payout_years=int(payout_years),
            pricing_rate=float(pricing_rate),
            payout_frequency=str(pay_freq),
        )
        ann_df = ppva_annuitize_period_certain(acct_val, basis, ann_plan, taxes)

        st.markdown("### Annuitization cashflows")
        st.metric("Payment per period", fmt_money2(float(ann_df.attrs["payment"])))
        st.metric("Exclusion ratio", f"{ann_df.attrs['exclusion_ratio']*100:.2f}%")
        st.metric("Total after-tax cash", fmt_money(float(ann_df["after_tax_cash"].sum())))
        st.dataframe(ann_df.head(36), use_container_width=True)
        st.download_button("Download annuitization CSV", ann_df.to_csv(index=False).encode("utf-8"), "ppva_annuitization.csv", "text/csv", key="dl_ppva_ann")

    st.markdown("### Schedule (last 24 months)")
    df_show = df_ppva.tail(24).copy()
    for c in ["value_start", "value_end", "basis", "withdraw_gross", "withdraw_net_cash", "tax", "penalty"]:
        if c in df_show.columns:
            df_show[c] = df_show[c].map(lambda v: fmt_money(float(v)))
    st.dataframe(df_show, use_container_width=True)
    st.download_button("Download PPVA CSV", df_ppva.to_csv(index=False).encode("utf-8"), "ppva.csv", "text/csv", key="dl_ppva")

# ---------------- PPLI tab ----------------
with tabs[3]:
    st.subheader("PPLI (Advanced, Generic)")

    st.markdown("### Charges")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ppli_load = st.number_input("Premium load (%)", value=float(st.session_state["ppli_cfg"]["premium_load"]) * 100.0, key="ppli_premium_load_pct") / 100.0
    with c2:
        ppli_admin = st.number_input("Admin fee ($/yr)", value=float(st.session_state["ppli_cfg"]["admin_fee_annual"]), step=100.0, key="ppli_admin_fee")
    with c3:
        ppli_asset = st.number_input("Asset charge (bps)", value=float(st.session_state["ppli_cfg"]["asset_charge_bps"]), step=5.0, key="ppli_asset_charge_bps")
    with c4:
        ppli_er = st.number_input("Fund ER (bps)", value=float(st.session_state["ppli_cfg"]["fund_er_bps"]), step=5.0, key="ppli_fund_er_bps")

    st.markdown("### COI curve (bps of NAAR per year)")
    st.caption("Enter as age:bps per line. Example: 65:35")
    coi_text = st.text_area("COI curve", value=str(st.session_state["ppli_cfg"]["coi_text"]), key="ppli_coi_text")

    st.markdown("### Loan strategy")
    c1, c2, c3 = st.columns(3)
    with c1:
        loan_start_age = st.number_input("Loan start age", value=int(st.session_state["ppli_cfg"]["loan_start_age"]), step=1, key="ppli_loan_start_age")
    with c2:
        loan_amt = st.number_input("Loan amount ($)", value=float(st.session_state["ppli_cfg"]["loan_amount"]), step=50_000.0, key="ppli_loan_amt")
    with c3:
        loan_freq = st.selectbox("Loan frequency", ["annual", "monthly"],
                                 index=(0 if st.session_state["ppli_cfg"]["loan_frequency"] == "annual" else 1),
                                 key="ppli_loan_freq")

    c1, c2, c3 = st.columns(3)
    with c1:
        loan_infl = st.number_input("Loan inflation (%/yr)", value=float(st.session_state["ppli_cfg"]["loan_inflation"]) * 100.0, key="ppli_loan_infl") / 100.0
    with c2:
        loan_int = st.number_input("Loan interest (%/yr)", value=float(st.session_state["ppli_cfg"]["loan_interest_rate"]) * 100.0, key="ppli_loan_int") / 100.0
    with c3:
        loan_cred = st.number_input("Loan crediting (%/yr)", value=float(st.session_state["ppli_cfg"]["loan_crediting_rate"]) * 100.0, key="ppli_loan_cred") / 100.0

    is_mec = st.checkbox("Treat as MEC (loans taxable)", value=bool(st.session_state["ppli_cfg"]["is_mec"]), key="ppli_is_mec")

    st.markdown("### Death comparison")
    death_age = st.number_input("Assumed death age", value=int(st.session_state["ppli_cfg"]["death_age"]), step=1, key="ppli_death_age")

    # Save PPLI settings for Compare
    st.session_state["ppli_cfg"] = {
        "premium_load": float(ppli_load),
        "admin_fee_annual": float(ppli_admin),
        "asset_charge_bps": float(ppli_asset),
        "fund_er_bps": float(ppli_er),
        "coi_text": str(coi_text),
        "loan_start_age": int(loan_start_age),
        "loan_amount": float(loan_amt),
        "loan_frequency": str(loan_freq),
        "loan_inflation": float(loan_infl),
        "loan_interest_rate": float(loan_int),
        "loan_crediting_rate": float(loan_cred),
        "is_mec": bool(is_mec),
        "death_age": int(death_age),
    }

    years = int(horizon_years)
    months = years * 12
    mode = "monte_carlo" if return_mode == "monte_carlo" else "deterministic"
    r = make_monthly_return_path(strategy, months, mode=mode, seed=int(mc_seed))

    coi_curve = _parse_coi_text(str(coi_text))

    ppli_inputs = PPLIInputs(
        db_option="A_level",
        corridor=1.10,
        premium_load=float(ppli_load),
        admin_fee_annual=float(ppli_admin),
        asset_charge_bps=float(ppli_asset),
        fund_er_bps=float(ppli_er),
        coi_bps_by_age=coi_curve,
        loan_start_age=int(loan_start_age),
        loan_amount=float(loan_amt),
        loan_frequency=str(loan_freq),
        loan_inflation=float(loan_infl),
        loan_interest_rate=float(loan_int),
        loan_crediting_rate=float(loan_cred),
        is_mec=bool(is_mec),
    )

    df_ppli = run_ppli(policy, taxes, ppli_inputs, years=years, monthly_returns=r)

    st.metric("Ending cash value", fmt_money(float(df_ppli["cash_value_end"].iloc[-1])))
    st.metric("Loans received (sum)", fmt_money(float(df_ppli["loan_gross"].sum())))
    st.metric("Loan balance", fmt_money(float(df_ppli["loan_balance"].iloc[-1])))

    death_m = int(max(1, (int(death_age) - int(issue_age)) * 12))
    death_m = min(death_m, len(df_ppli))
    row = df_ppli.iloc[death_m - 1]
    lapsed = bool(row["lapsed"])

    if lapsed:
        st.error("Policy lapsed before death age in this run (planning-grade lapse logic).")
        st.metric("Lapse tax event", fmt_money(float(row.get("lapse_tax_event", 0.0))))
    else:
        net_to_heirs = max(0.0, float(row["death_benefit"]) - float(row["loan_balance"]))
        st.metric("Death benefit", fmt_money(float(row["death_benefit"])))
        st.metric("Net to heirs (DB - loan)", fmt_money(net_to_heirs))
        st.metric("Total benefit (loans + heirs)", fmt_money(float(df_ppli["loan_gross"].sum()) + float(net_to_heirs)))

    st.markdown("### Schedule (last 24 months)")
    df_show = df_ppli.tail(24).copy()
    for c in ["cash_value_start", "cash_value_end", "loan_balance", "loan_gross", "coi_charge"]:
        if c in df_show.columns:
            df_show[c] = df_show[c].map(lambda v: fmt_money(float(v)))
    st.dataframe(df_show, use_container_width=True)
    st.download_button("Download PPLI CSV", df_ppli.to_csv(index=False).encode("utf-8"), "ppli.csv", "text/csv", key="dl_ppli")

# ---------------- Monte Carlo tab ----------------
with tabs[4]:
    st.subheader("Monte Carlo")
    st.write("Runs the selected scenarios multiple times and shows distribution of after-tax outcomes and IRR.")

    if st.button("Run Monte Carlo", type="primary", key="btn_run_mc"):
        sims = int(mc_sims)
        rows = []
        for i in range(sims):
            seed = int(mc_seed) + i
            _, summ = run_once(seed)
            for _, rr in summ.iterrows():
                rows.append({
                    "scenario": rr["scenario"],
                    "after_tax_value": float(rr.get("ending_value_after_tax_exit", np.nan)),
                    "irr_annual": float(rr.get("irr_annual", np.nan)) if rr.get("irr_annual", None) is not None else np.nan,
                    "ppli_total_benefit": float(rr.get("ppli_total_benefit", np.nan)) if "ppli_total_benefit" in rr else np.nan,
                })
        mc_df = pd.DataFrame(rows)

        def stats(g):
            return pd.Series({
                "p10_after_tax": np.nanquantile(g["after_tax_value"], 0.10),
                "p50_after_tax": np.nanquantile(g["after_tax_value"], 0.50),
                "p90_after_tax": np.nanquantile(g["after_tax_value"], 0.90),
                "p50_irr": np.nanquantile(g["irr_annual"], 0.50),
            })

        out = mc_df.groupby("scenario").apply(stats).reset_index()

        display = out.copy()
        for c in ["p10_after_tax", "p50_after_tax", "p90_after_tax"]:
            display[c] = display[c].map(lambda v: fmt_money(float(v)) if pd.notnull(v) else "")
        display["p50_irr"] = display["p50_irr"].map(lambda v: f"{100*float(v):.2f}%" if pd.notnull(v) else "")

        st.dataframe(display, use_container_width=True)
