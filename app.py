import streamlit as st
import pandas as pd

from engine.types import (
    PolicyInputs, StrategyInputs, TaxInputs, FeeInputs, TaxableInputs, PPLIInputs,
    WithdrawalPlan, AnnuitizationPlan
)
from engine.taxable import run_taxable
from engine.ppva import run_ppva_accum, apply_ppva_withdrawals, ppva_annuitize_period_certain
from engine.ppli import run_ppli
from engine.metrics import summarize_path
from engine.liquidation import liquidate_taxable, liquidate_ppva
from engine.formatting import fmt_money, fmt_money2, fmt_pct

st.set_page_config(page_title="Taxable vs PPVA vs PPLI (Planning)", layout="wide")
st.title("Taxable vs PPVA vs PPLI — Planning-Grade Model")

with st.sidebar:
    st.header("Scenarios")
    do_taxable = st.checkbox("Taxable", value=True)
    do_ppva = st.checkbox("PPVA", value=True)
    do_ppli = st.checkbox("PPLI", value=True)

    st.header("Base")
    premium = st.number_input("Initial investment / premium ($)", value=10_000_000.0, step=100_000.0)
    issue_age = st.number_input("Issue age", value=65, step=1)
    horizon_years = st.number_input("Horizon (years)", value=30, step=1)

    st.header("Strategy")
    exp_ret = st.number_input("Expected total return (%/yr)", value=8.0) / 100.0
    vol = st.number_input("Volatility (%/yr) (placeholder)", value=12.0) / 100.0

    st.subheader("Taxable return composition (%/yr of NAV)")
    ordinary_yield = st.number_input("Ordinary yield", value=4.0) / 100.0
    qdiv_yield = st.number_input("Qualified dividend yield", value=1.0) / 100.0
    realized_stcg = st.number_input("Realized STCG", value=1.0) / 100.0
    realized_ltcg = st.number_input("Realized LTCG", value=1.0) / 100.0
    unreal = st.number_input("Unrealized appreciation", value=1.0) / 100.0
    mgmt_fee_bps = st.number_input("Underlying strategy mgmt fee (bps)", value=0.0)

    st.header("Taxes")
    fed_ord = st.number_input("Fed ordinary (%)", value=37.0) / 100.0
    fed_st = st.number_input("Fed STCG (%)", value=37.0) / 100.0
    fed_lt = st.number_input("Fed LTCG (%)", value=20.0) / 100.0
    fed_qd = st.number_input("Fed qualified div (%)", value=20.0) / 100.0
    niit = st.number_input("NIIT (%)", value=3.8) / 100.0
    include_niit = st.checkbox("Include NIIT", value=True)

    st.subheader("State")
    st_ord = st.number_input("State ordinary (%)", value=3.0) / 100.0
    st_st = st.number_input("State STCG (%)", value=3.0) / 100.0
    st_lt = st.number_input("State LTCG (%)", value=3.0) / 100.0
    st_qd = st.number_input("State qualified div (%)", value=3.0) / 100.0

policy = PolicyInputs(premium=premium, issue_age=int(issue_age), premium_load=0.0)

strategy = StrategyInputs(
    expected_return=exp_ret,
    volatility=vol,
    ordinary_yield=ordinary_yield,
    qualified_div_yield=qdiv_yield,
    realized_stcg=realized_stcg,
    realized_ltcg=realized_ltcg,
    unrealized_appreciation=unreal,
    mgmt_fee_bps=mgmt_fee_bps,
)

taxes = TaxInputs(
    federal_ordinary=fed_ord,
    federal_ltcg=fed_lt,
    federal_stcg=fed_st,
    qualified_div=fed_qd,
    niit=niit,
    state_ordinary=st_ord,
    state_ltcg=st_lt,
    state_stcg=st_st,
    state_qualified_div=st_qd,
    include_niit=include_niit,
)

tabs = st.tabs(["Compare", "Taxable", "PPVA", "PPLI"])

# Session state to persist results after clicking Run
if "results" not in st.session_state:
    st.session_state["results"] = {}
if "summary" not in st.session_state:
    st.session_state["summary"] = pd.DataFrame()

with tabs[0]:
    st.subheader("Run & Compare")
    if st.button("Run model", type="primary"):
        results = {}
        summaries = []

       if do_taxable:
        c1, c2, c3 = st.columns(3)
        with c1:
            step_up = st.checkbox("Step-up at death", value=True, key="tx_stepup")
        with c2:
            death_age = st.number_input("Death age (for step-up)", value=90, step=1, key="tx_death_age")
        with c3:
            tlh = st.number_input("Tax-loss harvesting benefit (bps/yr)", value=0.0, key="tx_tlh")

    taxable_inputs = TaxableInputs(
        step_up_at_death=step_up, 
        death_age=int(death_age), 
        tlh_bps=float(tlh)
    )
    df_tax = run_taxable(policy, strategy, taxes, taxable_inputs, years=int(horizon_years))
    results["Taxable"] = df_tax

    end_val = float(df_tax["value_end"].iloc[-1])
    basis = float(df_tax["basis"].iloc[-1])

    after_tax_exit, exit_tax = liquidate_taxable(end_val, basis, taxes)

    summ = summarize_path("Taxable", df_tax, "value_end", tax_col="tax_paid")
    summ["exit_tax_at_horizon"] = exit_tax
    summ["ending_value_after_tax_exit"] = after_tax_exit
    summaries.append(summ)

        if do_ppva:
    st.write("PPVA settings are in the PPVA tab (defaults applied here).")
    ppva_fees = FeeInputs(wrapper_fee_bps=75.0, fund_er_bps=50.0, admin_fee_annual=5000.0)
    df_ppva = run_ppva_accum(policy, strategy, ppva_fees, years=int(horizon_years))
    results["PPVA (accum)"] = df_ppva

    end_val = float(df_ppva["value_end"].iloc[-1])
    basis = float(df_ppva["basis"].iloc[-1])

    after_tax_exit, exit_tax = liquidate_ppva(end_val, basis, taxes)

    summ = summarize_path("PPVA (accum)", df_ppva, "value_end")
    summ["exit_tax_at_horizon"] = exit_tax
    summ["ending_value_after_tax_exit"] = after_tax_exit
    summaries.append(summ)

        if do_ppli:
            default_curve = {65: 35, 75: 80, 85: 160, 95: 300}
            ppli_inputs = PPLIInputs(
                db_option="A_level",
                corridor=1.10,
                premium_load=0.01,
                admin_fee_annual=1200.0,
                asset_charge_bps=50.0,
                fund_er_bps=50.0,
                coi_bps_by_age=default_curve,
                loan_start_age=75,
                loan_amount=300_000.0,
                loan_frequency="annual",
                loan_inflation=0.0,
                loan_interest_rate=0.06,
                loan_crediting_rate=0.00,
                is_mec=False,
            )
            df_ppli = run_ppli(policy, strategy, taxes, ppli_inputs, years=int(horizon_years))
            results["PPLI"] = df_ppli
            summaries.append(summarize_path("PPLI", df_ppli, "cash_value_end", cashflow_col="loan_gross", tax_col="loan_tax"))

        st.session_state["results"] = results
        st.session_state["summary"] = pd.DataFrame(summaries)

   if not st.session_state["summary"].empty:
    summ = st.session_state["summary"].copy()

    # Add a few readable headline metrics
    st.markdown("### Headline Results (After-Tax Exit Where Applicable)")

    # formatted table for display
    display = summ.copy()

    money_cols = [
        "ending_value",
        "ending_value_after_tax_exit",
        "total_cashflows",
        "total_tax_paid",
        "exit_tax_at_horizon",
    ]
    for c in money_cols:
        if c in display.columns:
            display[c] = display[c].map(lambda v: fmt_money(float(v)) if pd.notnull(v) else "")

    st.dataframe(display, use_container_width=True)

    # Prefer after-tax exit column if present
    chart_col = "ending_value_after_tax_exit" if "ending_value_after_tax_exit" in summ.columns else "ending_value"
    st.bar_chart(summ.set_index("scenario")[chart_col])
with tabs[1]:
    st.subheader("Taxable schedule")
    df = st.session_state["results"].get("Taxable")
    if df is None:
        st.info("Run the model first.")
    else:
        df_show = df.tail(24).copy()
for c in ["value_start", "value_end", "basis", "tax_paid"]:
    if c in df_show.columns:
        df_show[c] = df_show[c].map(lambda v: fmt_money(float(v)))
st.dataframe(df_show, use_container_width=True)
        st.dataframe(df.tail(24), use_container_width=True)
        st.download_button("Download taxable CSV", df.to_csv(index=False).encode("utf-8"), "taxable.csv", "text/csv")

with tabs[2]:
    st.subheader("PPVA schedule (accumulation + optional withdrawals + annuitization)")
    df = st.session_state["results"].get("PPVA (accum)")
    if df is None:
        st.info("Run the model first (default PPVA fees).")
    else:
        df_show = df.tail(24).copy()
for c in ["value_start", "value_end", "basis", "tax_paid"]:
    if c in df_show.columns:
        df_show[c] = df_show[c].map(lambda v: fmt_money(float(v)))
st.dataframe(df_show, use_container_width=True)
        st.dataframe(df.tail(24), use_container_width=True)
        st.download_button("Download PPVA CSV", df.to_csv(index=False).encode("utf-8"), "ppva_accum.csv", "text/csv")

with tabs[3]:
    st.subheader("PPLI schedule")
    df = st.session_state["results"].get("PPLI")
    if df is None:
        st.info("Run the model first.")
    else:
        df_show = df.tail(24).copy()
for c in ["value_start", "value_end", "basis", "tax_paid"]:
    if c in df_show.columns:
        df_show[c] = df_show[c].map(lambda v: fmt_money(float(v)))
st.dataframe(df_show, use_container_width=True)
        st.dataframe(df.tail(24), use_container_width=True)
        st.download_button("Download PPLI CSV", df.to_csv(index=False).encode("utf-8"), "ppli.csv", "text/csv")
