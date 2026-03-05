import pandas as pd

def summarize_path(name: str, df: pd.DataFrame, value_col: str, cashflow_col: str = None, tax_col: str = None):
    end_value = float(df[value_col].iloc[-1]) if len(df) else 0.0
    total_cash = float(df[cashflow_col].sum()) if (cashflow_col and cashflow_col in df.columns) else 0.0
    total_tax = float(df[tax_col].sum()) if (tax_col and tax_col in df.columns) else 0.0
    return {
        "scenario": name,
        "ending_value": end_value,
        "total_cashflows": total_cash,
        "total_tax_paid": total_tax,
        "months": int(df.shape[0]),
    }
