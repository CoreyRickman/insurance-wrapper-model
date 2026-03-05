def fmt_money(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return str(x)

def fmt_money2(x: float) -> str:
    try:
        return f"${x:,.2f}"
    except Exception:
        return str(x)

def fmt_pct(x: float) -> str:
    try:
        return f"{100*x:.2f}%"
    except Exception:
        return str(x)
