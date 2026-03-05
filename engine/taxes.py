from .types import TaxInputs

def eff_rate_ordinary(t: TaxInputs) -> float:
    r = t.federal_ordinary + t.state_ordinary
    if t.include_niit:
        r += t.niit
    return min(max(r, 0.0), 0.90)

def eff_rate_stcg(t: TaxInputs) -> float:
    r = t.federal_stcg + t.state_stcg
    if t.include_niit:
        r += t.niit
    return min(max(r, 0.0), 0.90)

def eff_rate_ltcg(t: TaxInputs) -> float:
    r = t.federal_ltcg + t.state_ltcg
    if t.include_niit:
        r += t.niit
    return min(max(r, 0.0), 0.90)

def eff_rate_qdiv(t: TaxInputs) -> float:
    r = t.qualified_div + t.state_qualified_div
    if t.include_niit:
        r += t.niit
    return min(max(r, 0.0), 0.90)
