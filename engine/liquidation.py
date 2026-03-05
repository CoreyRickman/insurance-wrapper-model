from .taxes import eff_rate_ordinary, eff_rate_ltcg
from .types import TaxInputs

def liquidate_taxable(end_value: float, basis: float, taxes: TaxInputs) -> float:
    """After-tax value if the taxable account is sold at horizon (planning-grade LTCG)."""
    gain = max(0.0, end_value - basis)
    tax = gain * eff_rate_ltcg(taxes)
    return max(0.0, end_value - tax), tax

def liquidate_ppva(end_value: float, basis: float, taxes: TaxInputs) -> float:
    """After-tax value if PPVA is surrendered at horizon (ordinary income on gain)."""
    gain = max(0.0, end_value - basis)
    tax = gain * eff_rate_ordinary(taxes)
    return max(0.0, end_value - tax), tax
