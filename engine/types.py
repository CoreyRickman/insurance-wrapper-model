from dataclasses import dataclass
from typing import Dict, Optional, Literal

Frequency = Literal["monthly", "annual"]

@dataclass
class TaxInputs:
    federal_ordinary: float
    federal_ltcg: float
    federal_stcg: float
    qualified_div: float
    niit: float
    state_ordinary: float
    state_ltcg: float
    state_stcg: float
    state_qualified_div: float
    include_niit: bool = True

@dataclass
class StrategyInputs:
    expected_return: float
    volatility: float

    ordinary_yield: float
    qualified_div_yield: float
    realized_stcg: float
    realized_ltcg: float
    unrealized_appreciation: float

    mgmt_fee_bps: float = 0.0

@dataclass
class PolicyInputs:
    premium: float
    issue_age: int
    premium_load: float = 0.0

@dataclass
class FeeInputs:
    wrapper_fee_bps: float
    fund_er_bps: float
    admin_fee_annual: float

@dataclass
class WithdrawalPlan:
    start_age: int
    end_age: int
    amount: float
    frequency: Frequency = "monthly"
    inflation: float = 0.0
    apply_59_5_penalty: bool = True
    penalty_rate: float = 0.10

@dataclass
class AnnuitizationPlan:
    annuitize_age: int
    payout_years: int
    pricing_rate: float
    payout_frequency: Frequency = "monthly"

@dataclass
class TaxableInputs:
    step_up_at_death: bool = True
    death_age: Optional[int] = None
    tlh_bps: float = 0.0

@dataclass
class PPLIInputs:
    db_option: Literal["A_level", "B_increasing"] = "A_level"
    corridor: float = 1.10

    premium_load: float = 0.0
    admin_fee_annual: float = 0.0
    asset_charge_bps: float = 0.0
    fund_er_bps: float = 0.0

    coi_bps_by_age: Dict[int, float] = None

    loan_start_age: Optional[int] = None
    loan_amount: float = 0.0
    loan_frequency: Frequency = "annual"
    loan_inflation: float = 0.0

    loan_interest_rate: float = 0.06
    loan_crediting_rate: float = 0.00
    is_mec: bool = False
