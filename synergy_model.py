
# synergy_model.py

import copy
from dcf3 import DCFModel  # adjust import path as needed

class SynergyModel:
    """
    Apply merger synergies to an existing DCFModel instance,
    producing a new DCFModel with adjusted cash flows and WACC.
    """
    def __init__(self, base_dcf: DCFModel):
        # deep-copy the entire DCFModel so we don't mutate the original
        self.base = base_dcf.replicate()

    def merged_wacc(self, target_dcf: DCFModel, new_debt_ratio: float):
        """Blend acquirer & target betas and debt ratios to get merged WACC."""
        acq, tgt = self.base, target_dcf
        # blend betas by equity value
        total_equity = acq.EnterpriseValue - acq.debt + acq.cash \
                     + tgt.EnterpriseValue - tgt.debt + tgt.cash
        blend_beta = (
            (acq.beta * (acq.EnterpriseValue - acq.debt + acq.cash)) +
            (tgt.beta * (tgt.EnterpriseValue - tgt.debt + tgt.cash))
        ) / total_equity

        rf = acq.RiskFree
        mr = acq.MarketReturn
        # cost of equity and debt as before
        ke = rf + blend_beta * (mr - rf)
        kd = 0.03  # you could parameterize or pull from base info
        wd = new_debt_ratio
        we = 1 - wd
        tax = self.base.TaxRate
        return wd * kd * (1 - tax) + we * ke

    def apply_synergies(
        self,
        cost_savings_pct: float = 0.0,
        revenue_boost_pct: float = 0.0,
        phase_in_years: int = 3,
        new_debt_ratio: float = None
    ) -> DCFModel:
        """
        Returns a new DCFModel with:
         - OpEx reduced by cost_savings_pct over phase_in_years
         - Revenue boosted linearly by revenue_boost_pct over phase_in_years
         - (optional) WACC recalculated with new debt ratio
        """
        sy = self.base

        # 1) Phase‐in schedules
        # for year i in 0..phase_in_years-1:
        cost_schedule = [(1 - cost_savings_pct * (i+1)/phase_in_years)
                         for i in range(phase_in_years)]
        rev_schedule  = [(1 + revenue_boost_pct * (i+1)/phase_in_years)
                         for i in range(phase_in_years)]

        # 2) Apply to first phase_in_years of projections
        for yr in range(len(sy.FutureRevenue)):
            if yr < phase_in_years:
                sy.FutureRevenue.iloc[yr] *= rev_schedule[yr]
                # assume OpEx = Revenue – EBIT – DA (we need an explicit OpEx series):
                # derive original OpEx from base, then scale
                op_ex = sy.FutureRevenue.iloc[yr] - sy.FutureEBIT.iloc[yr] - sy.FutureDA.iloc[yr]
                sy.FutureEBIT.iloc[yr] = (
                    sy.FutureRevenue.iloc[yr] - op_ex * cost_schedule[yr] - sy.FutureDA.iloc[yr]
                )
            # beyond phase_in_years, revenue and EBIT stay at converged levels

        # 3) Recalculate FCF & EV
        sy.FutureTax = sy.FutureEBIT * sy.TaxRate
        sy.FutureFCF = (
            sy.FutureEBIT
            - sy.FutureTax
            + sy.FutureDA
            - sy.FutureCapEx
            - sy.FutureNWC
        )
        sy.EnterpriseValue = sy._DiscountedCashFlow()

        # 4) Optionally update WACC if merger financing changes capital structure
        if new_debt_ratio is not None:
            sy.WACC = self.merged_wacc(self.base, new_debt_ratio)
            sy.EnterpriseValue = sy._DiscountedCashFlow()

        # 5) Re‐calc share price
        sy.ImpliedPrice = sy._CalcSharePrice()

        return sy
