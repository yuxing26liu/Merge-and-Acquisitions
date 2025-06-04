import sys
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
PROJECTION_YEARS = 10  # Total forecast years
CONVERGE_START = 6     # Year to start converging to terminal growth
# if running inside a notebook (ipykernel) or got no __file__, skip argparse
if 'ipykernel' in sys.argv[0] or '__file__' not in globals():
    class DCFArgs:
        def __init__(self, ticker, TerminalGrowthRate=0.03):
            self.ticker = ticker
            self.TerminalGrowthRate = TerminalGrowthRate
    # default values for notebook/demo
    args = DCFArgs(ticker="AAPL", TerminalGrowthRate=0.03)
else:
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--ticker", required=True,
                        help="Ticker symbol for the company (e.g. AAPL)")
    parser.add_argument("--TGR", dest="TerminalGrowthRate", type=float, default=0.03,
                        help="Terminal growth rate (default 3%)")
    args = parser.parse_args()

def dynamic_converger(current, expected, number_of_steps, period_to_begin_to_converge):
    n, p = int(number_of_steps), int(period_to_begin_to_converge)
    phase1 = np.full(p, current)
    phase2 = np.linspace(current, expected, n-p+1)
    return pd.Series(np.concatenate([phase1, phase2[1:]]), index=range(n))

class DCFModel:
    def __init__(self, ticker, args):
        self.ticker = ticker
        self.stock  = yf.Ticker(ticker)

        # fetch last 4 years revenue, oldest→newest
        inc           = self.stock.income_stmt
        self.rev_years = inc.loc["Total Revenue"][:4][::-1].astype(float)

        # cap terminal growth
        self.TerminalGrowthRate = min(args.TerminalGrowthRate, 0.035)

        info = self.stock.info
        self.beta         = info.get("beta", 1.28)
        self.RiskFree     = 0.043
        self.MarketReturn = 0.08
        self.MarketCap    = info.get("marketCap", 0)
        self.debt         = info.get("totalDebt", 0)
        self.cash         = info.get("totalCash", 0)
        self.shares       = info.get("sharesOutstanding", 1)
        self.price        = info.get("previousClose", np.nan)

        # build forecast (fills in hist_cagr, analyst_g5, schedule, FutureFCF…)
        self._build_forecast()
        self.WACC            = self._WACC()
        self.TerminalValue   = self._CalcTerminalValue()
        self.EnterpriseValue = self._DiscountedCashFlow()
        self.ImpliedPrice    = self._CalcSharePrice()
        raw_beta = info.get("beta", 1.28)
        self.beta = np.clip(raw_beta, 0.5, 2.5)  # Constrain between 0.5-2.5

    def _build_forecast(self, override_growth=None):
        inc = self.stock.income_stmt
        cf  = self.stock.cash_flow

        # 1) historical 3-yr CAGR
        rev = self.rev_years.values
        n   = len(rev) - 1
        self.hist_cagr = (rev[-1]/rev[0])**(1/n) - 1

        # 2) pick starting growth
        if override_growth is not None:
            g5 = override_growth
        else:
            try:
                raw = self.stock.analyst_price_recommendations["Growth (Next 5 Years)"].iloc[0]
                g5  = float(str(raw).strip("%"))/100
            except:
                g5  = 0.07
        self.analyst_g5 = g5

        # 3) growth schedule
        sched = dynamic_converger(
            current=g5,
            expected=self.TerminalGrowthRate,
            number_of_steps=PROJECTION_YEARS,
            period_to_begin_to_converge=CONVERGE_START
        )
        self.schedule = sched

        # 4) project revenues
        base = self.rev_years.iloc[-1]
        self.FutureRevenue = pd.Series([
            base * np.prod(1 + sched[:i+1])
            for i in range(PROJECTION_YEARS)
        ])

        # 5) trending EBIT margin (last 3 yrs weighted 0.2,0.3,0.5)
        hist_margins = (inc.loc["EBIT"][:4].astype(float) / self.rev_years)
        last3        = hist_margins.iloc[-3:]
        trend_margin = np.average(last3, weights=[0.2,0.3,0.5])
        self.FutureEBIT  = self.FutureRevenue * trend_margin

        # 6) tax
        tax = inc.loc["Tax Provision"][:3].astype(float)[::-1]
        e   = inc.loc["EBIT"][:3].astype(float)[::-1]
        self.TaxRate   = (tax/e).mean()
        self.FutureTax = self.FutureEBIT * self.TaxRate

        # 7) D&A
        da   = abs(cf.loc["Depreciation And Amortization"][:4].astype(float))
        da_m = (da/self.rev_years).mean()
        self.FutureDA = self.FutureRevenue * da_m

        # 8) CapEx (20% haircut)
        capex   = abs(cf.loc["Capital Expenditure"][:4].astype(float))
        capex_m = (capex/self.rev_years).mean() * 0.8
        self.FutureCapEx = self.FutureRevenue * capex_m

        # 9) NWC
        nwc = abs(cf.loc["Change In Working Capital"][:4].astype(float))
        nwc_m = (nwc/self.rev_years).mean()
        self.FutureNWC = self.FutureRevenue * nwc_m

        # 10) Free Cash Flow
        self.FutureFCF = (
            self.FutureEBIT
            - self.FutureTax
            + self.FutureDA
            - self.FutureCapEx
            - self.FutureNWC
        )

    def _WACC(self):
        D, E = self.debt, self.MarketCap
        wd   = D/(D+E) if D+E else 0
        we   = E/(D+E) if D+E else 1
        kd   = 0.03
        ke   = self.RiskFree + self.beta*(self.MarketReturn - self.RiskFree)
        return wd*kd*(1-self.TaxRate) + we*ke

    def _CalcTerminalValue(self):
        g     = self.TerminalGrowthRate
        fcf_t = self.FutureFCF.iloc[-1]
        return fcf_t*(1+g)/(self.WACC - g)

    def _DiscountedCashFlow(self):
        dcfs = sum(
            self.FutureFCF.iloc[i] / ((1+self.WACC)**(i+1))
            for i in range(PROJECTION_YEARS)
        )
        tv = self.TerminalValue/((1+self.WACC)**PROJECTION_YEARS)
        return dcfs + tv

    def _CalcSharePrice(self):
        # enterprise value from our DCF
        ev  = self.EnterpriseValue
        # equity value = EV minus debt plus cash
        eq  = ev - self.debt + self.cash

        # try to pull the most recent weighted-average diluted shares
        try:
            df_cf = self.stock.cash_flow
            wavg  = df_cf.loc["Weighted Average Shares Outstanding"].iloc[-1]
            shares_used = float(wavg)
        except Exception:
            shares_used = float(self.shares)

        return eq / shares_used
    
    def find_implied_growth(self, tol=1e-4, max_iter=50, low=0.0, high=0.30):
        target = self.price
        for _ in range(max_iter):
            mid = (low + high) / 2
            self._build_forecast(override_growth=mid)
            self.WACC            = self._WACC()
            self.TerminalValue   = self._CalcTerminalValue()
            self.EnterpriseValue = self._DiscountedCashFlow()
            implied = self._CalcSharePrice()
            if abs(implied - target)/target < tol:
                return mid
            if implied > target:
                high = mid
            else:
                low = mid
        return mid
    
    def replicate(self):
        # Create empty instance
        clone = self.__class__.__new__(self.__class__)
        # Copy only the “pure data” attributes
        for attr in (
            "FutureRevenue","FutureEBIT","FutureTax","FutureDA",
            "FutureCapEx","FutureNWC","FutureFCF",
            "WACC","TerminalValue","EnterpriseValue",
            "beta","RiskFree","MarketReturn","TerminalGrowthRate",
            "debt","cash","shares","price","TaxRate"
        ):
            setattr(clone, attr, getattr(self, attr).copy() if hasattr(getattr(self, attr), "copy") else getattr(self, attr))
        return clone


if __name__=="__main__":
    warnings.filterwarnings("ignore")
    model = DCFModel(args.ticker.upper(), args)

    # 1) print historical & analyst & terminal cap
    print(f"Historical 3-yr CAGR: {model.hist_cagr:6.2%}")
    print(f"Analyst 5-yr Growth:  {model.analyst_g5:6.2%}")
    print(f"Terminal Growth Cap:  {model.TerminalGrowthRate:6.2%}")
    print("Year-by-Year Growth Schedule:")
    for i, g in enumerate(model.schedule, start=1):
        print(f" Year {i:3d}: {g:6.2%}")
    print()

    # 2) implied price & MoS & MoS targets
    print(f"Implied Price:    ${model.ImpliedPrice:,.2f}")
    mos = (model.ImpliedPrice - model.price)/model.price
    print(f"MoS vs Market:    {mos: .2%}\n")
    for pct in (0.10, 0.20, 0.30):
        target = model.price * (1+pct)
        print(f" Target for {int(pct*100)}% MoS: ${target:,.2f}")
    print()

    # 3) discounted cash flows
    print("Yearly Discounted Cash Flows:")
    pv_list = []
    for i, fcf in enumerate(model.FutureFCF, start=1):
        pv = fcf/((1+model.WACC)**i)
        pv_list.append(pv)
        print(f" Year {i:3d}: FCF = ${fcf:,.0f}, PV = ${pv:,.0f}")
    pv_tv = model.TerminalValue/((1+model.WACC)**PROJECTION_YEARS)
    print(f"\nPV of Terminal Value: ${pv_tv:,.0f}\n")

    # 4) Standalone EV & Equity
    ev_standalone = sum(pv_list) + pv_tv
    eq_standalone = ev_standalone - model.debt + model.cash
    print(f"Standalone EV:    ${ev_standalone/1e9:,.1f}B")
    print(f"Standalone Equity:${eq_standalone/1e9:,.1f}B\n")

        # 5) back-solve for implied current growth
    ig = model.find_implied_growth()
    print(f"Implied “Current” Growth to justify ${model.price:,.2f}: {ig:,.2%}\n")

    # 6) **Rebuild** everything *with* that implied growth
    model._build_forecast(override_growth=ig)
    model.WACC            = model._WACC()
    model.TerminalValue   = model._CalcTerminalValue()
    model.EnterpriseValue = model._DiscountedCashFlow()
    model.ImpliedPrice    = model._CalcSharePrice()

    # 7) Print the summary *again*, now calibrated to market
    print("=== DCF WITH IMPLIED GROWTH ===")
    print(f"Implied Price:    ${model.ImpliedPrice:,.2f}")
    mos2 = (model.ImpliedPrice - model.price)/model.price
    print(f"MoS vs Market:    {mos2: .2%}\n")

    print("Yearly Discounted Cash Flows (calibrated):")
    pv_list2 = []
    for i, fcf in enumerate(model.FutureFCF, start=1):
        pv2 = fcf/((1+model.WACC)**i)
        pv_list2.append(pv2)
        print(f" Year {i:3d}: FCF = ${fcf:,.0f}, PV = ${pv2:,.0f}")
    pv_tv2 = model.TerminalValue/((1+model.WACC)**PROJECTION_YEARS)
    print(f"\nPV of Terminal Value: ${pv_tv2:,.0f}\n")

    ev2 = sum(pv_list2) + pv_tv2
    eq2 = ev2 - model.debt + model.cash
    print(f"Standalone EV:    ${ev2/1e9:,.1f}B")
    print(f"Standalone Equity:${eq2/1e9:,.1f}B")

