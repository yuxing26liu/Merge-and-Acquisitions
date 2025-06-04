# Mergers & Acquisitions (M&A) Overview
M&A describes how companies combine (merger) or one company buys another (acquisition) to accelerate growth, access new markets, acquire talent/technology, and capture cost or revenue synergies. A robust M&A process requires:

Identifying attractive targets (Which companies are most likely to be acquired?)

Valuing those targets accurately (What is their fair standalone price?)

Estimating deal synergies (How much extra value will the combined entity generate?)

I developed a three-model approach—drawing on Crunchbase data (via Crystalnix/crunchbase-ml and Kaggle’s Crunchbase dataset, and DCF training guidance from Wall Street Prep—to cover each step.

1. Acquisition-Likelihood Model (Machine Learning Classification)
Purpose: Given a company’s industry, geography, funding history, team credentials, and other characteristics, predict its probability of being acquired within a set timeframe.

Data Sources & Feature Engineering:

Ingested dozens of CSVs (Crunchbase profiles, funding rounds, acquisitions) into a single SQL schema, resolving mismatches (e.g., “target_name” vs. “acquired_company_name”).

Engineered features such as total funding, number of funding rounds, educational credentials, and geographic flags.

Core Technique: Used scikit-learn’s GradientBoostingClassifier to learn from historical M&A outcomes. After imputing missing values (custom RANSAC imputers for funding data) and one-hot encoding categorical fields, the model distinguishes “acquired” vs. “not acquired.”

Key Output: A probability score (0–100%) indicating how “M&A-friendly” a target is, which helps prioritize outreach and due diligence.

2. Discounted Cash Flow (DCF) Valuation Model
Purpose: Estimate a target’s standalone intrinsic value by forecasting its future free cash flows and discounting them to present value.

Methodology (inspired by Wall Street Prep’s six-step training):

Revenue Growth Projection: Project multi-phase growth (high-growth → transition → maturity) using beginning/end growth rates and convergence schedules.

Operating Margin & Tax Rate Projection: Simulate how margins and tax rates evolve from current to terminal values.

WACC Projection: Compute annual cost of equity (from projected betas and ERP) and cost of debt (from projected interest rates), combine with capital structure, and derive yearly discount factors.

Free Cash Flow Forecast: Calculate unlevered free cash flow as EBIT × (1–tax) minus reinvestment each year.

Terminal Value: In the final projection year, assume perpetual growth at a conservative terminal rate and calculate the terminal cash-flow multiple.

Present Value Calculation: Discount all forecasted cash flows and terminal value back to today.

Key Output: A detailed year-by-year cash-flow schedule, the target’s DCF value, implied equity value per share, and “margin of safety” versus current market price.

3. Synergy Value Adjustment (“With Synergies” Model)
Purpose: Quantify the incremental value a buyer gains by merging operations—e.g., cost savings (headcount rationalization, facility consolidation), revenue uplifts (cross-selling, expanded distribution), or tax benefits—on top of the target’s standalone DCF.

Approach:

Identify Synergy Drivers: Estimate specific cost or revenue synergy items (e.g., “$5 M annual cost reduction,” “$3 M incremental sales”).

Synergy Realization Timeline: Model when synergies ramp up (e.g., over 3 years) rather than assume immediate capture.

Discount & Add: Forecast the synergy cash flows, discount them at the buyer’s WACC, and add that present value to the standalone DCF.

Key Output:

A “With Synergies” implied price (what a strategic buyer might pay) and an updated margin of safety relative to market price.

How These Models Fit Together
Screening & Prioritization: Score hundreds of potential targets using the Acquisition-Likelihood Model to rank them by M&A attractiveness.

Valuation & Negotiation: Run a DCF Valuation on top candidates—using multi-phase growth, projected WACC, and terminal value—to establish a fair standalone price benchmark.

Deal Structuring & Offer Price: Layer in Synergy Analysis to capture incremental value. The “With Synergies” price often justifies a premium above the standalone DCF. The final offer typically lands between standalone DCF and synergies-adjusted value, accounting for integration risk and competitive dynamics.

By combining:

A data-driven ML model (who is likely to sell),

A rigorous financial model (what they’re worth on their own), and

A synergy calculator (what they’re worth to me),

