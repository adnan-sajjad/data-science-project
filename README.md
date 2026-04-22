# Primetrade.ai — Round-0 Assignment
## Trader Performance vs Market Sentiment (Fear/Greed Index)

---

## Setup & How to Run

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter nbformat
```

### Run the notebook
```bash
# Place both CSVs in the same directory as analysis.ipynb
# Then:
jupyter notebook analysis.ipynb
```

### Or run the script directly
```bash
python3 analysis.py
# Charts will be saved to ./charts/
```

---

## Methodology

**Data Sources**
- `fear_greed_index.csv` — 2,644 daily Fear/Greed readings (2018–2025), 5 classifications
- `historical_data.csv` — 211,224 Hyperliquid trades across 32 accounts (Jan–Dec 2024)

**Preparation Steps**
1. Loaded both datasets, confirmed zero missing values and zero duplicates
2. Parsed `Timestamp IST` (format: `DD-MM-YYYY HH:MM`) → extracted date
3. Filtered trader data to **closed trades only** (84,691 rows with real PnL) — open-position rows have PnL = 0 and would distort averages
4. Merged on date; only 6 rows (<0.01%) had no matching sentiment date — dropped cleanly
5. Collapsed 5 sentiment labels → 3 buckets: **Fear** (Fear + Extreme Fear), **Neutral**, **Greed** (Greed + Extreme Greed)
6. Built daily per-trader aggregates: PnL, win rate, trade count, avg size, long/short ratio

**Key Design Choices**
- Leverage is not directly in the data. Used `max_size_USD / median_size_USD` per account as a leverage proxy — captures traders who occasionally take very large positions relative to their typical size
- Long/Short ratio computed from `Open Long` vs `Open Short` directions on all trades (not just closed)
- PnL distribution charts winsorized at 2%–98% to handle outliers while preserving trend visibility

---

## Key Insights

### Insight 1 — Fear Days Outperform Greed Days (Counterintuitive)
- Average daily PnL per trader-day: **$8,152 on Fear days vs $2,731 on Greed days**
- Median also higher during Fear ($826 vs $540)
- Interpretation: Hyperliquid traders on this dataset are sophisticated enough to "buy the dip." Fear creates price dislocations that skilled traders exploit more effectively than during Greed-driven rallies where competition is highest and edges are narrower.

### Insight 2 — Traders Trade More and Bigger During Fear
- Avg trades/day: **69 on Fear days vs 46 on Greed days** (50% more active)
- Avg trade size: **$12,445 on Fear vs $7,595 on Greed** (64% larger)
- Long/Short ratio: **32x on Fear vs 11.5x on Greed** — traders go heavily long during Fear (buy-the-dip behavior), then become more balanced on Greed days

### Insight 3 — High Leverage Traders Earn More But Carry More Risk
- High leverage segment: avg total PnL **$289,394** vs Low leverage **$166,246**
- Win rates are nearly identical (84–85%) — the difference is position sizing, not skill
- Risk implication: high-leverage traders have higher variance; the PnL std dev is largest on Fear days, exactly when high-leverage traders are most active

### Insight 4 — Consistent Winners Are a Distinct Group
- 18 of 32 accounts qualify as "Consistent Winners" (win rate ≥ 55% AND positive total PnL)
- These traders average **$259,747 total PnL** and 84% win rate vs Consistent Losers at **-$80,804** and 60% win rate
- Consistent Losers also trade less frequently (avg 1,310 trades vs 2,785) — suggesting they lack edge compounding over time

---

## Strategy Recommendations

### Strategy 1 — "Stay Active During Fear, Pull Back During Extreme Greed"
> *Applies to: Consistent Winners, Low-Leverage traders*

- During **Fear days**: maintain or increase trade frequency; these days yield 3× the PnL per trader-day
- During **Extreme Greed** (FG index > 75): reduce trade count by ~30%; returns compress and risk increases
- Rationale: Fear = price dislocations = better entry/exit points. Greed = crowded trades = narrower alpha

### Strategy 2 — "High-Leverage Traders Must Hedge Long Bias on Fear Days"
> *Applies to: High-Leverage segment*

- High-leverage traders go **32:1 long-to-short** on Fear days — this works when the market recovers, but creates catastrophic drawdown risk if Fear deepens into capitulation
- Rule: Cap L/S ratio at **5:1** on Fear days; add short hedges on positions >$50K when FG index < 20 (Extreme Fear territory)
- Rationale: The biggest loss events (-$117,990 single-trade drawdown in the data) likely occurred on exactly these unhedged long-heavy Fear positions

---

## Bonus

### Behavioral Clustering (K-Means, k=4)
Four archetypes emerge from clustering on total PnL, trades, win rate, leverage proxy, and avg PnL/trade:
- **Elite Performers** — high win rate (83–89%), positive PnL, moderate-high leverage
- **High-Volume Specialists** — very high trade count (8,000+), strong absolute PnL, lower avg PnL/trade
- **Cautious/Inactive** — very few trades, low PnL contribution
- **Struggling Traders** — below-average win rate, negative cumulative PnL

### Predictive Model (Gradient Boosting)
- Target: Will tomorrow be a profitable day for this trader? (binary)
- Accuracy: **89%** on held-out test set
- Top features: `daily_pnl` (momentum), `win_rate` (trader quality), `trade_count` (activity signal)
- Market sentiment contributes but is not the dominant predictor — trader-specific behavior matters more than macro sentiment alone

---

## File Structure
```
primetrade_analysis/
├── analysis.ipynb          # Main Jupyter notebook
├── analysis.py             # Standalone Python script
├── README.md               # This file
├── fear_greed_index.csv    # Input data (place here)
├── historical_data.csv     # Input data (place here)
└── charts/
    ├── B1_performance_by_sentiment.png
    ├── B2_behavior_by_sentiment.png
    ├── B3_trader_segments.png
    ├── Insight1_pnl_distribution.png
    ├── Insight2_winrate_sentiment_leverage.png
    ├── Insight3_pnl_activity_timeline.png
    ├── Insight4_longshort_by_sentiment.png
    ├── Bonus_clustering.png
    └── Bonus_feature_importance.png
```
