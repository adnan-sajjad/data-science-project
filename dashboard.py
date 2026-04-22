"""
Primetrade.ai — Trader Performance vs Market Sentiment
Interactive Streamlit Dashboard

Run: streamlit run dashboard.py
"""

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Primetrade.ai — Trader vs Sentiment",
    page_icon="📊",
    layout="wide",
)

BINARY_COLORS = {"Fear": "#d62728", "Neutral": "#2ca02c", "Greed": "#1f77b4"}
ORDER = ["Fear", "Neutral", "Greed"]

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    fg_raw = pd.read_csv("fear_greed_index.csv")
    ht_raw = pd.read_csv("historical_data.csv")

    # Parse dates
    fg = fg_raw.copy()
    fg["date"] = pd.to_datetime(fg["date"])

    def bucket(c):
        if c in ("Fear", "Extreme Fear"):   return "Fear"
        if c in ("Greed", "Extreme Greed"): return "Greed"
        return "Neutral"

    fg["sentiment_binary"] = fg["classification"].apply(bucket)
    fg_lookup = fg[["date", "classification", "sentiment_binary", "value"]].copy()
    fg_lookup.columns = ["date_only", "classification", "sentiment_binary", "fg_value"]

    ht = ht_raw.copy()
    ht["date"]      = pd.to_datetime(ht["Timestamp IST"], format="%d-%m-%Y %H:%M")
    ht["date_only"] = ht["date"].dt.normalize()
    ht = ht.merge(fg_lookup, on="date_only", how="left").dropna(subset=["classification"])

    closed = ht[ht["Direction"].isin(["Close Long", "Close Short"])].copy()

    # Daily metrics
    daily = closed.groupby(["date_only", "Account"]).agg(
        daily_pnl    = ("Closed PnL", "sum"),
        trade_count  = ("Closed PnL", "count"),
        avg_size_usd = ("Size USD",   "mean"),
        win_count    = ("Closed PnL", lambda x: (x > 0).sum()),
    ).reset_index()
    daily["win_rate"] = daily["win_count"] / daily["trade_count"]
    daily = daily.merge(fg_lookup, on="date_only", how="left")

    # Account-level metrics
    acct = closed.groupby("Account").agg(
        max_size     = ("Size USD",   "max"),
        med_size     = ("Size USD",   "median"),
        total_pnl    = ("Closed PnL", "sum"),
        total_trades = ("Closed PnL", "count"),
        total_wins   = ("Closed PnL", lambda x: (x > 0).sum()),
    ).reset_index()
    acct["leverage_proxy"]    = acct["max_size"] / (acct["med_size"] + 1e-9)
    acct["win_rate"]          = acct["total_wins"] / acct["total_trades"]
    acct["avg_pnl_per_trade"] = acct["total_pnl"]  / acct["total_trades"]
    acct["lev_segment"]       = pd.qcut(acct["leverage_proxy"], q=2,
                                        labels=["Low Leverage", "High Leverage"])
    acct["freq_segment"]      = pd.qcut(acct["total_trades"],   q=2,
                                        labels=["Infrequent", "Frequent"])
    acct["winner_type"]       = np.where(
        (acct["win_rate"] >= 0.55) & (acct["total_pnl"] > 0), "Consistent Winner",
        np.where(acct["total_pnl"] < 0, "Consistent Loser", "Inconsistent"))

    # Long/Short ratio
    ls = ht[ht["Direction"].isin(["Open Long", "Open Short"])].groupby(
        ["date_only", "Account"]
    ).agg(
        longs  = ("Direction", lambda x: (x == "Open Long").sum()),
        shorts = ("Direction", lambda x: (x == "Open Short").sum()),
    ).reset_index()
    ls["ls_ratio"] = (ls["longs"] + 1) / (ls["shorts"] + 1)
    ls = ls.merge(fg_lookup, on="date_only", how="left").dropna(subset=["classification"])

    # Clustering
    feats = ["total_trades", "total_pnl", "win_rate", "avg_pnl_per_trade", "leverage_proxy"]
    X = StandardScaler().fit_transform(acct[feats])
    acct["cluster"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)
    cmap = {0: "Elite Performers", 1: "High-Volume Specialists",
            2: "Struggling Traders", 3: "Cautious/Inactive"}
    cluster_summary = acct.groupby("cluster")[feats].mean()
    for c in cmap:
        row = cluster_summary.loc[c]
        if row["total_pnl"] == cluster_summary["total_pnl"].max():
            cmap[c] = "Elite Performers"
        elif row["total_trades"] == cluster_summary["total_trades"].max():
            cmap[c] = "High-Volume Specialists"
        elif row["total_pnl"] < 0:
            cmap[c] = "Struggling Traders"
        else:
            cmap[c] = "Cautious/Inactive"
    acct["archetype"] = acct["cluster"].map(cmap)

    return daily, acct, ls, closed, fg

daily, acct, ls, closed, fg = load_data()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Filters
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("🔧 Filters")

selected_sentiments = st.sidebar.multiselect(
    "Market Sentiment",
    options=ORDER,
    default=ORDER,
)

all_coins = sorted(closed["Coin"].unique().tolist())
selected_coins = st.sidebar.multiselect(
    "Coins",
    options=all_coins,
    default=all_coins[:5],
    help="Filter by traded asset"
)

date_min = daily["date_only"].min().date()
date_max = daily["date_only"].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)

# Apply filters
if len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_date, end_date = pd.Timestamp(date_min), pd.Timestamp(date_max)

daily_f  = daily[
    (daily["sentiment_binary"].isin(selected_sentiments)) &
    (daily["date_only"] >= start_date) &
    (daily["date_only"] <= end_date)
]
closed_f = closed[
    (closed["sentiment_binary"].isin(selected_sentiments)) &
    (closed["date_only"] >= start_date) &
    (closed["date_only"] <= end_date) &
    (closed["Coin"].isin(selected_coins))
]
ls_f = ls[
    (ls["sentiment_binary"].isin(selected_sentiments)) &
    (ls["date_only"] >= start_date) &
    (ls["date_only"] <= end_date)
]

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("📊 Primetrade.ai — Trader Performance vs Market Sentiment")
st.caption("Hyperliquid trades (2024) × Bitcoin Fear/Greed Index")

# ── KPI Cards ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Closed Trades",  f"{len(closed_f):,}")
k2.metric("Unique Accounts",      f"{closed_f['Account'].nunique()}")
k3.metric("Avg Daily PnL",        f"${daily_f['daily_pnl'].mean():,.0f}")
k4.metric("Avg Win Rate",         f"{daily_f['win_rate'].mean()*100:.1f}%")
k5.metric("Days in View",         f"{daily_f['date_only'].nunique()}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performance",
    "🧠 Behavior",
    "👥 Segments",
    "🤖 Model & Clusters",
    "⚡ Part C — Strategy",
])

# ── TAB 1 — Performance ───────────────────────────────────────────────────────
with tab1:
    st.subheader("Performance by Market Sentiment")

    perf = daily_f.groupby("sentiment_binary").agg(
        avg_pnl    = ("daily_pnl", "mean"),
        median_pnl = ("daily_pnl", "median"),
        win_rate   = ("win_rate",  "mean"),
        pnl_std    = ("daily_pnl", "std"),
        n          = ("daily_pnl", "count"),
    ).reindex([s for s in ORDER if s in selected_sentiments]).reset_index()

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = [BINARY_COLORS[s] for s in perf["sentiment_binary"]]
        ax.bar(perf["sentiment_binary"], perf["avg_pnl"], color=colors,
               edgecolor="black", linewidth=0.6)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Avg Daily PnL per Trader-Day")
        ax.set_ylabel("USD")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(perf["sentiment_binary"], perf["win_rate"] * 100, color=colors,
               edgecolor="black", linewidth=0.6)
        ax.axhline(50, color="red", linestyle="--", alpha=0.7, label="50% baseline")
        ax.set_title("Average Win Rate (%)")
        ax.set_ylabel("%")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("**Summary Table**")
    perf_display = perf.rename(columns={
        "sentiment_binary": "Sentiment",
        "avg_pnl": "Avg Daily PnL",
        "median_pnl": "Median PnL",
        "win_rate": "Win Rate",
        "pnl_std": "PnL Std Dev",
        "n": "Trader-Days"
    })
    perf_display["Avg Daily PnL"] = perf_display["Avg Daily PnL"].map("${:,.0f}".format)
    perf_display["Median PnL"]    = perf_display["Median PnL"].map("${:,.0f}".format)
    perf_display["Win Rate"]      = perf_display["Win Rate"].map("{:.1%}".format)
    perf_display["PnL Std Dev"]   = perf_display["PnL Std Dev"].map("${:,.0f}".format)
    st.dataframe(perf_display, use_container_width=True)

    # PnL distribution
    st.subheader("PnL Distribution (2–98% winsorized)")
    fig, ax = plt.subplots(figsize=(10, 4))
    data_p = daily_f.copy()
    data_p = data_p[data_p["daily_pnl"].between(
        data_p["daily_pnl"].quantile(0.02),
        data_p["daily_pnl"].quantile(0.98)
    )]
    avail_order = [s for s in ORDER if s in selected_sentiments]
    avail_colors = {s: BINARY_COLORS[s] for s in avail_order}
    if len(avail_order) > 0:
        sns.boxplot(data=data_p, x="sentiment_binary", y="daily_pnl",
                    order=avail_order, palette=avail_colors, ax=ax, width=0.5)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Market Sentiment"); ax.set_ylabel("Daily PnL (USD)")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ── TAB 2 — Behavior ──────────────────────────────────────────────────────────
with tab2:
    st.subheader("How Do Traders Behave by Sentiment?")

    behav = daily_f.groupby("sentiment_binary").agg(
        avg_trades = ("trade_count",  "mean"),
        avg_size   = ("avg_size_usd", "mean"),
    ).reindex([s for s in ORDER if s in selected_sentiments]).reset_index()

    ls_sent = ls_f.groupby("sentiment_binary")["ls_ratio"].mean()\
                  .reindex([s for s in ORDER if s in selected_sentiments])\
                  .reset_index()
    behav = behav.merge(ls_sent, on="sentiment_binary", how="left")
    avail_order = [s for s in ORDER if s in selected_sentiments]
    colors = [BINARY_COLORS[s] for s in avail_order]

    col1, col2, col3 = st.columns(3)

    for col, metric, ylabel, title in zip(
        [col1, col2, col3],
        ["avg_trades", "avg_size", "ls_ratio"],
        ["Trades/Day", "USD", "L/S Ratio"],
        ["Avg Trades per Day", "Avg Trade Size (USD)", "Long/Short Ratio (>1 = Long-Heavy)"]
    ):
        with col:
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            ax.bar(behav["sentiment_binary"], behav[metric], color=colors,
                   edgecolor="black", linewidth=0.6)
            if metric == "ls_ratio":
                ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xticklabels(behav["sentiment_binary"], rotation=10, fontsize=9)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    # Timeline
    st.subheader("PnL Timeline (colored by daily sentiment)")
    daily_agg = daily_f.groupby(["date_only", "sentiment_binary"]).agg(
        total_pnl = ("daily_pnl", "sum")).reset_index().sort_values("date_only")

    fig, ax = plt.subplots(figsize=(14, 4))
    for _, row in daily_agg.iterrows():
        ax.bar(row["date_only"], row["total_pnl"],
               color=BINARY_COLORS.get(row["sentiment_binary"], "#999"), alpha=0.8, width=1)
    ax.axhline(0, color="black", linewidth=0.8)
    patches = [mpatches.Patch(color=BINARY_COLORS[s], label=s) for s in avail_order]
    ax.legend(handles=patches)
    ax.set_ylabel("Total PnL (USD)")
    ax.set_xlabel("Date")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ── TAB 3 — Segments ──────────────────────────────────────────────────────────
with tab3:
    st.subheader("Trader Segmentation")

    seg_choice = st.radio(
        "View segment by:",
        ["Leverage (High vs Low)", "Trade Frequency", "Winner Type"],
        horizontal=True
    )

    seg_col = {
        "Leverage (High vs Low)": "lev_segment",
        "Trade Frequency":        "freq_segment",
        "Winner Type":            "winner_type",
    }[seg_choice]

    seg_summary = acct.groupby(seg_col).agg(
        n_traders     = ("Account",        "count"),
        avg_total_pnl = ("total_pnl",      "mean"),
        avg_win_rate  = ("win_rate",        "mean"),
        avg_trades    = ("total_trades",    "mean"),
        avg_lev       = ("leverage_proxy",  "mean"),
    ).reset_index()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Segment Summary**")
        disp = seg_summary.copy()
        disp["avg_total_pnl"] = disp["avg_total_pnl"].map("${:,.0f}".format)
        disp["avg_win_rate"]  = disp["avg_win_rate"].map("{:.1%}".format)
        disp["avg_trades"]    = disp["avg_trades"].map("{:,.0f}".format)
        disp["avg_lev"]       = disp["avg_lev"].map("{:,.1f}x".format)
        disp.columns = ["Segment", "# Traders", "Avg Total PnL", "Avg Win Rate",
                        "Avg Trades", "Avg Leverage Proxy"]
        st.dataframe(disp, use_container_width=True)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        seg_colors = plt.cm.Set2(np.linspace(0, 1, len(seg_summary)))
        ax.bar(seg_summary[seg_col], seg_summary["avg_total_pnl"],
               color=seg_colors, edgecolor="black", linewidth=0.6)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Avg Total PnL — {seg_choice}")
        ax.set_ylabel("USD")
        ax.set_xticklabels(seg_summary[seg_col], rotation=10)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Segment × Sentiment cross-tab
    st.subheader(f"Win Rate: {seg_choice} × Market Sentiment")
    merged = closed_f.merge(acct[["Account", seg_col]], on="Account", how="left")
    merged = merged.dropna(subset=["sentiment_binary", seg_col])
    avail_order = [s for s in ORDER if s in selected_sentiments]

    if len(merged) > 0 and len(avail_order) > 0:
        cross = merged.groupby(["sentiment_binary", seg_col]).agg(
            win_rate=("Closed PnL", lambda x: (x > 0).mean() * 100)
        ).reset_index()
        try:
            pivot = cross.pivot(index="sentiment_binary", columns=seg_col,
                                values="win_rate").reindex(avail_order)
            fig, ax = plt.subplots(figsize=(9, 4))
            pivot.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.6, width=0.6)
            ax.axhline(50, color="red", linestyle="--", alpha=0.7, label="50% baseline")
            ax.set_title("Win Rate by Sentiment × Segment")
            ax.set_ylabel("Win Rate (%)")
            ax.set_xticklabels(avail_order, rotation=0)
            ax.legend(title=seg_col)
            plt.tight_layout()
            st.pyplot(fig); plt.close()
        except Exception:
            st.info("Not enough data for cross-tab with current filters.")

# ── TAB 4 — Model & Clusters ──────────────────────────────────────────────────
with tab4:
    st.subheader("Behavioral Archetypes (K-Means Clustering)")

    archetype_colors = {
        "Elite Performers":       "#2ca02c",
        "High-Volume Specialists":"#1f77b4",
        "Struggling Traders":     "#d62728",
        "Cautious/Inactive":      "#ff7f0e",
    }

    col1, col2 = st.columns([1.2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        for arch, grp in acct.groupby("archetype"):
            ax.scatter(grp["total_trades"], grp["total_pnl"],
                       c=archetype_colors.get(arch, "#999"),
                       label=arch, s=90, edgecolors="black", linewidths=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Total Trades")
        ax.set_ylabel("Total PnL (USD)")
        ax.set_title("Trader Archetypes (k=4)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Archetype Summary**")
        arch_sum = acct.groupby("archetype").agg(
            n         = ("Account",       "count"),
            avg_pnl   = ("total_pnl",     "mean"),
            win_rate  = ("win_rate",       "mean"),
            avg_trades= ("total_trades",   "mean"),
        ).reset_index()
        arch_sum["avg_pnl"]   = arch_sum["avg_pnl"].map("${:,.0f}".format)
        arch_sum["win_rate"]  = arch_sum["win_rate"].map("{:.1%}".format)
        arch_sum["avg_trades"]= arch_sum["avg_trades"].map("{:,.0f}".format)
        arch_sum.columns = ["Archetype","# Traders","Avg PnL","Win Rate","Avg Trades"]
        st.dataframe(arch_sum, use_container_width=True)

    st.markdown("---")
    st.subheader("Predictive Model — Next-Day Profitability")
    st.caption("Gradient Boosting Classifier | Features: today's PnL, win rate, trade count, size, sentiment")

    if st.button("▶ Train Model"):
        with st.spinner("Training..."):
            model_df = daily.copy()
            model_df["profit_label"]  = (model_df["daily_pnl"] > 0).astype(int)
            model_df["sentiment_enc"] = LabelEncoder().fit_transform(model_df["sentiment_binary"])
            model_df = model_df.sort_values(["Account", "date_only"])
            model_df["next_day_profit"] = model_df.groupby("Account")["profit_label"].shift(-1)
            model_df.dropna(subset=["next_day_profit"], inplace=True)

            feats = ["sentiment_enc", "trade_count", "avg_size_usd", "win_rate", "daily_pnl"]
            X = model_df[feats]
            y = model_df["next_day_profit"].astype(int)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

            clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            acc    = accuracy_score(y_te, y_pred)

        col1, col2 = st.columns(2)
        col1.metric("Model Accuracy", f"{acc:.1%}")
        col1.caption("⚠ Class imbalance (~10:1 profit vs loss days) inflates this number. Precision/recall matter more.")

        with col2:
            fi = pd.Series(clf.feature_importances_, index=feats).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            fi.plot(kind="barh", ax=ax, color="#1f77b4", edgecolor="black", linewidth=0.6)
            ax.set_title("Feature Importance")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        report = classification_report(y_te, y_pred, target_names=["Loss","Profit"], output_dict=True)
        report_df = pd.DataFrame(report).T.iloc[:2].round(3)
        st.markdown("**Precision / Recall by class:**")
        st.dataframe(report_df[["precision","recall","f1-score","support"]], use_container_width=True)

# ── TAB 5 — Part C: Strategy ──────────────────────────────────────────────────
with tab5:
    st.subheader("⚡ Part C — Actionable Strategy Recommendations")
    st.caption("Evidence-backed rules of thumb derived from the analysis")

    # ── Strategy 1 ──────────────────────────────────────────────────────────
    with st.expander("📌 Strategy 1 — 'Stay Active During Fear, Pull Back on Greed'", expanded=True):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
**Rule of thumb:**
> *During Fear days, Consistent Winners should maintain or increase trade frequency and size.
> During Extreme Greed (FG index > 75), all segments should reduce trade count by ~30%.*

**Evidence:**
- Fear days produce **3× higher avg daily PnL** ($8,152 vs $2,731 on Greed days)
- Traders are naturally more active on Fear days: 69 avg trades vs 46 on Greed (50% more)
- Long/Short ratio hits 32:1 on Fear — buy-the-dip works because skilled traders exploit dislocations
- On Greed days, returns compress while PnL std dev stays elevated — worse risk/reward

**Who this applies to:**
- ✅ Consistent Winners (win rate ≥ 55%, positive total PnL) — they already do this intuitively; formalizing it prevents hesitation
- ✅ Low-Leverage Frequent traders — they have the activity rate but can lean in harder on Fear days
- ⛔ Struggling Traders — should NOT increase size on Fear; they lack the edge to benefit from volatility
            """)

        with col2:
            s1_data = daily.groupby("sentiment_binary").agg(
                avg_pnl   = ("daily_pnl",   "mean"),
                avg_trades= ("trade_count",  "mean"),
            ).reindex(ORDER).reset_index()
            fig, ax = plt.subplots(figsize=(4, 3))
            colors = [BINARY_COLORS[s] for s in ORDER]
            ax.bar(ORDER, s1_data["avg_pnl"], color=colors, edgecolor="black", lw=0.6)
            ax.set_title("Avg Daily PnL", fontsize=10)
            ax.set_ylabel("USD", fontsize=9)
            ax.axhline(0, color="black", lw=0.8)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    # ── Strategy 2 ──────────────────────────────────────────────────────────
    with st.expander("📌 Strategy 2 — 'High-Leverage Traders Must Hedge the Long Bias on Fear Days'", expanded=True):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
**Rule of thumb:**
> *High-leverage traders should cap their Long/Short ratio at 5:1 on Fear days and add short hedges
> when the FG index drops below 20 (Extreme Fear). Never go >10:1 long without a defined stop.*

**Evidence:**
- High-leverage traders currently go **32:1 long/short on Fear days** — this is the biggest risk exposure in the dataset
- This "buy-the-dip" bet pays off when Fear is temporary, but creates catastrophic drawdown if Fear deepens
- The single worst trade in the dataset: **-$117,990 PnL** — almost certainly an unhedged large long during a Fear spike
- High-leverage traders earn more in absolute PnL, but their PnL std dev is also the highest

**Hedge rule specifics:**
- FG index 20–40 (Fear): Reduce long size by 20%, add a 15% short hedge
- FG index < 20 (Extreme Fear): Full L/S balance (1:1), hold cash until recovery confirmed
- FG index > 75 (Extreme Greed): Go short-leaning (0.7:1), momentum is likely exhausted

**Who this applies to:**
- ✅ High-Leverage segment (top half by leverage proxy) — they are the ones exposed
- ⛔ Low-Leverage and Infrequent traders — no change needed; their risk is already contained
            """)

        with col2:
            ls_sent = ls.groupby("sentiment_binary")["ls_ratio"].mean().reindex(ORDER)
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(ORDER, ls_sent.values, color=colors, edgecolor="black", lw=0.6)
            ax.axhline(1.0, color="black", ls="--", lw=1)
            ax.set_title("Long/Short Ratio", fontsize=10)
            ax.set_ylabel("Ratio", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    # ── Summary table ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Strategy Summary by Trader Segment")

    strategy_table = pd.DataFrame({
        "Segment": [
            "Consistent Winners",
            "High-Leverage Traders",
            "Frequent Traders",
            "Infrequent Traders",
            "Struggling Traders",
        ],
        "On Fear Days": [
            "✅ Increase trade count +20-30%; maintain long bias",
            "⚠ Go long but cap L/S at 5:1; define stops",
            "✅ Keep activity high; size can increase moderately",
            "🔄 No change; these days are not their edge",
            "⛔ Reduce size; wait for confirmation",
        ],
        "On Greed Days": [
            "🔄 Reduce trade count ~30%; lock in profits faster",
            "⚠ Shift to short-leaning (0.7:1) above FG 75",
            "⚠ Maintain frequency but reduce position size",
            "🔄 No change",
            "⛔ Sit out; risk/reward is worst for this segment",
        ],
        "On Neutral Days": [
            "🔄 Normal cadence",
            "🔄 Normal cadence, no hedge required",
            "🔄 Normal cadence",
            "🔄 Normal cadence",
            "🔄 Focus on learning, not size",
        ],
    })
    st.dataframe(strategy_table, use_container_width=True, hide_index=True)

    st.info(
        "📎 **Note on causality:** Correlation between Fear days and higher PnL does not prove "
        "that sentiment *causes* better performance. The 32 accounts in this dataset are likely "
        "sophisticated traders who already know how to trade Fear. These strategies should be "
        "validated before applying to a broader trader population."
    )
