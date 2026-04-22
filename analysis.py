"""
Primetrade.ai — Round-0 Assignment
Trader Performance vs Market Sentiment (Fear/Greed)
Author: Intern Candidate
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── aesthetics ─────────────────────────────────────────────────────────────────
PALETTE = {
    'Extreme Fear': '#d62728',
    'Fear':         '#ff7f0e',
    'Neutral':      '#2ca02c',
    'Greed':        '#1f77b4',
    'Extreme Greed':'#9467bd',
}
BINARY = {'Fear/Extreme Fear': '#d62728', 'Neutral': '#2ca02c',
          'Greed/Extreme Greed': '#1f77b4'}

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f9f9f9',
    'axes.grid':        True,
    'grid.alpha':       0.4,
    'font.size':        11,
})
SAVE = '/home/claude/primetrade_analysis/charts/'

# ══════════════════════════════════════════════════════════════════════════════
# PART A — DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PART A — DATA PREPARATION")
print("=" * 60)

# ── Load ───────────────────────────────────────────────────────────────────────
fg_raw = pd.read_csv('/mnt/user-data/uploads/fear_greed_index.csv')
ht_raw = pd.read_csv('/mnt/user-data/uploads/historical_data.csv')

print(f"\nFear/Greed  → rows: {len(fg_raw):,}  cols: {fg_raw.shape[1]}")
print(f"Trader Data → rows: {len(ht_raw):,}  cols: {ht_raw.shape[1]}")
print(f"\nMissing values — Fear/Greed:\n{fg_raw.isnull().sum().to_string()}")
print(f"\nMissing values — Trader Data:\n{ht_raw.isnull().sum().to_string()}")
print(f"\nDuplicates — Fear/Greed: {fg_raw.duplicated().sum()}")
print(f"Duplicates — Trader Data: {ht_raw.duplicated().sum()}")

# ── Clean Fear/Greed ───────────────────────────────────────────────────────────
fg = fg_raw.copy()
fg['date'] = pd.to_datetime(fg['date'])

# Binary sentiment (collapse 5 → 3 buckets)
def bucket(c):
    if c in ('Fear', 'Extreme Fear'):   return 'Fear'
    if c in ('Greed', 'Extreme Greed'): return 'Greed'
    return 'Neutral'

fg['sentiment_binary'] = fg['classification'].apply(bucket)

# ── Clean Trader Data ──────────────────────────────────────────────────────────
ht = ht_raw.copy()
ht['date'] = pd.to_datetime(ht['Timestamp IST'], format='%d-%m-%Y %H:%M')
ht['date_only'] = ht['date'].dt.normalize()

# Keep only closed trades for PnL analysis
closed = ht[ht['Direction'].isin(['Close Long', 'Close Short'])].copy()
print(f"\nClosed trades (PnL trades): {len(closed):,}")

# ── Merge ──────────────────────────────────────────────────────────────────────
fg_lookup = fg[['date', 'classification', 'sentiment_binary', 'value']].copy()
fg_lookup.columns = ['date_only', 'classification', 'sentiment_binary', 'fg_value']

closed = closed.merge(fg_lookup, on='date_only', how='left')
ht    = ht.merge(fg_lookup, on='date_only', how='left')
unmatched = closed['classification'].isnull().sum()
print(f"Unmatched dates after merge: {unmatched} "
      f"({100*unmatched/len(closed):.1f}%)")
closed.dropna(subset=['classification'], inplace=True)
ht.dropna(subset=['classification'], inplace=True)

# ── Key Metrics (daily per trader) ────────────────────────────────────────────
daily = closed.groupby(['date_only', 'Account']).agg(
    daily_pnl     = ('Closed PnL', 'sum'),
    trade_count   = ('Closed PnL', 'count'),
    avg_size_usd  = ('Size USD',   'mean'),
    win_count     = ('Closed PnL', lambda x: (x > 0).sum()),
    loss_count    = ('Closed PnL', lambda x: (x < 0).sum()),
).reset_index()

daily['win_rate'] = daily['win_count'] / daily['trade_count']

# Merge sentiment
daily = daily.merge(fg_lookup, on='date_only', how='left')

# Leverage proxy: Size USD / (avg execution price * avg size tokens)
# Use the ratio of close-trade size to position size as leverage proxy
# More practical: group by account, compute leverage as max_size / median_size
lev_proxy = closed.groupby('Account').agg(
    max_size   = ('Size USD', 'max'),
    med_size   = ('Size USD', 'median'),
    total_pnl  = ('Closed PnL', 'sum'),
    total_trades = ('Closed PnL', 'count'),
    total_wins = ('Closed PnL', lambda x: (x > 0).sum()),
).reset_index()
lev_proxy['leverage_proxy'] = lev_proxy['max_size'] / (lev_proxy['med_size'] + 1e-9)
lev_proxy['win_rate'] = lev_proxy['total_wins'] / lev_proxy['total_trades']
lev_proxy['avg_pnl_per_trade'] = lev_proxy['total_pnl'] / lev_proxy['total_trades']

print("\nAccount-level summary (top 5 by PnL):")
print(lev_proxy.nlargest(5, 'total_pnl')[['Account','total_pnl','win_rate',
                                          'total_trades','leverage_proxy']].to_string(index=False))

# Long/Short ratio per day
ls = ht[ht['Direction'].isin(['Open Long','Open Short'])].groupby(
    ['date_only','Account']
).agg(
    longs  = ('Direction', lambda x: (x == 'Open Long').sum()),
    shorts = ('Direction', lambda x: (x == 'Open Short').sum()),
).reset_index()
ls['ls_ratio'] = (ls['longs'] + 1) / (ls['shorts'] + 1)
ls = ls.merge(fg_lookup, on='date_only', how='left').dropna(subset=['classification'])

print("\nPART A COMPLETE ✓")

# ══════════════════════════════════════════════════════════════════════════════
# PART B — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART B — ANALYSIS")
print("=" * 60)

# ── B1 — PnL & Win Rate: Fear vs Greed ────────────────────────────────────────
print("\nB1 — Performance by sentiment")
perf = daily.groupby('sentiment_binary').agg(
    avg_daily_pnl   = ('daily_pnl', 'mean'),
    median_daily_pnl= ('daily_pnl', 'median'),
    avg_win_rate    = ('win_rate',  'mean'),
    pnl_std         = ('daily_pnl', 'std'),
    n_trader_days   = ('daily_pnl', 'count'),
).reset_index()
print(perf.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('B1 — Trader Performance by Market Sentiment', fontsize=14, fontweight='bold')

colors = [BINARY.get(s, '#999') for s in perf['sentiment_binary']]

# Avg daily PnL
axes[0].bar(perf['sentiment_binary'], perf['avg_daily_pnl'], color=colors, edgecolor='black', linewidth=0.6)
axes[0].set_title('Avg Daily PnL per Trader-Day')
axes[0].set_ylabel('USD')
axes[0].set_xticklabels(perf['sentiment_binary'], rotation=10)

# Win rate
axes[1].bar(perf['sentiment_binary'], perf['avg_win_rate']*100, color=colors, edgecolor='black', linewidth=0.6)
axes[1].set_title('Average Win Rate (%)')
axes[1].set_ylabel('%')
axes[1].axhline(50, color='red', linestyle='--', alpha=0.7, label='50% baseline')
axes[1].legend()
axes[1].set_xticklabels(perf['sentiment_binary'], rotation=10)

# PnL std (volatility proxy / drawdown proxy)
axes[2].bar(perf['sentiment_binary'], perf['pnl_std'], color=colors, edgecolor='black', linewidth=0.6)
axes[2].set_title('PnL Std Dev (Risk/Drawdown Proxy)')
axes[2].set_ylabel('USD')
axes[2].set_xticklabels(perf['sentiment_binary'], rotation=10)

plt.tight_layout()
plt.savefig(f'{SAVE}B1_performance_by_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Chart saved: B1_performance_by_sentiment.png")

# ── B2 — Behavior by Sentiment ─────────────────────────────────────────────────
print("\nB2 — Behavior changes by sentiment")
behav = daily.groupby('sentiment_binary').agg(
    avg_trades_per_day = ('trade_count', 'mean'),
    avg_size_usd       = ('avg_size_usd', 'mean'),
).reset_index()

ls_sent = ls.groupby('sentiment_binary').agg(
    avg_ls_ratio = ('ls_ratio', 'mean'),
).reset_index()

behav = behav.merge(ls_sent, on='sentiment_binary')
print(behav.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('B2 — Trader Behavior by Market Sentiment', fontsize=14, fontweight='bold')

for ax, col, ylabel, title in zip(
    axes,
    ['avg_trades_per_day', 'avg_size_usd', 'avg_ls_ratio'],
    ['Trades / Day', 'USD', 'Long/Short Ratio'],
    ['Avg Trades per Day', 'Avg Trade Size (USD)', 'Long/Short Bias (>1 = Long-heavy)']
):
    cols = [BINARY.get(s, '#999') for s in behav['sentiment_binary']]
    ax.bar(behav['sentiment_binary'], behav[col], color=cols, edgecolor='black', linewidth=0.6)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if col == 'avg_ls_ratio':
        ax.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Neutral')
        ax.legend()
    ax.set_xticklabels(behav['sentiment_binary'], rotation=10)

plt.tight_layout()
plt.savefig(f'{SAVE}B2_behavior_by_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Chart saved: B2_behavior_by_sentiment.png")

# ── B3 — Trader Segmentation ───────────────────────────────────────────────────
print("\nB3 — Trader Segmentation")

# Segment 1: High vs Low Leverage (proxy = max_size / med_size ratio)
lev_proxy['lev_segment'] = pd.qcut(
    lev_proxy['leverage_proxy'], q=2, labels=['Low Leverage', 'High Leverage'])

# Segment 2: Frequent vs Infrequent
lev_proxy['freq_segment'] = pd.qcut(
    lev_proxy['total_trades'], q=2, labels=['Infrequent', 'Frequent'])

# Segment 3: Consistent Winners vs Inconsistent
lev_proxy['winner_segment'] = np.where(
    (lev_proxy['win_rate'] >= 0.55) & (lev_proxy['total_pnl'] > 0),
    'Consistent Winner',
    np.where(lev_proxy['total_pnl'] < 0, 'Consistent Loser', 'Inconsistent')
)

print("\nSegment 1 — Leverage:")
print(lev_proxy.groupby('lev_segment')[['total_pnl','win_rate','avg_pnl_per_trade']].mean().round(2).to_string())

print("\nSegment 2 — Frequency:")
print(lev_proxy.groupby('freq_segment')[['total_pnl','win_rate','avg_pnl_per_trade']].mean().round(2).to_string())

print("\nSegment 3 — Winner Type:")
print(lev_proxy.groupby('winner_segment')[['total_pnl','win_rate','total_trades']].mean().round(2).to_string())

# Segment 1 chart
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('B3 — Trader Segments Comparison', fontsize=14, fontweight='bold')

seg1 = lev_proxy.groupby('lev_segment')[['total_pnl','win_rate']].mean().reset_index()
seg2 = lev_proxy.groupby('freq_segment')[['total_pnl','win_rate']].mean().reset_index()
seg3 = lev_proxy.groupby('winner_segment')[['total_pnl','win_rate']].mean().reset_index()

for ax, df, seg_col, title in zip(
    axes,
    [seg1, seg2, seg3],
    ['lev_segment', 'freq_segment', 'winner_segment'],
    ['High vs Low Leverage\n(Total PnL)', 'Frequent vs Infrequent\n(Total PnL)', 'Winner Segments\n(Total PnL)']
):
    ax.bar(df[seg_col], df['total_pnl'], color=['#1f77b4','#ff7f0e','#2ca02c'][:len(df)],
           edgecolor='black', linewidth=0.6)
    ax.set_title(title)
    ax.set_ylabel('Avg Total PnL (USD)')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticklabels(df[seg_col], rotation=10)

plt.tight_layout()
plt.savefig(f'{SAVE}B3_trader_segments.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Chart saved: B3_trader_segments.png")

# ── Insight Charts ─────────────────────────────────────────────────────────────
print("\nInsight Charts")

# Insight 1 — PnL distribution: Fear vs Greed (box plot)
fig, ax = plt.subplots(figsize=(10, 6))
order = ['Fear', 'Neutral', 'Greed']
palette = {'Fear': '#d62728', 'Neutral': '#2ca02c', 'Greed': '#1f77b4'}
data_plot = daily[daily['sentiment_binary'].isin(order)]
data_plot = data_plot[data_plot['daily_pnl'].between(
    data_plot['daily_pnl'].quantile(0.02),
    data_plot['daily_pnl'].quantile(0.98)
)]
sns.boxplot(data=data_plot, x='sentiment_binary', y='daily_pnl', order=order,
            palette=palette, ax=ax, width=0.5, linewidth=1.2)
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_title('Insight 1 — Daily PnL Distribution: Fear vs Greed\n(2%–98% winsorized)', fontsize=13)
ax.set_xlabel('Market Sentiment')
ax.set_ylabel('Daily PnL (USD)')
plt.tight_layout()
plt.savefig(f'{SAVE}Insight1_pnl_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Chart saved: Insight1_pnl_distribution.png")

# Insight 2 — Win rate by sentiment + leverage segment
merged = closed.merge(lev_proxy[['Account','lev_segment']], on='Account', how='left')
merged = merged.dropna(subset=['sentiment_binary','lev_segment'])
ins2 = merged.groupby(['sentiment_binary','lev_segment']).agg(
    win_rate = ('Closed PnL', lambda x: (x > 0).mean() * 100),
    n        = ('Closed PnL', 'count')
).reset_index()

fig, ax = plt.subplots(figsize=(9, 5))
pivot = ins2.pivot(index='sentiment_binary', columns='lev_segment', values='win_rate')
pivot = pivot.reindex(['Fear','Neutral','Greed'])
pivot.plot(kind='bar', ax=ax, color=['#1f77b4','#ff7f0e'], edgecolor='black', linewidth=0.6, width=0.6)
ax.axhline(50, color='red', linestyle='--', alpha=0.7, label='50% baseline')
ax.set_title('Insight 2 — Win Rate by Sentiment × Leverage Segment', fontsize=13)
ax.set_ylabel('Win Rate (%)')
ax.set_xlabel('Market Sentiment')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Leverage')
plt.tight_layout()
plt.savefig(f'{SAVE}Insight2_winrate_sentiment_leverage.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Chart saved: Insight2_winrate_sentiment_leverage.png")

# Insight 3 — Trade frequency & size by sentiment over time
daily_agg = daily.groupby(['date_only','sentiment_binary']).agg(
    total_pnl   = ('daily_pnl',  'sum'),
    avg_trades  = ('trade_count','mean'),
).reset_index().sort_values('date_only')

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle('Insight 3 — Aggregate PnL & Activity Over Time\n(colored by daily sentiment)',
             fontsize=13, fontweight='bold')

sent_colors = {'Fear': '#d62728', 'Neutral': '#2ca02c', 'Greed': '#1f77b4'}
for _, row in daily_agg.iterrows():
    c = sent_colors.get(row['sentiment_binary'], '#999')
    axes[0].bar(row['date_only'], row['total_pnl'], color=c, alpha=0.7, width=1)
    axes[1].bar(row['date_only'], row['avg_trades'], color=c, alpha=0.7, width=1)

axes[0].set_ylabel('Total PnL (USD)')
axes[0].axhline(0, color='black', linewidth=0.8)
axes[1].set_ylabel('Avg Trades/Account')
axes[1].set_xlabel('Date')

patches = [mpatches.Patch(color=v, label=k) for k, v in sent_colors.items()]
axes[0].legend(handles=patches, title='Sentiment')
plt.tight_layout()
plt.savefig(f'{SAVE}Insight3_pnl_activity_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Chart saved: Insight3_pnl_activity_timeline.png")

# Insight 4 — Long/Short bias by sentiment
fig, ax = plt.subplots(figsize=(8, 5))
ls_agg = ls.groupby('sentiment_binary')['ls_ratio'].mean().reindex(['Fear','Neutral','Greed'])
ls_agg.plot(kind='bar', ax=ax, color=['#d62728','#2ca02c','#1f77b4'], edgecolor='black', linewidth=0.6)
ax.axhline(1.0, color='black', linestyle='--', linewidth=1, label='L/S = 1 (balanced)')
ax.set_title('Insight 4 — Long/Short Ratio by Sentiment\n(>1 = long-heavy; <1 = short-heavy)', fontsize=13)
ax.set_ylabel('Avg Long/Short Ratio')
ax.set_xticklabels(['Fear','Neutral','Greed'], rotation=0)
ax.legend()
plt.tight_layout()
plt.savefig(f'{SAVE}Insight4_longshort_by_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Chart saved: Insight4_longshort_by_sentiment.png")

print("\nPART B COMPLETE ✓")

# ══════════════════════════════════════════════════════════════════════════════
# BONUS — PREDICTIVE MODEL + CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("BONUS — PREDICTIVE MODEL + CLUSTERING")
print("=" * 60)

# ── Clustering — behavioral archetypes ────────────────────────────────────────
print("\nClustering traders into behavioral archetypes")
cluster_features = lev_proxy[['total_trades','total_pnl','win_rate',
                               'avg_pnl_per_trade','leverage_proxy']].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_features)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
lev_proxy['cluster'] = kmeans.fit_predict(X_scaled)

cluster_summary = lev_proxy.groupby('cluster').agg(
    n_traders         = ('Account', 'count'),
    avg_pnl           = ('total_pnl', 'mean'),
    avg_win_rate      = ('win_rate', 'mean'),
    avg_trades        = ('total_trades', 'mean'),
    avg_lev_proxy     = ('leverage_proxy', 'mean'),
).round(2)

# Name clusters
archetype_map = {}
for c, row in cluster_summary.iterrows():
    if row['avg_pnl'] > 5000 and row['avg_win_rate'] > 0.5:
        archetype_map[c] = 'Elite Performers'
    elif row['avg_trades'] > cluster_summary['avg_trades'].median() and row['avg_pnl'] < 0:
        archetype_map[c] = 'Overtraders'
    elif row['avg_lev_proxy'] > cluster_summary['avg_lev_proxy'].median() and row['avg_pnl'] < 0:
        archetype_map[c] = 'Risky Gamblers'
    else:
        archetype_map[c] = 'Cautious/Inactive'

lev_proxy['archetype'] = lev_proxy['cluster'].map(archetype_map)
cluster_summary['archetype'] = cluster_summary.index.map(archetype_map)
print(cluster_summary.to_string())

# Cluster scatter
fig, ax = plt.subplots(figsize=(10, 6))
arch_colors = {
    'Elite Performers': '#2ca02c',
    'Overtraders':      '#ff7f0e',
    'Risky Gamblers':   '#d62728',
    'Cautious/Inactive':'#1f77b4',
}
for arch, grp in lev_proxy.groupby('archetype'):
    ax.scatter(grp['total_trades'], grp['total_pnl'],
               c=arch_colors.get(arch,'#999'), label=arch, s=80, edgecolors='black', linewidth=0.5)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('Bonus — Trader Behavioral Archetypes (K-Means Clustering)', fontsize=13)
ax.set_xlabel('Total Trades')
ax.set_ylabel('Total PnL (USD)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{SAVE}Bonus_clustering.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Chart saved: Bonus_clustering.png")

# ── Predictive Model — next-day profitability ─────────────────────────────────
print("\nPredictive model — next-day profitability bucket")

# Build feature set at daily level
model_df = daily.copy()
model_df['profit_label'] = (model_df['daily_pnl'] > 0).astype(int)
model_df['sentiment_enc'] = LabelEncoder().fit_transform(model_df['sentiment_binary'])

model_df = model_df.sort_values(['Account','date_only'])
model_df['next_day_profit'] = model_df.groupby('Account')['profit_label'].shift(-1)
model_df.dropna(subset=['next_day_profit'], inplace=True)

features = ['sentiment_enc','trade_count','avg_size_usd','win_rate','daily_pnl']
X = model_df[features]
y = model_df['next_day_profit'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy: {acc:.3f}")
print(classification_report(y_test, y_pred, target_names=['Loss','Profit']))

# Feature importance
fi = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 4))
fi.plot(kind='barh', ax=ax, color='#1f77b4', edgecolor='black', linewidth=0.6)
ax.set_title('Bonus — Feature Importance\n(Predict Next-Day Profitability)', fontsize=13)
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig(f'{SAVE}Bonus_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Chart saved: Bonus_feature_importance.png")

print("\nBONUS COMPLETE ✓")
print("\n" + "=" * 60)
print("ALL ANALYSIS COMPLETE — charts saved to", SAVE)
print("=" * 60)
