# =========================
# Basket Analysis (Thesis-Ready, 6 Visuals, Clean)
# =========================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
from itertools import combinations
from sklearn.linear_model import LinearRegression

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.facecolor"] = "white"

# -------------------------
# CONFIG
# -------------------------
file_path = r"C:\Users\karun\Documents\new\Groceries_dataset.csv"

# Save figures in same folder as dataset
SAVE_FIGS = True
FIG_DIR = os.path.dirname(file_path)  # same folder as dataset

# Visual params (tight & clean)
TOP_N_ITEMS_BAR = 10             # bar chart
TOP_K_COOC_ITEMS = 10            # co-occurrence heatmap
TOP_K_CLUSTER_ITEMS = 10         # clustered corr heatmap
TOP_R_RULES_NETWORK = 12         # network edges (rules)
SCATTER_RULES = 400              # top rules by lift for scatter

# -------------------------
# Step 1: Load dataset
# -------------------------
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Could not find file at: {file_path}")

df = pd.read_csv(file_path)

print("Sample Data:")
print(df.head())
print("\nDataset shape:", df.shape)

def normcol(c):
    return c.strip().lower().replace(" ", "").replace("_", "")

df.columns = [normcol(c) for c in df.columns]

candidate_member = [c for c in df.columns if c in ("membernumber", "memberno", "customerid", "customer", "userid")]
candidate_item   = [c for c in df.columns if c in ("itemdescription", "item", "product", "description")]
candidate_date   = [c for c in df.columns if c in ("date", "orderdate", "timestamp")]

if not candidate_member or not candidate_item:
    raise ValueError(f"Could not find member and item columns. Found: {df.columns}")

member_col = candidate_member[0]
item_col   = candidate_item[0]
date_col   = candidate_date[0] if candidate_date else None

# -------------------------
# Step 2: Basket format
# -------------------------
basket = (
    df.groupby([member_col, item_col])
      .size()
      .unstack(fill_value=0)
)
basket = basket.applymap(lambda x: 1 if x >= 1 else 0).astype("uint8")

item_freq = basket.sum().sort_values(ascending=False)
top_items = item_freq.head(10)
total_baskets = basket.shape[0]
total_items = basket.shape[1]

print("\nTop 10 Items by Frequency:")
print(top_items)
print(f"\nTotal Baskets: {total_baskets}")
print(f"Total Unique Items: {total_items}")

# -------------------------
# Step 3: Apriori + Rules
# -------------------------
frequent_items = apriori(basket, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=1.0).sort_values("lift", ascending=False)

print("\nTop Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(15))

# -------------------------
# Step 4: Regression Example (eggs -> butter)
# -------------------------
cols_lower = {c.lower(): c for c in basket.columns}
eggs_col   = cols_lower.get('eggs')
butter_col = cols_lower.get('butter')

if eggs_col and butter_col:
    X = basket[[eggs_col]]
    y = basket[butter_col]
    model = LinearRegression().fit(X, y)
    print("\n[Regression: Eggs → Butter]")
    print("Coefficient:", float(model.coef_[0]))
    print("Intercept:", float(model.intercept_))
    print("R² Score:", float(model.score(X, y)))
else:
    print("\n[Regression: Eggs → Butter] Skipped (columns not found).")

# -------------------------
# Helper: Save figs
# -------------------------
def _savefig(name):
    if SAVE_FIGS:
        os.makedirs(FIG_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIG_DIR, name), bbox_inches="tight")

# =========================================
# 6 Visualizations (trimmed for clarity)
# =========================================

# 1) Top items bar chart (Top 10)
plt.figure(figsize=(9, 5))
ax = item_freq.head(TOP_N_ITEMS_BAR).plot(kind='bar')
plt.title(f"Top {TOP_N_ITEMS_BAR} Items by Frequency")
plt.xlabel("Item")
plt.ylabel("Baskets containing item")
plt.xticks(rotation=35, ha='right')
plt.tight_layout()
_ = _savefig("1_top_items_bar.png")
plt.show()

# 2) Basket size distribution
basket_sizes = basket.sum(axis=1)
plt.figure(figsize=(8, 5))
basket_sizes.value_counts().sort_index().plot(kind='bar')
plt.title("Basket Size Distribution")
plt.xlabel("Items per basket")
plt.ylabel("Number of baskets")
plt.xticks(rotation=0)
plt.tight_layout()
_ = _savefig("2_basket_size_distribution.png")
plt.show()

# 3) Co-occurrence heatmap (Top 10 items)
topk_items = item_freq.head(TOP_K_COOC_ITEMS).index
cooc = basket[topk_items].T.dot(basket[topk_items])
np.fill_diagonal(cooc.values, 0)

plt.figure(figsize=(8.8, 7.2))
sns.heatmap(cooc, cmap='Reds', cbar_kws={'shrink': 0.8})
plt.title(f"Item Co-occurrence (Top {TOP_K_COOC_ITEMS})")
plt.xticks(rotation=35, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
_ = _savefig("3_cooccurrence_heatmap.png")
plt.show()

# 4) Clustered correlation heatmap (Top 10 items)
topk_items = item_freq.head(TOP_K_CLUSTER_ITEMS).index
corr_topk = basket[topk_items].corr().fillna(0)
g = sns.clustermap(corr_topk, cmap='coolwarm', center=0, figsize=(8.8, 8.8), cbar_kws={'shrink': 0.8})
plt.suptitle(f"Clustered Correlation (Top {TOP_K_CLUSTER_ITEMS} Items)", y=1.02)
if SAVE_FIGS:
    g.savefig(os.path.join(FIG_DIR, "4_clustered_corr_heatmap.png"))
plt.show()

# 5) Support vs Confidence (Top 400 rules by lift)
rules_scatter = rules.head(SCATTER_RULES).copy()
plt.figure(figsize=(8, 6))
sizes = np.clip(rules_scatter['lift'] * 20, 20, 200)
plt.scatter(rules_scatter['support'], rules_scatter['confidence'], s=sizes, alpha=0.6)
plt.title('Association Rules: Support vs Confidence (size ≈ Lift)')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.tight_layout()
_ = _savefig("5_support_vs_confidence.png")
plt.show()

# 6) Directed rule network (Top 12 rules by lift)
rules_exp = rules.copy()
rules_exp['antecedent'] = rules_exp['antecedents'].apply(lambda s: ', '.join(sorted(list(s))))
rules_exp['consequent'] = rules_exp['consequents'].apply(lambda s: ', '.join(sorted(list(s))))
rules_top = rules_exp.head(TOP_R_RULES_NETWORK)[['antecedent','consequent','lift','confidence']]

G_dir = nx.DiGraph()
for _, r in rules_top.iterrows():
    G_dir.add_edge(r['antecedent'], r['consequent'], lift=float(r['lift']), confidence=float(r['confidence']))

plt.figure(figsize=(11, 8))
pos = nx.spring_layout(G_dir, k=0.7, seed=42)
edge_widths = [2 + 4*(G_dir[u][v]['confidence']) for u, v in G_dir.edges()]
nx.draw_networkx_nodes(G_dir, pos, node_size=1100, node_color='lightsteelblue')
nx.draw_networkx_labels(G_dir, pos, font_size=9)
nx.draw_networkx_edges(G_dir, pos, arrows=True, width=edge_widths, arrowstyle='-|>', arrowsize=14)
plt.title("Directed Rule Network (Top 12 by Lift; edge width ∝ confidence)")
plt.axis('off')
plt.tight_layout()
_ = _savefig("6_directed_rule_network.png")
plt.show()

# -------------------------
# Pair frequency table (appendix-friendly)
# -------------------------
pair_counts = {}
for _, row in basket.iterrows():
    items_in_basket = basket.columns[row.values.astype(bool)]
    for a, b in combinations(items_in_basket, 2):
        key = tuple(sorted((a, b)))
        pair_counts[key] = pair_counts.get(key, 0) + 1

pair_freq_df = (pd.Series(pair_counts)
                  .sort_values(ascending=False)
                  .head(25)
                  .rename("count")
                  .reset_index())
pair_freq_df.columns = ["item_a", "item_b", "count"]

print("\nTop 25 Item Pairs by Frequency:")
print(pair_freq_df)
