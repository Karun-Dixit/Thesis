import os, io, textwrap
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mlxtend.frequent_patterns import apriori, association_rules

# ====== update this path ======
FILE_PATH = r"C:\Users\karun\Documents\new\Groceries_dataset.csv"

st.set_page_config(page_title="Grocery Recommender", layout="wide")

# ---------- light, safe CSS ----------
st.markdown("""
<style>
.stApp {
  color: #e5e7eb;
  background:
    radial-gradient(900px 420px at -10% -10%, #182742 0%, transparent 60%),
    radial-gradient(900px 420px at 115% 0%, #0f172a 0%, transparent 60%),
    linear-gradient(180deg, #0b1020 0%, #0a0f1c 100%);
}
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* cards */
.card {
  background: rgba(255,255,255,.055);
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 16px;
  padding: 16px;
  backdrop-filter: blur(6px);
}

/* tidy spacing */
h2, h3 { margin-top: .25rem; }
.help { color: #94a3b8; font-size: 12px; }

/* buttons */
.stButton>button { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------- hide Streamlit default header & menu ----------
st.markdown("""
    <style>
    header {visibility: hidden;}
    [data-testid="stToolbar"] {display: none;}
    [data-testid="stDecoration"] {display: none;}
    [data-testid="stHeader"] {display: none;}
    [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# ---------- helpers ----------
def normcol(c): return c.strip().lower().replace(" ", "").replace("_", "")
def fs2str(fs): return ", ".join(sorted(list(fs)))

@st.cache_data(show_spinner=False)
def load_basket_rules(path):
    df = pd.read_csv(path)
    df.columns = [normcol(c) for c in df.columns]
    mem = [c for c in df.columns if c in ("membernumber","memberno","customerid","customer","userid")]
    itm = [c for c in df.columns if c in ("itemdescription","item","product","description")]
    if not mem or not itm:
        raise ValueError(f"Member/Item columns not found. Columns: {df.columns.tolist()}")
    mcol, icol = mem[0], itm[0]
    basket = df.groupby([mcol, icol]).size().unstack(fill_value=0)
    basket = basket.applymap(lambda x: 1 if x >= 1 else 0).astype("uint8")
    freq = apriori(basket, min_support=0.01, use_colnames=True)
    rules = association_rules(freq, metric="lift", min_threshold=1.0)
    rules = rules.sort_values(["lift","confidence","support"], ascending=False).reset_index(drop=True)
    return basket, rules

def curated_top10(basket):
    must = ["eggs","domestic eggs","whole milk","milk","butter","yogurt",
            "bread","rolls/buns","other vegetables","soda"]
    must = [m for m in must if m in basket.columns]
    freq = basket.sum().sort_values(ascending=False)
    items = []
    for m in must:
        if m not in items: items.append(m)
    for it in freq.index:
        if it not in items: items.append(it)
        if len(items) >= 10: break
    return items

def top2_recs(rules: pd.DataFrame, item: str):
    if rules.empty: return pd.DataFrame()
    sub = rules[rules["antecedents"].apply(lambda s: item in s)].copy()
    if sub.empty: return pd.DataFrame()
    sub["Antecedent(s)"] = sub["antecedents"].apply(fs2str)
    sub["Consequent(s)"] = sub["consequents"].apply(fs2str)
    sub = sub.sort_values(["lift","confidence","support"], ascending=False)
    sub = sub.drop_duplicates(subset=["Consequent(s)"], keep="first")
    return sub[["Antecedent(s)","Consequent(s)","support","confidence","lift"]].head(2)

def plot_bar(rows: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    if rows is None or rows.empty:
        ax.text(0.5,0.5,"No recommendations for this item.",ha="center",va="center"); ax.axis("off"); return fig
    x = rows["Consequent(s)"].tolist()
    y = rows["confidence"].tolist()
    lifts = rows["lift"].tolist()
    ax.bar(x, y)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence (P(B|A))")
    ax.set_title("People also buy — confidence")
    ax.tick_params(axis='x', rotation=10)
    for i,(v,lf) in enumerate(zip(y,lifts)):
        ax.text(i, v+0.02, f"lift {lf:.2f}", ha='center', va='bottom', fontsize=9)
    fig.tight_layout(); return fig

def plot_network(rows: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    if rows is None or rows.empty:
        ax.text(0.5,0.5,"No recommendations for this item.",ha="center",va="center"); ax.axis("off"); return fig
    G = nx.DiGraph()
    for _, r in rows.iterrows():
        a, c = r["Antecedent(s)"], r["Consequent(s)"]
        G.add_edge(a, c, lift=float(r["lift"]), confidence=float(r["confidence"]))
    pos = nx.kamada_kawai_layout(G) if len(G.nodes())<=6 else nx.spring_layout(G, k=0.7, seed=42)
    lifts = [G[u][v]['lift'] for u,v in G.edges()]
    confs = [G[u][v]['confidence'] for u,v in G.edges()]
    norm = mcolors.Normalize(vmin=min(lifts), vmax=max(lifts)); cmap = cm.get_cmap('viridis')
    edge_colors = [cmap(norm(L)) for L in lifts]; edge_widths = [1.5 + 4*c for c in confs]
    nx.draw_networkx_nodes(G, pos, node_size=1100, node_color="#a3bffa",
                           edgecolors="#334155", linewidths=1.1, ax=ax)
    labels = {n: "\n".join(textwrap.wrap(n, 18)) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax)
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=12, width=edge_widths,
                           edge_color=edge_colors, connectionstyle='arc3,rad=0.12', ax=ax)
    edge_lbls = {(u,v): f"{G[u][v]['lift']:.2f}" for u,v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_lbls, font_size=8, ax=ax,
                                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([]); cb = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label('Lift', rotation=270, labelpad=10)
    ax.set_title("People also buy — network"); ax.axis("off"); fig.tight_layout(); return fig

# ---------- load ----------
basket, rules = load_basket_rules(FILE_PATH)
top_items = curated_top10(basket)

# ---------- state ----------
if "picked" not in st.session_state:
    st.session_state.picked = "whole milk" if "whole milk" in top_items else top_items[0]
if "chart" not in st.session_state:
    st.session_state.chart = "Bar (confidence)"

# ---------- layout: left rail | right content ----------
rail, content = st.columns([0.25, 0.75])

with rail:
    st.markdown("### 🛒 Grocery Recommender")
    st.caption("Apriori-based suggestions")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("**Choose an item** (Top-10 curated)")
    st.session_state.picked = st.selectbox(
        " ", options=top_items,
        index=top_items.index(st.session_state.picked),
        label_visibility="collapsed"
    )
    st.write("")
    st.write("Quick picks")
    q = [it for it in ["whole milk","yogurt","butter","eggs"] if it in top_items]
    cols = st.columns(len(q))
    for i, qi in enumerate(q):
        if cols[i].button(qi):
            st.session_state.picked = qi
    st.markdown("</div>", unsafe_allow_html=True)

with content:
    st.markdown("## Results")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    recs2 = top2_recs(rules, st.session_state.picked)

    if recs2.empty:
        st.info("No clear recommendations for this item.")
    else:
        for _, r in recs2.iterrows():
            st.markdown(f"- **{r['Consequent(s)']}**  "
                        f"(confidence: {r['confidence']:.2f}, lift: {r['lift']:.2f})")

        st.write("")  

        col_table, col_chart = st.columns([0.55, 0.45], vertical_alignment="top")

        with col_table:
            st.subheader("Details")
            st.dataframe(
                recs2.rename(columns={
                    "support":"support (share of all baskets)",
                    "confidence":"confidence (P(B|A))",
                    "lift":"lift (strength)"
                }).reset_index(drop=True),
                use_container_width=True, height=160
            )
            st.download_button(
                "Download these 2 recs (CSV)",
                data=recs2.to_csv(index=False).encode("utf-8"),
                file_name=f"recs_{st.session_state.picked.replace(' ','_')}.csv"
            )

        with col_chart:
            st.subheader("Chart")
            st.session_state.chart = st.radio(
                " ", ["Bar (confidence)", "Network (styled)"],
                index=0 if st.session_state.chart.startswith("Bar") else 1,
                horizontal=False, label_visibility="collapsed"
            )
            fig = plot_bar(recs2) if st.session_state.chart.startswith("Bar") else plot_network(recs2)
            st.pyplot(fig, use_container_width=True)
            buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight")
            st.download_button(
                "Download chart as PNG",
                data=buf.getvalue(),
                file_name=f"chart_{st.session_state.picked.replace(' ','_')}.png",
                mime="image/png"
            )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Confidence = how often the recommendation appears when your chosen item is present. Lift > 1 means a real association (not by chance).")
