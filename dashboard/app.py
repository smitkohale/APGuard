"""APGuard dashboard.

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apguard import metrics  # noqa: E402
from apguard.cases import STATUS_LABELS, STATUSES, WorkflowError, allowed_transitions, update_flag  # noqa: E402
from apguard.db import read_sql  # noqa: E402
from apguard.evidence import evidence_for_flag  # noqa: E402
from rules.engine import BENFORD, benford_test  # noqa: E402
from rules.registry import RULES, SEVERITIES  # noqa: E402

st.set_page_config(page_title="APGuard", layout="wide", initial_sidebar_state="expanded")

READ_ONLY = os.getenv("APGUARD_DASHBOARD_READ_ONLY", "false").lower() in ("1", "true", "yes")

# --------------------------------------------------------------------------- #
# visual system
# --------------------------------------------------------------------------- #
INK, MUTED, RULE_LINE = "#1A1D23", "#5F6B7A", "#E3E6EA"
PRIMARY, PRIMARY_LIGHT = "#1F3A5F", "#A9BCD6"
SEVERITY_COLOR = {"critical": "#0B1F3A", "high": "#1F3A5F", "medium": "#5B7BA6", "low": "#A9BCD6"}
ALERT = "#B42318"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stDataFrame, button, input, textarea, select {
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif !important;
}
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }
h1, h2, h3 { font-weight: 600 !important; letter-spacing: -0.01em; color: #1A1D23; }
h3 { font-size: 1.05rem !important; margin-top: 0.6rem !important; }
[data-testid="stSidebar"] { border-right: 1px solid #E3E6EA; }
[data-testid="stSidebar"] h3 { font-size: 0.75rem !important; text-transform: uppercase;
    letter-spacing: 0.06em; color: #5F6B7A; }
.ap-header { display: flex; justify-content: space-between; align-items: flex-end;
    border-bottom: 1px solid #E3E6EA; padding-bottom: 0.9rem; margin-bottom: 1.2rem; }
.ap-title { font-size: 1.45rem; font-weight: 600; color: #1A1D23; line-height: 1.2; }
.ap-sub { color: #5F6B7A; font-size: 0.88rem; margin-top: 0.15rem; }
.ap-meta { color: #5F6B7A; font-size: 0.8rem; text-align: right; line-height: 1.5; }
.ap-kpis { display: grid; gap: 0;
    border: 1px solid #E3E6EA; border-radius: 6px; margin-bottom: 1.4rem; }
.ap-kpi { padding: 0.95rem 1.1rem; border-right: 1px solid #E3E6EA; }
.ap-kpi:last-child { border-right: none; }
.ap-kpi-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: #5F6B7A; }
.ap-kpi-value { font-size: 1.45rem; font-weight: 600; color: #1A1D23; margin-top: 0.25rem;
    font-variant-numeric: tabular-nums; }
.ap-kpi-note { font-size: 0.76rem; color: #5F6B7A; margin-top: 0.15rem; }
.ap-note { font-size: 0.8rem; color: #5F6B7A; line-height: 1.5; }
.ap-section-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: #5F6B7A; margin: 0.3rem 0 0.4rem 0; }
.ap-field { font-size: 0.85rem; line-height: 1.7; }
.ap-field span { color: #5F6B7A; display: inline-block; min-width: 128px; }
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid #E3E6EA; }
.stTabs [data-baseweb="tab"] { padding: 0.5rem 0; font-size: 0.9rem; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
@media (max-width: 900px) { .ap-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
    .ap-kpi { border-bottom: 1px solid #E3E6EA; } }
</style>
""", unsafe_allow_html=True)

PLOT_CONFIG = {"displayModeBar": False}


def base_layout(fig: go.Figure, height=320, **kw) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12, color=INK), showlegend=kw.pop("showlegend", False),
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=RULE_LINE), **kw)
    fig.update_xaxes(showgrid=False, linecolor=RULE_LINE, ticks="", tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor="#F0F2F4", zeroline=False, linecolor=RULE_LINE,
                     tickfont=dict(color=MUTED))
    return fig


def money(x: float, compact=True) -> str:
    if x is None or pd.isna(x):
        return "-"
    if compact and abs(x) >= 1e6:
        return f"${x / 1e6:,.2f}M"
    if compact and abs(x) >= 1e4:
        return f"${x / 1e3:,.0f}K"
    return f"${x:,.0f}"


def pct(x) -> str:
    return "-" if x is None or pd.isna(x) else f"{x:.1%}"


def kpi_row(items: list[tuple[str, str, str]]) -> None:
    style = f'style="grid-template-columns: repeat({len(items)}, minmax(0, 1fr))"'
    cells = "".join(f'<div class="ap-kpi"><div class="ap-kpi-label">{lab}</div>'
                    f'<div class="ap-kpi-value">{val}</div><div class="ap-kpi-note">{note}</div></div>'
                    for lab, val, note in items)
    st.markdown(f'<div class="ap-kpis" {style}>{cells}</div>', unsafe_allow_html=True)


def label(text: str) -> None:
    st.markdown(f'<div class="ap-section-label">{text}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=60, show_spinner=False)
def load_flags() -> pd.DataFrame:
    df = read_sql("""SELECT f.flag_id, f.rule_code, f.rule_name, f.category, f.vendor_id, f.vendor_name,
                            f.invoice_id, f.po_id, f.payment_id, f.severity, f.amount_at_risk, f.details,
                            f.status, f.assignee, f.recovered_amount, f.updated_at, i.invoice_date
                     FROM v_active_flags f LEFT JOIN invoices i ON i.invoice_id = f.invoice_id""")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


@st.cache_data(ttl=60, show_spinner=False)
def load_summary() -> dict:
    return metrics.summary()


@st.cache_data(ttl=60, show_spinner=False)
def load_vendor_risk() -> pd.DataFrame:
    return metrics.vendor_risk()


@st.cache_data(ttl=60, show_spinner=False)
def load_by_rule() -> pd.DataFrame:
    return metrics.by_rule()


@st.cache_data(ttl=300, show_spinner=False)
def load_evaluation() -> dict | None:
    p = ROOT / "reports" / "evaluation.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(ttl=300, show_spinner=False)
def vendor_amounts(vendor_id: int) -> pd.Series:
    return read_sql("SELECT total_amount FROM invoices WHERE vendor_id = :v", {"v": vendor_id})["total_amount"]


def refresh() -> None:
    for fn in (load_flags, load_summary, load_vendor_risk, load_by_rule):
        fn.clear()


try:
    flags = load_flags()
    summary = load_summary()
except Exception as exc:  # pragma: no cover - UI path
    st.error(f"Could not reach the database ({exc.__class__.__name__}). Check DATABASE_URL and that the "
             "pipeline has been run (see README).")
    st.stop()

evaluation = load_evaluation()
last_run = summary.get("last_run") or {}

# --------------------------------------------------------------------------- #
# header and filters
# --------------------------------------------------------------------------- #
run_text = (f"Run {last_run.get('run_id')} &middot; {pd.Timestamp(last_run['finished_at']).strftime('%d %b %Y, %H:%M UTC')}"
            if last_run.get("finished_at") else "No completed run")
st.markdown(f"""
<div class="ap-header">
  <div><div class="ap-title">APGuard</div>
       <div class="ap-sub">Accounts payable leakage monitoring</div></div>
  <div class="ap-meta">{run_text}<br>{summary['active_flags']:,} active flags &middot;
       engine v{last_run.get('engine_version', '-')}</div>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Filters")
    rule_opts = [c for c in RULES if c in set(flags["rule_code"])]
    f_rules = st.multiselect("Detector", rule_opts, format_func=lambda c: f"{c}  {RULES[c].name}")
    f_sev = st.multiselect("Severity", SEVERITIES, format_func=str.capitalize)
    f_status = st.multiselect("Status", STATUSES, format_func=STATUS_LABELS.get)
    f_cat = st.radio("Detector type", ["All", "Controls", "Analytics"], horizontal=True)
    f_min = st.number_input("Minimum amount at risk ($)", min_value=0, value=0, step=500)
    st.markdown("### About the data")
    st.markdown('<div class="ap-note">Synthetic AP ledger with known injected exceptions. Review decisions '
                'marked "(simulated)" were generated by <code>scripts/simulate_reviews.py</code> to '
                'demonstrate the workflow.</div>', unsafe_allow_html=True)
    if st.button("Reload data", use_container_width=True):
        refresh()
        st.rerun()

view = flags.copy()
if f_rules:
    view = view[view["rule_code"].isin(f_rules)]
if f_sev:
    view = view[view["severity"].isin(f_sev)]
if f_status:
    view = view[view["status"].isin(f_status)]
if f_cat != "All":
    view = view[view["category"] == ("control" if f_cat == "Controls" else "analytic")]
if f_min:
    view = view[view["amount_at_risk"] >= f_min]
filtered = len(view) != len(flags)


def exposure(df: pd.DataFrame) -> tuple[float, float]:
    """(net invoice-level exposure, vendor-level exposure), excluding dismissed flags."""
    c = df[(df["category"] == "control") & (df["status"] != "false_positive")]
    net = c.dropna(subset=["invoice_id"]).groupby("invoice_id")["amount_at_risk"].max().sum()
    return float(net), float(c[c["invoice_id"].isna()]["amount_at_risk"].sum())


tab_overview, tab_detectors, tab_vendors, tab_cases, tab_method = st.tabs(
    ["Overview", "Detectors", "Vendors", "Case queue", "Methodology"])

# --------------------------------------------------------------------------- #
# overview
# --------------------------------------------------------------------------- #
with tab_overview:
    ctrl = view[view["category"] == "control"]
    gross = float(ctrl["amount_at_risk"].sum())
    net, vendor_level = exposure(view)
    open_n = int((view["status"] == "open").sum())
    reviewed = view["status"].isin(["confirmed", "recovered", "false_positive"])
    prec = view.loc[reviewed, "status"].isin(["confirmed", "recovered"]).mean() if reviewed.any() else None
    kpi_row([
        ("Net exposure", money(net), f"plus {money(vendor_level)} vendor-level (R10)"),
        ("Gross flagged", money(gross), f"{len(ctrl):,} control flags"),
        ("Recovered", money(view["recovered_amount"].sum()), f"{int((view['status'] == 'recovered').sum()):,} cases closed"),
        ("Open cases", f"{open_n:,}", f"{int((view['status'] == 'in_review').sum()):,} in review"),
        ("Reviewer precision", pct(prec), f"{int(reviewed.sum()):,} flags reviewed"),
    ])
    if filtered:
        st.caption(f"Filtered view: {len(view):,} of {len(flags):,} active flags.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("### Net exposure by detector")
        per_inv = ctrl[ctrl["status"] != "false_positive"].sort_values("amount_at_risk", ascending=False)
        primary = per_inv.dropna(subset=["invoice_id"]).drop_duplicates("invoice_id")
        by = primary.groupby("rule_code")["amount_at_risk"].sum().reindex([c for c in RULES], fill_value=0)
        by = by[by > 0].sort_values()
        fig = go.Figure(go.Bar(
            x=by.values, y=[f"{c}  {RULES[c].name}" for c in by.index], orientation="h",
            marker_color=PRIMARY, hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>"))
        base_layout(fig, height=max(260, 30 * len(by) + 40))
        fig.update_xaxes(showgrid=True, gridcolor="#F0F2F4", tickprefix="$", tickformat="~s")
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
        st.markdown('<div class="ap-note">Each invoice is attributed to its single largest control flag, so '
                    'invoices hit by several rules are not double-counted. Vendor-level exposure (R10) is '
                    'shown separately because it overlaps invoice-level flags.</div>', unsafe_allow_html=True)
    with right:
        st.markdown("### Flags by severity")
        sev = view.groupby("severity").size().reindex(SEVERITIES, fill_value=0)
        fig = go.Figure(go.Bar(x=[s.capitalize() for s in sev.index], y=sev.values,
                               marker_color=[SEVERITY_COLOR[s] for s in sev.index],
                               hovertemplate="%{x}: %{y:,}<extra></extra>", text=[f"{v:,}" for v in sev.values],
                               textposition="outside", cliponaxis=False))
        base_layout(fig, height=220)
        fig.update_yaxes(visible=False)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

        st.markdown("### Case status")
        status = view.groupby("status").agg(flags=("flag_id", "size"), amount=("amount_at_risk", "sum"),
                                            recovered=("recovered_amount", "sum")).reindex(STATUSES, fill_value=0)
        status.index = [STATUS_LABELS[s] for s in status.index]
        st.dataframe(status.reset_index(names="Status"), hide_index=True, use_container_width=True,
                     column_config={"flags": st.column_config.NumberColumn("Flags", format="%d"),
                                    "amount": st.column_config.NumberColumn("At risk", format="dollar"),
                                    "recovered": st.column_config.NumberColumn("Recovered", format="dollar")})

    st.markdown("### Flagged exposure by invoice month")
    monthly = ctrl.dropna(subset=["invoice_date"]).sort_values("amount_at_risk", ascending=False) \
        .drop_duplicates("invoice_id")
    monthly = monthly.groupby(monthly["invoice_date"].dt.to_period("M"))["amount_at_risk"].sum()
    monthly.index = monthly.index.to_timestamp()
    fig = go.Figure(go.Scatter(x=monthly.index, y=monthly.values, mode="lines", line=dict(color=PRIMARY, width=2),
                               fill="tozeroy", fillcolor="rgba(31,58,95,0.08)",
                               hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>"))
    base_layout(fig, height=230)
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

# --------------------------------------------------------------------------- #
# detectors
# --------------------------------------------------------------------------- #
with tab_detectors:
    by_rule = load_by_rule().set_index("rule_code")
    scores = {s["rule"]: s for s in evaluation["scores"]} if evaluation else {}
    rows = []
    for code, spec in RULES.items():
        r = by_rule.loc[code] if code in by_rule.index else None
        sc = scores.get(code, {})
        reviewed_n = (r["confirmed"] + r["false_positive"]) if r is not None else 0
        def p100(x):
            return None if x is None or pd.isna(x) else 100 * x
        rows.append({
            "Code": code, "Detector": spec.name, "Control tested": spec.control,
            "Flags": int(r["flag_count"]) if r is not None else 0,
            "At risk": float(r["total_at_risk"]) if r is not None else 0.0,
            "Test precision": p100(sc.get("precision")), "Test recall": p100(sc.get("recall")),
            "Reviewer precision": p100(r["confirmed"] / reviewed_n) if reviewed_n else None,
        })
    st.markdown("### Detector register")
    pct_col = st.column_config.NumberColumn(format="%.1f%%")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=35 * (len(rows) + 1) + 3,
                 column_config={"At risk": st.column_config.NumberColumn(format="dollar"),
                                "Test precision": pct_col, "Test recall": pct_col, "Reviewer precision": pct_col})
    st.markdown('<div class="ap-note">Test precision and recall are measured against the exceptions injected by '
                'the data generator (<code>reports/EVALUATION.md</code>). Reviewer precision comes from case '
                'decisions in the queue.</div>', unsafe_allow_html=True)

    with st.expander("What each detector checks"):
        for code, spec in RULES.items():
            st.markdown(f"**{code} {spec.name}**: {spec.description}")

    if evaluation:
        st.markdown("### What the refinements buy")
        cmp_df = pd.DataFrame(evaluation["baseline_comparison"])
        fig = go.Figure()
        labels = [f"{r}  {n}" for r, n in zip(cmp_df["rule"], cmp_df["name"], strict=True)]
        fig.add_bar(name="Naive implementation", x=labels, y=cmp_df["precision_naive"], marker_color=PRIMARY_LIGHT,
                    hovertemplate="%{x}<br>naive precision %{y:.1%}<extra></extra>")
        fig.add_bar(name="APGuard", x=labels, y=cmp_df["precision_apguard"], marker_color=PRIMARY,
                    hovertemplate="%{x}<br>APGuard precision %{y:.1%}<extra></extra>")
        base_layout(fig, height=300, showlegend=True, barmode="group",
                    legend=dict(orientation="h", y=1.12, x=0, font=dict(color=MUTED)))
        fig.update_yaxes(tickformat=".0%", range=[0, 1.05], title=dict(text="Precision", font=dict(color=MUTED)))
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
        st.markdown('<div class="ap-note">Naive versions are exact-match duplicates that flag both invoices, '
                    'per-invoice three-way matching, contracts looked up without dates, and splitting checks '
                    'unaware of partial deliveries. Recall is equal or higher for APGuard on every rule.</div>',
                    unsafe_allow_html=True)

        st.markdown("### Threshold sensitivity")
        sens = pd.DataFrame(evaluation["sensitivity"])
        param = st.selectbox("Parameter", sens["parameter"].unique().tolist())
        s = sens[sens["parameter"] == param]
        fig = go.Figure()
        fig.add_scatter(x=s["value"], y=s["precision"], name="Precision", mode="lines+markers",
                        line=dict(color=PRIMARY, width=2))
        fig.add_scatter(x=s["value"], y=s["recall"], name="Recall", mode="lines+markers",
                        line=dict(color=MUTED, width=2, dash="dot"))
        base_layout(fig, height=280, showlegend=True, legend=dict(orientation="h", y=1.12, x=0))
        fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
        fig.update_xaxes(title=dict(text=param, font=dict(color=MUTED)))
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

        st.markdown("### Rules vs an unsupervised model")
        iso = evaluation["isolation_forest"]
        iso_df = pd.DataFrame(iso["per_type"]).rename(columns={
            "exception_type": "Exception", "injected": "Injected",
            "isolation_forest_recall": "Isolation Forest recall", "rules_recall": "Control rules recall"})
        iso_df["Exception"] = iso_df["Exception"].str.replace("_", " ").str.capitalize()
        st.dataframe(iso_df, hide_index=True, use_container_width=True, column_config={
            "Isolation Forest recall": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Control rules recall": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
        st.markdown(f'<div class="ap-note">Isolation Forest on {len(iso["features"])} generic features, top '
                    f'{iso["k"]:,} invoices: precision {iso["isolation_forest_precision"]:.1%} vs '
                    f'{iso["rules_precision"]:.1%} for the rules. The model is weak on structured failures '
                    'but strong on inflated prices that no control rule targets, which is why A2 exists.</div>',
                    unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# vendors
# --------------------------------------------------------------------------- #
with tab_vendors:
    vr = load_vendor_risk()
    st.markdown("### Vendor risk ranking")
    st.markdown('<div class="ap-note">Score 0-100: 40% exposure as a share of the vendor\'s spend, 30% net exposure, '
                '30% vendor-integrity signal (cloned record R10, paid days after creation R11, Benford '
                'nonconformity A1). Transaction errors scale with volume, so concentration and integrity matter '
                'more than raw flag counts. Dismissed flags are excluded.</div>', unsafe_allow_html=True)
    show = vr.rename(columns={"vendor_name": "Vendor", "flag_count": "Flags", "rules": "Detectors",
                              "net_exposure": "Net exposure", "risk_score": "Risk score", "invoiced": "Invoiced",
                              "exposure_share": "Share of spend", "max_severity": "Worst severity"})
    show["Worst severity"] = show["Worst severity"].str.capitalize()
    show["Share of spend"] = 100 * show["Share of spend"]
    st.dataframe(show[["Vendor", "Risk score", "Net exposure", "Share of spend", "Invoiced", "Worst severity",
                       "Flags", "Detectors"]].head(50),
                 hide_index=True, use_container_width=True, height=420, column_config={
                     "Risk score": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
                     "Net exposure": st.column_config.NumberColumn(format="dollar"),
                     "Invoiced": st.column_config.NumberColumn(format="dollar"),
                     "Share of spend": st.column_config.NumberColumn(format="%.1f%%")})

    st.markdown("### Vendor profile")
    vid = st.selectbox("Vendor", vr["vendor_id"].tolist(),
                       format_func=lambda v: vr.set_index("vendor_id").loc[v, "vendor_name"])
    if vid is not None:
        vf = flags[flags["vendor_id"] == vid]
        row = vr.set_index("vendor_id").loc[vid]
        amounts = vendor_amounts(int(vid))
        bt = benford_test(amounts)
        kpi_row([
            ("Risk score", f"{row['risk_score']:.1f}", f"worst severity: {row['max_severity']}"),
            ("Net exposure", money(row["net_exposure"]), f"{row['exposure_share']:.1%} of {money(row['invoiced'])} invoiced"),
            ("Active flags", f"{len(vf):,}", row["rules"]),
            ("Benford MAD", f"{bt['mad']:.3f}",
             "nonconforming above 0.015 (Nigrini)" if bt["n"] >= 100 else f"only {bt['n']} invoices: too few to test"),
        ])
        left, right = st.columns([2, 3], gap="large")
        with left:
            label(f"First-digit distribution, {bt['n']:,} invoices")
            fig = go.Figure()
            fig.add_bar(x=list(range(1, 10)), y=bt["observed"], marker_color=PRIMARY, name="Observed",
                        hovertemplate="Digit %{x}: %{y:.1%}<extra></extra>")
            fig.add_scatter(x=list(range(1, 10)), y=BENFORD, mode="lines+markers", name="Benford",
                            line=dict(color=ALERT, width=1.5), marker=dict(size=5),
                            hovertemplate="Expected %{y:.1%}<extra></extra>")
            base_layout(fig, height=260, showlegend=True, legend=dict(orientation="h", y=1.15, x=0))
            fig.update_yaxes(tickformat=".0%")
            fig.update_xaxes(dtick=1)
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
        with right:
            label("Flags for this vendor")
            vt = vf[["rule_code", "rule_name", "severity", "amount_at_risk", "status", "details"]] \
                .sort_values("amount_at_risk", ascending=False).copy()
            vt["severity"] = vt["severity"].str.capitalize()
            vt["status"] = vt["status"].map(STATUS_LABELS)
            st.dataframe(vt, hide_index=True, use_container_width=True,
                         height=min(260, 35 * (len(vt) + 1) + 3), column_config={
                             "rule_code": "Code", "rule_name": "Detector", "severity": "Severity",
                             "amount_at_risk": st.column_config.NumberColumn("At risk", format="dollar"),
                             "status": "Status", "details": st.column_config.TextColumn("Details", width="large")})

# --------------------------------------------------------------------------- #
# case queue
# --------------------------------------------------------------------------- #
with tab_cases:
    st.markdown("### Case queue")
    queue = view.sort_values(["amount_at_risk"], ascending=False).reset_index(drop=True)
    st.caption(f"{len(queue):,} flags. Select a row to open the case.")
    table = queue[["flag_id", "rule_code", "rule_name", "vendor_name", "invoice_id", "severity",
                   "amount_at_risk", "status", "assignee", "details"]].copy()
    table["severity"] = table["severity"].str.capitalize()
    table["status"] = table["status"].map(STATUS_LABELS)
    table["assignee"] = table["assignee"].fillna("")
    table["invoice_id"] = table["invoice_id"].map(lambda v: "" if pd.isna(v) else str(int(v)))
    event = st.dataframe(table, hide_index=True, use_container_width=True, height=360, on_select="rerun",
                         selection_mode="single-row", key="queue", column_config={
                             "flag_id": st.column_config.NumberColumn("Flag", format="%d"),
                             "rule_code": "Code", "rule_name": "Detector", "vendor_name": "Vendor",
                             "invoice_id": "Invoice",
                             "severity": "Severity",
                             "amount_at_risk": st.column_config.NumberColumn("At risk", format="dollar"),
                             "status": "Status", "assignee": "Assignee",
                             "details": st.column_config.TextColumn("Details", width="large")})
    st.download_button("Export filtered flags (CSV)", queue.to_csv(index=False).encode("utf-8"),
                       file_name="apguard_flags.csv", mime="text/csv")

    rows_sel = event.selection.rows if event and event.selection else []
    if rows_sel:
        fid = int(queue.iloc[rows_sel[0]]["flag_id"])
        ev = evidence_for_flag(fid)
        f = ev["flag"]
        st.divider()
        st.markdown(f"### Flag {fid}: {f['rule_code']} {f['rule_name']}")
        st.markdown(f'<div class="ap-note">{f["details"]}</div>', unsafe_allow_html=True)

        left, right = st.columns([3, 2], gap="large")
        with left:
            label("Document chain")
            chain = []

            def num(x, dollars=False):
                if x is None or pd.isna(x):
                    return ""
                return f"${float(x):,.2f}" if dollars else f"{int(x):,}"

            def doc(name, d, qty=None, price=None, amount=None, note=""):
                chain.append({"Document": name, "Date": str(d)[:10], "Quantity": num(qty),
                              "Unit price": num(price, True), "Amount": num(amount, True), "Note": note})

            po, inv, con = ev["purchase_order"], ev["invoice"], ev["contract"]
            if con:
                doc(f"Contract {con['contract_id']}", con["start_date"], price=con["contract_price"],
                    note=f"in force to {con['end_date']}, tax {float(con['tax_rate']):.2%}")
            if po:
                doc(f"Purchase order {po['po_id']}", po["po_date"], po["quantity"], po["unit_price"],
                    po["po_amount"], po["item_sku"])
            for g in ev["goods_receipts"]:
                doc(f"Goods receipt {g['gr_id']}", g["received_date"], g["received_qty"])
            seen = set()
            for i in (ev["invoices_on_po"] or ([inv] if inv else [])):
                seen.add(i["invoice_id"])
                flagged_inv = inv and i["invoice_id"] == inv["invoice_id"]
                doc(f"Invoice {i['invoice_number'].strip()}", i["invoice_date"], i["quantity"], i["unit_price"],
                    i["total_amount"], "flagged invoice" if flagged_inv else "")
            for i in ev.get("similar_invoices", []):
                if i["invoice_id"] not in seen:
                    po_ref = f"PO {int(i['po_id'])}" if pd.notna(i["po_id"]) else "no PO"
                    doc(f"Invoice {i['invoice_number'].strip()}", i["invoice_date"], i["quantity"], i["unit_price"],
                        i["total_amount"], f"possible original, {po_ref}")
            for p in ev["payments"]:
                doc(f"Payment {p['payment_ref']}", p["payment_date"], amount=p["payment_amount"],
                    note="flagged payment" if p["payment_id"] == f.get("payment_id") else "")
            if chain:
                st.dataframe(pd.DataFrame(chain).sort_values("Date", kind="stable"), hide_index=True,
                             use_container_width=True)
            else:
                st.markdown('<div class="ap-note">Vendor-level flag: no single document chain.</div>',
                            unsafe_allow_html=True)
            v = ev["vendor"]
            if v:
                label("Vendor record")
                st.markdown(
                    f'<div class="ap-field"><span>Name</span>{v["vendor_name"]}<br>'
                    f'<span>Vendor ID</span>{v["vendor_id"]}<br><span>Tax ID</span>{v["tax_id"]}<br>'
                    f'<span>Bank account</span>{v["bank_account"]}<br><span>Created</span>{v["created_at"]}</div>',
                    unsafe_allow_html=True)
        with right:
            label("Case")
            st.markdown(
                f'<div class="ap-field"><span>Status</span>{STATUS_LABELS[f["status"]]}<br>'
                f'<span>Severity</span>{f["severity"].capitalize()}<br>'
                f'<span>Amount at risk</span>${float(f["amount_at_risk"]):,.2f}<br>'
                f'<span>Recovered</span>${float(f["recovered_amount"]):,.2f}<br>'
                f'<span>Assignee</span>{f["assignee"] or "Unassigned"}</div>', unsafe_allow_html=True)
            nxt = allowed_transitions(f["status"])
            if READ_ONLY:
                st.caption("This deployment is read-only.")
            elif nxt:
                with st.form(f"review-{fid}", clear_on_submit=True):
                    new_status = st.selectbox("Move to", nxt, format_func=STATUS_LABELS.get)
                    assignee = st.text_input("Assignee", value=f["assignee"] or "")
                    note = st.text_area("Note", placeholder="Required when dismissing as a false positive")
                    amount = st.number_input("Recovered amount ($)", min_value=0.0, step=100.0,
                                             help="Required when marking recovered")
                    if st.form_submit_button("Save", type="primary"):
                        try:
                            update_flag(fid, status=new_status, assignee=assignee or None, note=note or None,
                                        recovered_amount=amount if new_status == "recovered" else None,
                                        actor=assignee or "dashboard")
                            refresh()
                            st.success(f"Flag {fid} moved to {STATUS_LABELS[new_status]}.")
                            st.rerun()
                        except WorkflowError as exc:
                            st.error(str(exc))
            else:
                st.caption("Closed: no further transitions.")
            label("History")
            for e in reversed(ev["events"]):
                when = pd.Timestamp(e["created_at"]).strftime("%d %b %Y %H:%M")
                what = (f"{STATUS_LABELS.get(e['from_status'], '')} to {STATUS_LABELS.get(e['to_status'], '')}"
                        if e["event_type"] == "status_change" else e["event_type"].replace("_", " ").capitalize())
                st.markdown(f'<div class="ap-note"><b>{when}</b> &middot; {e["actor"]} &middot; {what}'
                            f'{"<br>" + e["note"] if e["note"] else ""}</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# methodology
# --------------------------------------------------------------------------- #
with tab_method:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("### How it works")
        st.markdown("""
APGuard runs thirteen detectors over a linked AP ledger (vendors, contracts, purchase orders, goods receipts,
invoices, payments). Eleven are deterministic **controls** that test a specific AP control; two are statistical
**analytics** (Benford's Law, unit-price benchmarking) that surface patterns no rule was written for.

Each run is recorded with the exact thresholds used. Flags have a stable identity across runs, so re-running
the engine never discards review work. Every review action is written to an append-only audit trail.

**Exposure.** Gross exposure sums every control flag. Net exposure keeps the largest control flag per invoice,
because one invoice can trip several rules (a duplicate keyed without a PO is both R1 and R4).
""")
        st.markdown("### Limitations")
        st.markdown("""
- The data is synthetic. The generator and the rules were written by the same person, so high recall on
  injected exceptions shows the rules do what they claim, not how they would perform on a real ledger.
  Legitimate look-alikes (partial deliveries, recurring billing, contract amendments, rounding) are included to
  make precision meaningful, but real data will contain patterns not modelled here.
- Duplicate detection does not yet use OCR-level text similarity on vendor names or line descriptions.
- Single currency, single entity, one line per invoice.
""")
    with right:
        st.markdown("### Control mapping")
        cm = pd.DataFrame([{"Code": c, "Control": s.control, "Assertion": s.assertion} for c, s in RULES.items()])
        st.dataframe(cm, hide_index=True, use_container_width=True, height=500)
