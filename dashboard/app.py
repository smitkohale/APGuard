import os
from dotenv import load_dotenv

load_dotenv()
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

st.set_page_config(page_title="APGuard", layout="wide", page_icon="🛡️")

DB_URL = os.getenv("DATABASE_URL")


@st.cache_resource
def get_engine():
    return create_engine(DB_URL)


@st.cache_data(ttl=60)
def load_flags():
    return pd.read_sql("SELECT f.*, v.vendor_name FROM flags f LEFT JOIN vendors v ON v.vendor_id = f.vendor_id", get_engine())


@st.cache_data(ttl=60)
def load_vendors():
    return pd.read_sql(
        """
        SELECT v.vendor_id, v.vendor_name, v.risk_category,
               count(f.flag_id) AS flag_count,
               COALESCE(sum(f.amount_at_risk),0) AS total_at_risk
        FROM vendors v LEFT JOIN flags f ON f.vendor_id = v.vendor_id
        GROUP BY v.vendor_id, v.vendor_name, v.risk_category
        ORDER BY total_at_risk DESC
        """, get_engine(),
    )


st.title("🛡️ APGuard — Accounts Payable Leakage Detection")
st.caption("Rule-based detection of payment leakage and contract compliance issues across AP transactions.")

flags = load_flags()
vendors = load_vendors()

# ---------------- Top-level KPIs ----------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Flags", f"{len(flags):,}")
col2.metric("Total $ at Risk", f"${flags['amount_at_risk'].sum():,.0f}")
col3.metric("Vendors Flagged", f"{flags['vendor_id'].nunique():,}")
col4.metric("Rules Firing", f"{flags['rule_code'].nunique()} / 8")

st.divider()

# ---------------- Charts ----------------
c1, c2 = st.columns(2)

with c1:
    by_rule = flags.groupby(["rule_code", "rule_name"], as_index=False).agg(
        flag_count=("flag_id", "count"), total_at_risk=("amount_at_risk", "sum")
    ).sort_values("total_at_risk", ascending=False)
    fig = px.bar(by_rule, x="rule_name", y="total_at_risk", color="rule_code",
                 title="$ At Risk by Rule", labels={"total_at_risk": "$ at risk", "rule_name": "Rule"})
    fig.update_layout(showlegend=False, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    by_sev = flags.groupby("severity", as_index=False).agg(flag_count=("flag_id", "count"))
    order = ["critical", "high", "medium", "low"]
    by_sev["severity"] = pd.Categorical(by_sev["severity"], categories=order, ordered=True)
    by_sev = by_sev.sort_values("severity")
    fig2 = px.pie(by_sev, names="severity", values="flag_count", title="Flags by Severity",
                  color="severity",
                  color_discrete_map={"critical": "#8B0000", "high": "#E63946", "medium": "#F4A261", "low": "#2A9D8F"})
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------------- Vendor risk ranking ----------------
st.subheader("Vendor Risk Ranking")
top_n = st.slider("Show top N vendors", 5, 50, 15)
top_vendors = vendors.head(top_n)
fig3 = px.bar(top_vendors.sort_values("total_at_risk"), x="total_at_risk", y="vendor_name",
              orientation="h", color="risk_category", title="Top Vendors by $ at Risk",
              labels={"total_at_risk": "$ at risk", "vendor_name": "Vendor"})
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ---------------- Vendor drill-down ----------------
st.subheader("Vendor Drill-Down")
vendor_options = vendors[vendors["flag_count"] > 0].sort_values("total_at_risk", ascending=False)
selected_vendor = st.selectbox(
    "Select a vendor", options=vendor_options["vendor_id"],
    format_func=lambda vid: vendor_options.set_index("vendor_id").loc[vid, "vendor_name"],
)

if selected_vendor:
    v_flags = flags[flags["vendor_id"] == selected_vendor].sort_values("amount_at_risk", ascending=False)
    vcol1, vcol2 = st.columns(2)
    vcol1.metric("Flags for this vendor", len(v_flags))
    vcol2.metric("$ at risk", f"${v_flags['amount_at_risk'].sum():,.0f}")
    st.dataframe(
        v_flags[["rule_code", "rule_name", "invoice_id", "po_id", "payment_id", "severity", "amount_at_risk", "details"]],
        use_container_width=True, hide_index=True,
    )
    csv = v_flags.to_csv(index=False).encode("utf-8")
    st.download_button("Export this vendor's flags (CSV)", csv, file_name=f"vendor_{selected_vendor}_flags.csv")

st.divider()

# ---------------- All flags (filterable) ----------------
st.subheader("All Flags")
fc1, fc2, fc3 = st.columns(3)
rule_filter = fc1.multiselect("Rule", options=sorted(flags["rule_code"].unique()))
sev_filter = fc2.multiselect("Severity", options=sorted(flags["severity"].unique()))
min_amount = fc3.number_input("Min $ at risk", min_value=0, value=0, step=100)

filtered = flags.copy()
if rule_filter:
    filtered = filtered[filtered["rule_code"].isin(rule_filter)]
if sev_filter:
    filtered = filtered[filtered["severity"].isin(sev_filter)]
filtered = filtered[filtered["amount_at_risk"] >= min_amount]

st.write(f"{len(filtered):,} flags match filters")
st.dataframe(
    filtered[["rule_code", "rule_name", "vendor_name", "invoice_id", "po_id", "payment_id",
              "severity", "amount_at_risk", "details"]].sort_values("amount_at_risk", ascending=False),
    use_container_width=True, hide_index=True, height=400,
)
csv_all = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Export filtered flags (CSV)", csv_all, file_name="apguard_flags_filtered.csv")
