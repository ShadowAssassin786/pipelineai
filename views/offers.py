import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import data, ui


def render():
    D = data.get_scoped()
    o = data.live_offers(D["cands"])
    ui.header("Offers", "Live dropout risk for every candidate at Offer or Accepted, scored by the saved XGBoost model.")
    if o.empty:
        st.info("No live offers for this view.")
        return

    lvl = o.risk_level.value_counts()
    ui.kpi_row([
        ("Live offers", f"{len(o)}", f"{(o.current_stage == 'Offer').sum()} pending, {(o.current_stage == 'Accepted').sum()} accepted"),
        ("Average risk", f"{o.risk.mean():.0f}%", "across live offers"),
        ("High or critical", f"{int(lvl.get('High', 0) + lvl.get('Critical', 0))}", "need action this week",
         "critical" if lvl.get("Critical", 0) else "serious"),
        ("CTC at risk", f"₹{(o.offer_ctc * o.risk / 100).sum():.0f}L", "risk-weighted annual CTC"),
    ])

    pick = st.segmented_control("Risk level", ["All"] + data.RISK_LEVELS, default="All", key="offer_filter")
    view = o if pick in (None, "All") else o[o.risk_level == pick]
    view = view.sort_values("risk", ascending=False)

    if view.empty:
        st.caption("No offers at this risk level.")
    else:
        t = pd.DataFrame({
            "Candidate": view["name"], "Role": view.title, "Stage": view.current_stage,
            "Risk": view.risk.round(0), "Level": view.risk_level,
            "Top reasons": view.reasons.map(lambda rs: "; ".join(x["label"] for x in rs) or "No strong drivers"),
            "Protective": view.protective.map(lambda ps: "; ".join(x["label"] for x in ps)),
            "Offer CTC (₹L)": view.offer_ctc, "Recruiter": view.recruiter})
        st.dataframe(t, hide_index=True, width="stretch", column_config={
            "Risk": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100),
            "Offer CTC (₹L)": st.column_config.NumberColumn(format="%.1f"),
            "Top reasons": st.column_config.TextColumn(width="large"),
            "Protective": st.column_config.TextColumn(width="large")})

    st.markdown("#### What drives dropout overall")
    imp = pd.Series(data.metrics()["shap_importance"]).sort_values()
    fig = go.Figure(go.Bar(
        x=imp.values, y=[data.FEATURE_LABELS[k] for k in imp.index], orientation="h", marker=dict(color=ui.ACCENT),
        hovertemplate="%{y}: mean |SHAP| %{x:.2f}<extra></extra>"))
    fig.update_xaxes(title="Mean |SHAP| across 1,800 historical offers")
    ui.show(fig, 340)
