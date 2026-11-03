import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import data, ui


def render():
    D = data.get_scoped()
    c = D["cands"]
    ui.header("Pipeline", "Funnel, queue health, sourcing and speed. Filters apply to every chart on this page.")

    if "focus_role" in st.session_state:
        st.session_state["pl_role"] = [st.session_state.pop("focus_role")]
    f1, f2, f3 = st.columns(3)
    depts = f1.multiselect("Department", sorted(c.dept.unique()), key="pl_dept")
    titles = sorted(c[c.dept.isin(depts)].title.unique() if depts else c.title.unique())
    st.session_state["pl_role"] = [t for t in st.session_state.get("pl_role", []) if t in titles]
    role = f2.multiselect("Role", titles, key="pl_role")
    src = f3.multiselect("Source", sorted(c.source.unique()), key="pl_src")
    if depts:
        c = c[c.dept.isin(depts)]
    if role:
        c = c[c.title.isin(role)]
    if src:
        c = c[c.source.isin(src)]
    if c.empty:
        st.info("No candidates match these filters.")
        return

    left, right = st.columns([1, 1.15])
    with left:
        st.markdown("#### Funnel")
        stages = data.BOARD
        counts = [int(c.history.map(lambda h, s=s: any(x["stage"] == s for x in h)).sum()) for s in stages]
        fig = go.Figure(go.Funnel(
            y=stages, x=counts, textinfo="value+percent initial", marker=dict(color=ui.ACCENT),
            textfont=dict(color=ui.INK), connector=dict(line=dict(color="#262625")),
            hovertemplate="%{y}: %{x} candidates<extra></extra>"))
        ui.show(fig, 340)
    with right:
        st.markdown("#### Past-SLA candidates by role and stage")
        act = data.active(c)
        slas = list(data.SLA_DAYS)
        piv = (act[act.past_sla].pivot_table(index="title", columns="current_stage", values="candidate_id",
                                             aggfunc="count", fill_value=0)
               .reindex(index=sorted(c.title.unique()), columns=slas, fill_value=0))
        fig = go.Figure(go.Heatmap(
            z=piv.values, x=[f"{s} ({data.SLA_DAYS[s]}d)" for s in slas], y=piv.index, xgap=3, ygap=3,
            colorscale=[[0, "#1a1a19"], [0.001, "#3a2a1f"], [1, ui.CRITICAL]], showscale=False,
            text=piv.values, texttemplate="%{text}", textfont=dict(color=ui.INK),
            hovertemplate="%{y} · %{x}<br>%{z} past SLA<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        ui.show(fig, 340)

    st.markdown("#### Sources")
    s = data.source_table(c)
    a, b = st.columns([1.2, 1])
    with a:
        st.dataframe(
            s.rename(columns={"source": "Source", "applied": "Applied", "hires": "Hires", "hire_rate": "Hire rate",
                              "spend": "Spend (₹)", "cost_per_hire": "Cost per hire (₹)",
                              "hires_per_lakh": "Hires per ₹1L"}),
            hide_index=True, width="stretch",
            column_config={"Hire rate": st.column_config.ProgressColumn(format="%.1f%%", min_value=0,
                                                                       max_value=max(float(s.hire_rate.max()), 1)),
                           "Spend (₹)": st.column_config.NumberColumn(format="₹%d"),
                           "Cost per hire (₹)": st.column_config.NumberColumn(format="₹%d"),
                           "Hires per ₹1L": st.column_config.NumberColumn(format="%.2f")})
    with b:
        fig = go.Figure(go.Bar(x=s.hire_rate, y=s.source, orientation="h", marker=dict(color=ui.ACCENT),
                               hovertemplate="%{y}: %{x:.1f}% hire rate<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Hire rate (%)")
        ui.show(fig, 280)

    st.markdown("#### Time to hire by role")
    t = c.dropna(subset=["time_to_hire"]).groupby("title").time_to_hire.agg(["median", "size"]).sort_values("median")
    if t.empty:
        st.caption("No accepted offers in this selection yet.")
    else:
        fig = go.Figure(go.Bar(
            x=t["median"], y=t.index, orientation="h", marker=dict(color=ui.ACCENT), customdata=t["size"],
            hovertemplate="%{y}<br>%{x:.0f} days median (%{customdata} hires)<extra></extra>"))
        fig.update_xaxes(title="Days from application to accepted offer")
        fig.update_yaxes(autorange="reversed")
        ui.show(fig, max(220, 34 * len(t) + 60))
