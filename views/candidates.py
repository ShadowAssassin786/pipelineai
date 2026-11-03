import plotly.graph_objects as go
import streamlit as st

from core import data, ui

PER_COLUMN = 6
TONE_COLOR = {"Positive": "good", "Neutral": "muted", "Negative": "critical"}


def _select(cid):
    st.session_state["cand_sel"] = cid


def _chips(r):
    out = ""
    if pd_notna(r.risk):
        out += ui.chip(f"{r.risk_level} · {r.risk:.0f}%", data.RISK_TONE[r.risk_level])
    elif r.past_sla:
        out += ui.chip(f"{int(r.days_over)}d over SLA", "serious")
    elif r.sla_days == r.sla_days:
        out += ui.chip(f"{int(r.days_in_stage)}/{int(r.sla_days)}d in stage", "good")
    if r.tone in TONE_COLOR:
        out += ui.chip(r.tone, TONE_COLOR[r.tone])
        if r.sentiment_change == r.sentiment_change and r.sentiment_change <= -0.3:
            out += ui.chip("Tone drop", "warning")
    return out


def pd_notna(v):
    return v == v and v is not None


def _urgency(df):
    key = df.risk.fillna(-1) * 1000 + df.days_over * 10 + df.days_in_stage
    return df.assign(_u=key).sort_values("_u", ascending=False).drop(columns="_u")


def render():
    D = data.get_scoped()
    act = data.active(D["cands"])
    ui.header("Candidates", "Each column is sorted by urgency: offer risk first, then days past SLA.")
    if act.empty:
        st.info("No active candidates for this view.")
        return
    ids = act.sort_values("name").candidate_id.tolist()
    if st.session_state.get("cand_sel") not in ids:
        st.session_state["cand_sel"] = _urgency(act[act.risk.notna()] if act.risk.notna().any() else act).iloc[0].candidate_id
    names = act.set_index("candidate_id")["name"]
    st.selectbox("Selected candidate", ids, key="cand_sel",
                 format_func=lambda i: f"{names[i]} · {act.set_index('candidate_id').loc[i, 'current_stage']}")
    sel = st.session_state["cand_sel"]

    for col, stage in zip(st.columns(5), data.BOARD):
        g = _urgency(act[act.current_stage == stage])
        with col:
            ui.html(f'<div class="pa-kcol">{stage} <span class="pa-muted">{len(g)}</span></div>')
            for r in g.head(PER_COLUMN).itertuples():
                ui.html(f'<div class="pa-kcard {"sel" if r.candidate_id == sel else ""}"><div class="n">{ui.e(r.name)}'
                        f'</div><div class="r">{ui.e(r.title)}</div>{_chips(r)}</div>')
                st.button("Open", key=f"open_{r.candidate_id}", on_click=_select, args=(r.candidate_id,))
            if len(g) > PER_COLUMN:
                ui.html(f'<div class="pa-muted">+{len(g) - PER_COLUMN} more. Use the selector above.</div>')

    st.divider()
    _profile(D, act[act.candidate_id == sel].iloc[0])


def _profile(D, r):
    cid = r.candidate_id
    ui.html(f'<h3 style="margin:0">{ui.e(r["name"])}</h3><div class="pa-muted">{ui.e(r.title)} · {ui.e(r.city)} · '
            f'{ui.e(r.source)} · recruiter {ui.e(r.recruiter)} · {ui.e(r.current_stage)} for {int(r.days_in_stage)} days'
            f'</div><div style="margin-top:8px">{_chips(r)}</div>')

    left, right = st.columns([1.1, 1])
    msgs = D["msgs"][D["msgs"].candidate_id == cid].sort_values("date")
    fb = D["fb"][D["fb"].candidate_id == cid].sort_values("date")
    with left:
        st.markdown("##### Message tone (VADER)")
        if msgs.empty:
            st.caption("No messages from this candidate yet.")
        else:
            fig = go.Figure(go.Scatter(
                x=msgs.date, y=msgs.compound, mode="lines+markers", line=dict(color=ui.ACCENT, width=2),
                marker=dict(size=8, color=[ui.GOOD if v >= 0.05 else ui.CRITICAL if v <= -0.05 else ui.MUTED
                                           for v in msgs.compound]),
                customdata=msgs.text, hovertemplate="%{x|%d %b}: %{y:.2f}<br>%{customdata}<extra></extra>"))
            fig.add_hline(y=0, line=dict(color="#3a3a37", width=1))
            fig.update_yaxes(range=[-1, 1], title="Compound score")
            ui.show(fig, 230)
            for m in msgs.sort_values("date", ascending=False).itertuples():
                col = ui.GOOD if m.compound >= 0.05 else ui.CRITICAL if m.compound <= -0.05 else ui.MUTED
                ui.html(f'<div class="pa-card" style="padding:10px 14px;margin-bottom:6px"><span class="pa-muted">'
                        f'{m.date:%d %b} · <span style="color:{col}">{m.compound:+.2f}</span></span><p>{ui.e(m.text)}'
                        f'</p></div>')
        st.markdown("##### Interview feedback")
        if fb.empty:
            st.caption("No feedback recorded.")
        for f in fb.itertuples():
            flag = ui.chip("Note and rating disagree", "warning") if f.mismatch else ""
            ui.html(f'<div class="pa-card" style="padding:10px 14px;margin-bottom:6px"><span class="pa-muted">'
                    f'Round {f.round} · {ui.e(f.interviewer)} · {f.date:%d %b} · rating {f.rating}/5 · tone '
                    f'{f.compound:+.2f}</span> {flag}<p>{ui.e(f.note)}</p></div>')

    with right:
        if not pd_notna(r.risk):
            sla = f"{int(r.days_over)} days past the {int(r.sla_days)}-day SLA" if r.past_sla else \
                f"within the {int(r.sla_days)}-day SLA"
            st.markdown("##### Next step")
            ui.html(ui.card("Move this candidate forward", f"In {ui.e(r.current_stage)} for {int(r.days_in_stage)} "
                            f"days, {sla}. The dropout model scores candidates from the Offer stage.", "Action"))
            return
        _risk_panel(D, r)


def _risk_panel(D, r):
    cid = r.candidate_id
    tone = data.RISK_TONE[r.risk_level]
    st.markdown("##### Dropout risk")
    ui.html(f'<div class="pa-kpi"><div class="l">{r.risk_level} risk</div><div class="v" style="color:{ui.TONES[tone]}">'
            f'{r.risk:.0f}%</div>{ui.bar(r.risk, tone)}</div>')

    feats = data.metrics()["features"]
    contrib = dict(zip(feats, r.contribs))
    top = sorted(contrib.items(), key=lambda kv: abs(kv[1]))[-8:]
    fig = go.Figure(go.Bar(
        x=[v for _, v in top], y=[data.FEATURE_LABELS[k] for k, _ in top], orientation="h",
        marker=dict(color=[ui.CRITICAL if v > 0 else ui.GOOD for _, v in top]),
        hovertemplate="%{y}: %{x:+.2f} log-odds<extra></extra>"))
    fig.update_xaxes(title="SHAP (red raises risk)")
    ui.show(fig, 280)

    st.markdown("##### Suggested actions")
    if r.reasons:
        for x in r.reasons:
            ui.html(ui.card(x["label"], ui.e(x["action"]), "Risk driver", "serious"))
    else:
        st.caption("No strong risk drivers.")
    if r.protective:
        ui.html('<div class="pa-muted">Protective: ' + " · ".join(ui.e(x["label"]) for x in r.protective) + "</div>")

    st.markdown("##### What-if")
    hike0 = float(min(max(round(r.hike_pct), 0), 60))
    hike = st.slider("Hike %", 0, 60, int(hike0), key=f"wi_hike_{cid}")
    notice = st.selectbox("Notice period (days)", [15, 30, 60, 90], index=[15, 30, 60, 90].index(int(r.notice_days)),
                          key=f"wi_notice_{cid}")
    comp = st.toggle("Competing offer", value=bool(r.competing_offer), key=f"wi_comp_{cid}")
    new, _ = data.whatif(r, hike, notice, comp)
    delta = new - r.risk
    c1, c2 = st.columns(2)
    c1.metric("Offered CTC", f"₹{r.ctc_current * (1 + hike / 100):.1f}L", f"{hike}% hike", delta_color="off")
    c2.metric("New risk", f"{new:.0f}%", f"{delta:+.0f} pts", delta_color="inverse")
    ui.html(f'<div class="pa-muted">New level: {data.risk_level(new)}</div>')

    st.markdown("##### Follow-up draft")
    st.text_area("Draft", data.followup_draft(r, D["company"]), height=220, key=f"draft_{cid}_{r.risk_level}",
                 label_visibility="collapsed")
