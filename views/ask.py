import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import data, ui

SUGGESTIONS = ["Who is most likely to drop out?", "Where are the bottlenecks?", "Which roles will miss their deadline?",
               "Which sources work best?", "Whose tone is dropping?", "How long does hiring take?",
               "Recruiter workload", "Give me a summary"]
INTENTS = [  # first match wins
    ("summary", r"summary|overview|status|how (are|is) (we|things|hiring)|brief|where do we stand"),
    ("load", r"workload|caseload|work ?load|recruiter load|per recruiter|busiest|who has the most"),
    ("tth", r"time.?to.?hire|how long|days to hire|tth|hiring speed"),
    ("sources", r"source|channel|referral|linkedin|naukri|instahyre|agency|roi"),
    ("tone", r"tone|sentiment|mood|cooling|negative|going cold|unhappy"),
    ("bottleneck", r"bottleneck|stuck|sla|delay|slow|waiting|queue"),
    ("forecast", r"forecast|on track|deadline|will we|fill|target|close|miss"),
    ("risk", r"risk|drop|decline|renege|ghost|offer|likely to|safe"),
]


def _scope(D, q):
    """Narrow data to a role, department or recruiter named in the question."""
    roles = D["roles"]
    words = {}
    for t in roles.title:
        for w in re.findall(r"[a-z]+", t.lower()):
            words.setdefault(w, set()).add(t)
    hit, label = None, ""
    for t in roles.title:
        if t.lower() in q:
            hit, label = roles[roles.title == t], t
    if hit is None:
        for w, ts in words.items():
            if len(ts) == 1 and re.search(rf"\b{w}\b", q):
                hit, label = roles[roles.title.isin(ts)], next(iter(ts))
    if hit is None:
        for d in roles.dept.unique():
            if re.search(rf"\b{d.lower()}\b", q):
                hit, label = roles[roles.dept == d], f"{d} department"
    if hit is None:
        for rec in roles.recruiter.unique():
            if rec.lower() in q or re.search(rf"\b{rec.split()[0].lower()}\b", q):
                hit, label = roles[roles.recruiter == rec], f"{rec}'s roles"
    if hit is None:
        return D, "all roles"
    ids = set(hit.role_id)
    cands = D["cands"][D["cands"].role_id.isin(ids)]
    return {**D, "roles": hit, "cands": cands}, label


def _answer(intent, D, scope):
    c, roles = D["cands"], D["roles"]
    act = data.active(c)
    out = dict(text="", fig=None, table=None)
    if intent == "risk":
        o = data.live_offers(c).sort_values("risk", ascending=False)
        if o.empty:
            out["text"] = f"There are no live offers for {scope}."
            return out
        hi = (o.risk >= 50).sum()
        top = o.iloc[0]
        out["text"] = (f"**{len(o)} live offers for {scope}; {hi} are High or Critical.** Most exposed: **{top['name']}** "
                       f"({top.title}) at {top.risk:.0f}%. " + (f"Main driver: {top.reasons[0]['label'].lower()}. "
                                                              f"{top.reasons[0]['action']}" if top.reasons else ""))
        h = o.head(8)
        out["fig"] = go.Figure(go.Bar(x=h.risk, y=h["name"], orientation="h", marker=dict(
            color=[ui.TONES[data.RISK_TONE[l]] for l in h.risk_level]), hovertemplate="%{y}: %{x:.0f}%<extra></extra>"))
        out["fig"].update_yaxes(autorange="reversed")
        out["fig"].update_xaxes(title="Dropout risk (%)", range=[0, 100])
        out["table"] = pd.DataFrame({"Candidate": h["name"], "Role": h.title, "Risk %": h.risk.round(0),
                                     "Top reason": h.reasons.map(lambda r: r[0]["label"] if r else "-")})
    elif intent == "bottleneck":
        late = act[act.past_sla]
        if late.empty:
            out["text"] = f"Nothing is past SLA for {scope}."
            return out
        by = late.groupby("current_stage").agg(count=("candidate_id", "size"), days_over=("days_over", "sum"))
        by = by.reindex(list(data.SLA_DAYS), fill_value=0)
        worst = by["count"].idxmax()
        role = late[late.current_stage == worst].title.value_counts().index[0]
        out["text"] = (f"**{len(late)} candidates are past SLA for {scope}.** The worst queue is **{worst}** "
                       f"({by.loc[worst, 'count']} waiting, SLA {data.SLA_DAYS[worst]} days), concentrated in {role}.")
        out["fig"] = go.Figure(go.Bar(x=by.index, y=by["count"], marker=dict(color=ui.WARNING),
                                      hovertemplate="%{x}: %{y} past SLA<extra></extra>"))
        out["fig"].update_yaxes(title="Candidates past SLA")
        out["table"] = by.reset_index().rename(columns={"current_stage": "Stage", "count": "Past SLA",
                                                         "days_over": "Total days over"})
    elif intent == "forecast":
        r = roles.sort_values("p_fill")
        bad = r[r.forecast != "On track"]
        out["text"] = (f"**{len(r) - len(bad)} of {len(r)} roles are on track for {scope}.** Furthest behind: "
                       + ", ".join(f"{x.title} ({x.p_fill:.0%})" for x in r.head(3).itertuples()) + ".")
        col = {"On track": ui.GOOD, "At risk": ui.WARNING, "Off track": ui.CRITICAL}
        out["fig"] = go.Figure(go.Bar(x=r.p_fill * 100, y=r.title, orientation="h",
                                      marker=dict(color=[col[f] for f in r.forecast]),
                                      hovertemplate="%{y}: %{x:.0f}% chance of filling all seats<extra></extra>"))
        out["fig"].update_yaxes(autorange="reversed")
        out["fig"].update_xaxes(title="Chance of filling open seats by target date (%)", range=[0, 100])
        out["table"] = pd.DataFrame({"Role": r.title, "Open seats": r.open_seats, "Days left": r.days_left,
                                     "Fill chance": (r.p_fill * 100).round(0), "Forecast": r.forecast})
    elif intent == "sources":
        s = data.source_table(c)
        best = s.iloc[0]
        cheap = s.dropna(subset=["cost_per_hire"]).sort_values("cost_per_hire").iloc[0]
        out["text"] = (f"**{best.source} converts best for {scope}** ({best.hire_rate:.1f}% of applicants secured). "
                       f"Lowest cost per hire: **{cheap.source}** (₹{cheap.cost_per_hire:,.0f}).")
        out["fig"] = go.Figure(go.Bar(x=s.source, y=s.hire_rate, marker=dict(color=ui.ACCENT),
                                      hovertemplate="%{x}: %{y:.1f}% hire rate<extra></extra>"))
        out["fig"].update_yaxes(title="Hire rate (%)")
        out["table"] = s[["source", "applied", "hires", "hire_rate", "cost_per_hire"]].round(1).rename(columns={
            "source": "Source", "applied": "Applied", "hires": "Hires", "hire_rate": "Hire rate %",
            "cost_per_hire": "Cost per hire (₹)"})
    elif intent == "tone":
        t = act.dropna(subset=["sentiment_change"]).sort_values("sentiment_change").head(6)
        neg = (act.tone == "Negative").sum()
        out["text"] = (f"**{neg} active candidates sound negative for {scope}.** Sharpest tone drop: "
                       f"**{t.iloc[0]['name']}** ({t.iloc[0].sentiment_change:+.2f}).") if len(t) else \
            f"No message history for {scope}."
        if len(t):
            out["fig"] = go.Figure(go.Bar(x=t.sentiment_change, y=t["name"], orientation="h",
                                          marker=dict(color=ui.CRITICAL),
                                          hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
            out["fig"].update_yaxes(autorange="reversed")
            out["fig"].update_xaxes(title="Change in message tone (VADER)")
            out["table"] = pd.DataFrame({"Candidate": t["name"], "Stage": t.current_stage, "Role": t.title,
                                         "Tone change": t.sentiment_change.round(2),
                                         "Latest tone": t.sentiment_latest.round(2)})
    elif intent == "tth":
        t = c.dropna(subset=["time_to_hire"]).groupby("title").time_to_hire.agg(["median", "size"]).sort_values("median")
        if t.empty:
            out["text"] = f"No accepted offers yet for {scope}."
            return out
        out["text"] = (f"**Median time to hire for {scope} is {c.time_to_hire.median():.0f} days.** Fastest: "
                       f"{t.index[0]} ({t['median'].iloc[0]:.0f}d). Slowest: {t.index[-1]} ({t['median'].iloc[-1]:.0f}d).")
        out["fig"] = go.Figure(go.Bar(x=t["median"], y=t.index, orientation="h", marker=dict(color=ui.ACCENT),
                                      hovertemplate="%{y}: %{x:.0f} days<extra></extra>"))
        out["fig"].update_yaxes(autorange="reversed")
        out["fig"].update_xaxes(title="Median days, application to accepted offer")
    elif intent == "load":
        g = c.groupby("recruiter")
        tab = pd.DataFrame({
            "Active candidates": g.apply(lambda x: int((x.status == "Active").sum()), include_groups=False),
            "Past SLA": g.past_sla.sum(), "Live offers": g.apply(lambda x: len(data.live_offers(x)), include_groups=False),
            "Open seats": D["roles"].groupby("recruiter").open_seats.sum()}).fillna(0).astype(int)
        busiest = tab["Active candidates"].idxmax()
        out["text"] = (f"**{busiest} carries the most load** with {tab.loc[busiest, 'Active candidates']} active candidates "
                       f"and {tab.loc[busiest, 'Past SLA']} past SLA.")
        out["fig"] = go.Figure(go.Bar(x=tab.index, y=tab["Active candidates"], marker=dict(color=ui.ACCENT),
                                      hovertemplate="%{x}: %{y} active<extra></extra>"))
        out["fig"].update_yaxes(title="Active candidates")
        out["table"] = tab.reset_index().rename(columns={"recruiter": "Recruiter"})
    else:
        h = data.health(D)
        o = data.live_offers(c)
        out["text"] = (f"**Pipeline health for {scope}: {h['score']:.0f}/100.** {len(act)} active candidates, "
                       f"{int(act.past_sla.sum())} past SLA, {int((o.risk >= 50).sum())} of {len(o)} live offers at "
                       f"High or Critical risk, and {(roles.forecast == 'On track').sum()} of {len(roles)} roles on track.")
    return out


def _llm(question, D):
    try:
        key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return None
    try:
        import anthropic
        c, roles = D["cands"], D["roles"]
        o = data.live_offers(c).sort_values("risk", ascending=False).head(10)
        ctx = (f"Active candidates: {len(data.active(c))}. Past SLA: {int(data.active(c).past_sla.sum())}.\n"
               "Roles:\n" + roles[["title", "dept", "recruiter", "open_seats", "days_left", "p_fill", "forecast"]]
               .round(2).to_string(index=False) + "\nTop offer risks:\n"
               + o[["name", "title", "current_stage", "risk"]].round(0).to_string(index=False))
        try:
            model = st.secrets["ANTHROPIC_MODEL"]
        except Exception:
            model = "claude-haiku-5-5"
        msg = anthropic.Anthropic(api_key=key).messages.create(
            model=model, max_tokens=500,
            system="You are a hiring analytics assistant. Answer briefly using only the data provided. "
                   "If the data cannot answer the question, say so.",
            messages=[{"role": "user", "content": f"Data:\n{ctx}\n\nQuestion: {question}"}])
        return msg.content[0].text
    except Exception as exc:  # network, quota or auth problems should not break the page
        return f"The language-model fallback failed ({type(exc).__name__}). Try one of the suggested questions."


def respond(question, D):
    q = question.lower()
    scoped, label = _scope(D, q)
    for name, pat in INTENTS:
        if re.search(pat, q):
            return _answer(name, scoped, label)
    text = _llm(question, D)
    if text:
        return dict(text=text, fig=None, table=None)
    return dict(text="I can answer questions about **offer risk, bottlenecks, forecasts, sources, tone, time to hire, "
                     "recruiter workload** and a **summary**. Add a role, department or recruiter name to narrow it, "
                     "for example *which offers are risky for Data Scientist?*", fig=None, table=None)


def _queue(question):
    st.session_state["ask_pending"] = question


def _pill_changed():
    q = st.session_state.get("ask_pill")
    if q:
        _queue(q)
        st.session_state["ask_pill"] = None


def render():
    D = data.get_scoped()
    ui.header("Ask", "Plain-language questions about your pipeline. Name a role, department or recruiter to narrow it.")
    st.session_state.setdefault("ask_history", [])
    st.pills("Try asking", SUGGESTIONS, key="ask_pill", on_change=_pill_changed, label_visibility="collapsed")
    typed = st.chat_input("Ask about risk, bottlenecks, forecast, sources, tone...")
    question = typed or st.session_state.pop("ask_pending", None)
    if question:
        st.session_state["ask_history"].append(("user", dict(text=question)))
        st.session_state["ask_history"].append(("assistant", respond(question, D)))

    for i, (who, m) in enumerate(st.session_state["ask_history"]):
        with st.chat_message(who):
            st.markdown(m["text"])
            if m.get("fig") is not None:
                st.plotly_chart(ui.style(m["fig"], 280), width="stretch", config=ui.PLOT_CONFIG, key=f"ask_fig_{i}")
            if m.get("table") is not None:
                st.dataframe(m["table"], hide_index=True, width="stretch")
