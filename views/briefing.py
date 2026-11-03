import numpy as np
import streamlit as st

from core import data, ui


def _go(page, **state):
    st.session_state.update(state)
    st.switch_page(st.session_state["_pages"][page])


def _action(label, key, page, **state):
    if st.button(label, key=key):
        _go(page, **state)


def render():
    D = data.get_scoped()
    c, roles = D["cands"], D["roles"]
    act = data.active(c)
    offers = data.live_offers(c)
    h = data.health(D)

    hour = data.now_ist().hour
    greet = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
    who = data.viewer()
    who = "team" if who == data.ALL_RECRUITERS else who.split()[0]
    ui.header(f"{greet}, {who}", f"{data.now_ist():%A, %d %B}. Here is where hiring stands today.")

    g, k = st.columns([1, 2.2])
    with g:
        st.markdown('<div class="pa-muted">Pipeline health</div>', unsafe_allow_html=True)
        ui.show(ui.gauge(h["score"]), 210)
        ui.html(f'<div class="pa-muted">SLA compliance {h["sla"]:.0%} · offer safety {h["safety"]:.0%} · '
                f'roles on track {h["on_track"]:.0%}</div>')
    with k:
        hires = c[(c.status == "Joined") | ((c.status == "Active") & (c.current_stage == "Accepted"))]
        risky = offers[offers.risk >= 50]
        tth = c.time_to_hire.dropna()
        ui.kpi_row([
            ("Active candidates", f"{len(act)}", f"across {len(roles)} roles"),
            ("Open seats", f"{int(roles.open_seats.sum())}", f"{len(hires)} already secured"),
            ("Offers at high risk", f"{len(risky)}", f"of {len(offers)} live offers",
             "critical" if len(risky) else "good"),
            ("Past SLA", f"{int(act.past_sla.sum())}", "candidates waiting too long",
             "serious" if act.past_sla.sum() else "good"),
            ("Time to hire", f"{tth.median():.0f}d" if len(tth) else "-", "median, applied to accepted"),
        ])
        exp, seats = np.minimum(roles.expected, roles.open_seats).sum(), roles.open_seats.sum()
        on = (roles.forecast == "On track").sum()
        ui.html(ui.card(
            f"Forecast: about {min(exp, seats):.0f} of {int(seats)} open seats filled by target dates",
            f"{on} of {len(roles)} roles are on track, {(roles.forecast == 'At risk').sum()} at risk, "
            f"{(roles.forecast == 'Off track').sum()} off track. Probabilities blend stage conversion history with "
            f"each role's deadline.", "Forecast"))

    st.markdown("### Needs you today")
    cards = []
    if len(offers):
        r = offers.sort_values("risk", ascending=False).iloc[0]
        top = r.reasons[0] if r.reasons else None
        cards.append(dict(tag="Offer at risk", tone="critical" if r.risk >= 70 else "serious",
                          title=f"{r['name']} · {r.risk:.0f}% dropout risk",
                          body=f"{r.title} ({r.current_stage}). " + (f"{top['label']}. {top['action']}" if top else ""),
                          btn=("Open profile", "candidates", dict(cand_sel=r.candidate_id))))
    sla = act[act.past_sla]
    if len(sla):
        by = sla.groupby("current_stage").agg(n=("candidate_id", "size"), days=("days_over", "sum"))
        s = by.sort_values(["n", "days"], ascending=False).index[0]
        cards.append(dict(tag="Bottleneck", tone="warning", title=f"{s} stage is the slowest",
                          body=f"{int(by.loc[s, 'n'])} candidates are past the {data.SLA_DAYS[s]}-day SLA "
                               f"({int(by.loc[s, 'days'])} days over in total). Clear this queue first.",
                          btn=("View pipeline", "pipeline", {})))
    if len(roles):
        r = roles.sort_values("p_fill").iloc[0]
        cards.append(dict(tag="Role at risk", tone="serious", title=f"{r.title} · {r.p_fill:.0%} chance of filling",
                          body=f"{int(r.open_seats)} open seats, {r.days_left} days to target. Needs more qualified "
                               f"candidates in the funnel. Hiring manager: {r.hiring_manager}.",
                          btn=("View pipeline", "pipeline", dict(focus_role=r.title))))
    tone_c = act[act.sentiment_change.notna()]
    if len(tone_c):
        r = tone_c.sort_values("sentiment_change").iloc[0]
        cards.append(dict(tag="Tone drop", tone="warning", title=f"{r['name']}: tone down {abs(r.sentiment_change):.2f}",
                          body=f"{r.title} ({r.current_stage}). Recent messages read colder than earlier ones. "
                               f"A quick call can re-engage.",
                          btn=("Open profile", "candidates", dict(cand_sel=r.candidate_id))))
    mm = D["fb"][D["fb"].mismatch].merge(act[["candidate_id", "name", "title"]], on="candidate_id")
    if len(mm):
        r = mm.sort_values("date").iloc[-1]
        cards.append(dict(tag="Feedback mismatch", tone="accent",
                          title=f"{r['name']}: note and rating disagree",
                          body=f"{r.interviewer} rated {r.rating}/5 but the note reads "
                               f"{'positive' if r.compound > 0 else 'negative'}. Check before deciding.",
                          btn=("Open profile", "candidates", dict(cand_sel=r.candidate_id))))

    for row in (cards[:3], cards[3:]):
        for col, cd in zip(st.columns(3), row):
            with col:
                ui.html(ui.card(cd["title"], ui.e(cd["body"]), cd["tag"], cd["tone"], 170))
                label, page, state = cd["btn"]
                _action(label, f"brief_{cd['tag']}", page, **state)
