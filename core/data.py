"""Cached loading, business rules and live scoring."""
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
IST = timezone(timedelta(hours=5, minutes=30))
ALL_RECRUITERS = "All recruiters"

STAGES = ["Applied", "Screened", "Interview", "Offer", "Accepted", "Joined"]
BOARD = STAGES[:5]
SLA_DAYS = {"Applied": 5, "Screened": 5, "Interview": 7, "Offer": 5}
SCORED_STAGES = ("Offer", "Accepted")
RISK_LEVELS = ["Low", "Medium", "High", "Critical"]
RISK_TONE = {"Low": "good", "Medium": "warning", "High": "serious", "Critical": "critical"}
ON_TRACK, AT_RISK = 0.6, 0.3

SOURCE_COST = {  # INR: per applicant, per hire, share of offered CTC
    "LinkedIn": dict(per_applicant=450, per_hire=0, ctc_share=0),
    "Naukri": dict(per_applicant=250, per_hire=0, ctc_share=0),
    "Instahyre": dict(per_applicant=350, per_hire=0, ctc_share=0),
    "Careers page": dict(per_applicant=60, per_hire=0, ctc_share=0),
    "Referral": dict(per_applicant=0, per_hire=30000, ctc_share=0),
    "Agency": dict(per_applicant=0, per_hire=0, ctc_share=0.0833),
}

FEATURE_LABELS = {
    "hike_pct": "Hike offered", "notice_days": "Notice period", "competing_offer": "Competing offer",
    "relocation": "Relocation needed", "source_quality": "Source quality", "ctc_current": "Current CTC",
    "days_in_process": "Days in process", "days_since_last_reply": "Silence since last reply",
    "sentiment_latest": "Recent message tone", "sentiment_change": "Tone change",
}
ACTIONS = {
    "hike_pct": "Ask the hiring manager for a compensation review or a joining bonus.",
    "notice_days": "Offer a notice-period buyout or a flexible joining date.",
    "competing_offer": "Set up a call with the hiring manager and share the growth story to counter.",
    "relocation": "Offer relocation assistance or a remote-first start.",
    "source_quality": "Ask the referrer or sourcing partner to reconnect with the candidate.",
    "ctc_current": "Share a full total-compensation breakdown, including variable pay and benefits.",
    "days_in_process": "Lock the joining date and send the onboarding pack now.",
    "days_since_last_reply": "Call the candidate today. Silence is the strongest dropout signal.",
    "sentiment_latest": "Hold a recruiter call to understand and address their concerns.",
    "sentiment_change": "Hold a recruiter call; their tone has dropped since earlier conversations.",
}


def now_ist():
    return datetime.now(IST)


def risk_level(score):
    return "Low" if score < 30 else "Medium" if score < 50 else "High" if score < 70 else "Critical"


def tone_label(v):
    if v is None or pd.isna(v):
        return "No messages"
    return "Positive" if v >= 0.05 else "Negative" if v <= -0.05 else "Neutral"


def reason_text(feat, val, risk):
    if feat == "hike_pct":
        return f"Hike of only {val:.0f}%" if risk else f"Healthy {val:.0f}% hike"
    if feat == "notice_days":
        return f"Long {int(val)}-day notice period" if risk else f"Short {int(val)}-day notice period"
    if feat == "competing_offer":
        return "Has a competing offer" if risk else "No competing offer"
    if feat == "relocation":
        return "Needs to relocate" if risk else "No relocation needed"
    if feat == "source_quality":
        return "Lower-quality sourcing channel" if risk else "High-quality sourcing channel"
    if feat == "ctc_current":
        return f"High current CTC (₹{val:.0f}L)" if risk else f"Comfortable CTC gap (₹{val:.0f}L today)"
    if feat == "days_in_process":
        return f"{int(val)} days in process" if risk else f"Quick process ({int(val)} days)"
    if feat == "days_since_last_reply":
        return f"Silent for {int(val)} days" if risk else ("Replied recently" if val <= 1 else f"Replied {int(val)}d ago")
    if feat == "sentiment_latest":
        return "Recent messages sound negative" if risk else "Recent messages sound positive"
    return "Tone is dropping" if risk else "Tone is improving"


@st.cache_resource(show_spinner=False)
def booster():
    b = xgb.Booster()
    b.load_model(str(PROC / "model.json"))
    return b


@st.cache_data(show_spinner=False)
def metrics():
    return json.loads((PROC / "metrics.json").read_text())


def score_frame(X):
    """Return (risk %, contribution matrix) for a frame of model features."""
    feats = metrics()["features"]
    dm = xgb.DMatrix(X[feats], feature_names=feats)
    b = booster()
    return b.predict(dm) * 100, b.predict(dm, pred_contribs=True)[:, :-1]


def explain(row_feats, contribs):
    feats = metrics()["features"]
    pairs = sorted(zip(feats, contribs), key=lambda kv: -kv[1])
    risk = [(f, c) for f, c in pairs if c > 0.05][:3]
    prot = [(f, c) for f, c in reversed(pairs) if c < -0.05][:3]
    return (
        [dict(feature=f, label=reason_text(f, row_feats[f], True), action=ACTIONS[f], contrib=float(c)) for f, c in risk],
        [dict(feature=f, label=reason_text(f, row_feats[f], False), contrib=float(c)) for f, c in prot],
    )


def poisson_binomial_at_least(ps, k):
    if k <= 0:
        return 1.0
    dp = np.zeros(k + 1)
    dp[0] = 1.0
    for p in ps:
        nxt = dp * (1 - p)
        nxt[1:] += dp[:-1] * p
        nxt[k] += dp[k] * p  # k means "k or more"
        dp = nxt
    return float(dp[k])


def _shift_history(raw, shift):
    out = json.loads(raw)
    for h in out:
        h["entered"] = pd.Timestamp(h["entered"]) + shift
        h["exited"] = pd.Timestamp(h["exited"]) + shift if h["exited"] else None
    return out


@st.cache_data(show_spinner=False, ttl=3600)
def _build(today_iso):
    today = pd.Timestamp(today_iso)
    meta = json.loads((RAW / "meta.json").read_text())
    shift = pd.Timedelta(days=(today - pd.Timestamp(meta["anchor"])).days)

    roles = pd.read_csv(RAW / "roles.csv", parse_dates=["target_close"])
    roles["target_close"] += shift
    roles["days_left"] = (roles.target_close - today).dt.days

    c = pd.read_csv(RAW / "candidates.csv", parse_dates=["applied_date"])
    c["applied_date"] += shift
    c["history"] = c.stage_history.map(lambda r: _shift_history(r, shift))
    c = c.drop(columns="stage_history")
    c["stage_entered"] = c.history.map(lambda h: h[-1]["entered"])
    c["days_in_stage"] = (today - c.stage_entered).dt.days
    active = c.status == "Active"
    c["sla_days"] = c.current_stage.map(SLA_DAYS)
    c["days_over"] = np.where(active & c.sla_days.notna(), np.maximum(c.days_in_stage - c.sla_days, 0), 0)
    c["past_sla"] = c.days_over > 0
    acc = c.history.map(lambda h: next((x["entered"] for x in h if x["stage"] == "Accepted"), pd.NaT))
    c["accepted_date"] = pd.to_datetime(acc)
    c["time_to_hire"] = (c.accepted_date - c.applied_date).dt.days
    c = c.merge(roles[["role_id", "title", "dept", "recruiter", "city"]], on="role_id")
    feats = pd.read_csv(PROC / "candidate_features.csv").drop(columns=["notice_days", "competing_offer", "relocation", "ctc_current"])
    c = c.merge(feats, on="candidate_id", how="left")
    c["tone"] = c.sentiment_latest.map(tone_label)

    # live scoring for Offer / Accepted
    c["risk"], c["risk_level"] = np.nan, ""
    c["reasons"], c["protective"], c["contribs"] = None, None, None
    c["reasons"] = c.reasons.astype(object)
    c["protective"] = c.protective.astype(object)
    c["contribs"] = c.contribs.astype(object)
    m = active & c.current_stage.isin(SCORED_STAGES)
    if m.any():
        risk, contrib = score_frame(c[m])
        idx = c.index[m]
        for i, r, cv in zip(idx, risk, contrib):
            rs, pr = explain(c.loc[i], cv)
            c.at[i, "risk"], c.at[i, "risk_level"] = float(r), risk_level(float(r))
            c.at[i, "reasons"], c.at[i, "protective"], c.at[i, "contribs"] = rs, pr, cv.tolist()

    # funnel statistics from resolved history
    reached = {s: c.history.map(lambda h, s=s: any(x["stage"] == s for x in h)) for s in STAGES}
    trans, dwell = {}, {}
    for a, b in zip(STAGES[:4], STAGES[1:5]):
        decided = reached[a] & (reached[b] | (c.status != "Active"))
        trans[a] = float((reached[a] & reached[b]).sum() / max(decided.sum(), 1))
    for s in SLA_DAYS:
        d = [(x["exited"] - x["entered"]).days for h in c.history for x in h if x["stage"] == s and x["exited"] is not None]
        dwell[s] = float(np.median(d)) if d else float(SLA_DAYS[s])
    base_drop = metrics()["dropout_rate"]

    # per-candidate hire probability and deadline feasibility
    left = c.role_id.map(roles.set_index("role_id").days_left)
    p_hire, feas = [], []
    for r, d_left, stage, d_in in zip(c.risk, left, c.current_stage, c.days_in_stage):
        if stage in SCORED_STAGES:
            p_hire.append(1 - r / 100 if pd.notna(r) else 1.0)
            feas.append(1.0)
            continue
        if stage == "Joined":
            p_hire.append(1.0)
            feas.append(1.0)
            continue
        i = STAGES.index(stage)
        p = (1 - base_drop) * np.prod([trans[s] for s in STAGES[i:4]])
        remaining = max(dwell[stage] - d_in, 0) + sum(dwell[s] for s in STAGES[i + 1:4])
        p_hire.append(float(p))
        feas.append(float(1 / (1 + np.exp(-(d_left - remaining) / 4))))
    c["p_hire"], c["feasibility"] = p_hire, feas
    c.loc[~active & (c.status != "Joined"), ["p_hire", "feasibility"]] = 0.0

    # role forecast
    rows = []
    for r in roles.itertuples():
        g = c[c.role_id == r.role_id]
        live = g[(g.status == "Active") & (g.current_stage != "Accepted")]
        ps = (live.p_hire * live.feasibility).tolist()
        p_fill = poisson_binomial_at_least(ps, r.open_seats)
        rows.append(dict(role_id=r.role_id, expected=float(sum(ps)), p_fill=p_fill,
                         forecast="On track" if p_fill >= ON_TRACK else "At risk" if p_fill >= AT_RISK else "Off track"))
    roles = roles.merge(pd.DataFrame(rows), on="role_id")

    msgs = pd.read_csv(PROC / "messages_scored.csv", parse_dates=["date"])
    msgs["date"] += shift
    fb = pd.read_csv(PROC / "feedback_scored.csv", parse_dates=["date"])
    fb["date"] += shift
    return dict(roles=roles, cands=c, msgs=msgs, fb=fb, trans=trans, dwell=dwell, today=today,
                company=meta["company"])


def get_data():
    return _build(now_ist().date().isoformat())


def viewer():
    return st.session_state.get("viewer", ALL_RECRUITERS)


def get_scoped():
    """Data restricted to the 'Viewing as' recruiter from the sidebar."""
    D = get_data()
    v = viewer()
    if v == ALL_RECRUITERS:
        return D
    roles = D["roles"][D["roles"].recruiter == v]
    cands = D["cands"][D["cands"].role_id.isin(roles.role_id)]
    ids = set(cands.candidate_id)
    return {**D, "roles": roles, "cands": cands, "msgs": D["msgs"][D["msgs"].candidate_id.isin(ids)],
            "fb": D["fb"][D["fb"].candidate_id.isin(ids)]}


def active(c):
    return c[c.status == "Active"]


def live_offers(c):
    return c[(c.status == "Active") & c.current_stage.isin(SCORED_STAGES)]


def health(D):
    c, roles = D["cands"], D["roles"]
    act = active(c)
    tracked = act[act.sla_days.notna()]
    sla = 1 - tracked.past_sla.mean() if len(tracked) else 1.0
    offers = live_offers(c)
    safety = 1 - offers.risk.mean() / 100 if len(offers) else 1.0
    on_track = (roles.p_fill >= ON_TRACK).mean() if len(roles) else 1.0
    return dict(score=100 * (0.4 * sla + 0.3 * safety + 0.3 * on_track), sla=sla, safety=safety, on_track=on_track)


def whatif(row, hike_pct, notice_days, competing):
    """Re-score one candidate with modified offer levers."""
    f = {k: row[k] for k in metrics()["features"]}
    f.update(hike_pct=float(hike_pct), notice_days=int(notice_days), competing_offer=int(competing))
    risk, contrib = score_frame(pd.DataFrame([f]))
    return float(risk[0]), contrib[0]


def followup_draft(row, company):
    first = row["name"].split()[0]
    lead = (row["reasons"] or [None])[0]
    paras = {
        "hike_pct": "I've taken your compensation feedback to our hiring manager and we're reviewing what we can do.",
        "notice_days": "We understand your notice period is long and are happy to discuss a buyout or a flexible joining date.",
        "competing_offer": "I'd love to set up a quick call with the hiring manager so you can see the growth path here.",
        "relocation": "We can support your move with relocation assistance and a soft landing in the first month.",
        "days_since_last_reply": "It's been a few days since we last spoke, and I wanted to check that you have everything you need.",
    }
    body = paras.get(lead["feature"] if lead else "", "I wanted to check in and see if there's anything we can clarify.")
    return (f"Hi {first},\n\nHope you're doing well. {body}\n\nWe're excited about the possibility of you joining "
            f"{company} as {row['title']}, and I'm available for a call at a time that suits you.\n\n"
            f"Warm regards,\n{row['recruiter']}")


def source_table(c):
    """Hire rate and cost-effectiveness by source."""
    rows = []
    for s, g in c.groupby("source"):
        hires = g[(g.status == "Joined") | ((g.status == "Active") & (g.current_stage == "Accepted"))]
        k = SOURCE_COST[s]
        spend = k["per_applicant"] * len(g) + k["per_hire"] * len(hires) + k["ctc_share"] * (hires.offer_ctc.fillna(0).sum() * 1e5)
        rows.append(dict(source=s, applied=len(g), hires=len(hires), hire_rate=len(hires) / len(g) * 100,
                         spend=spend, cost_per_hire=spend / len(hires) if len(hires) else np.nan,
                         hires_per_lakh=len(hires) / (spend / 1e5) if spend else np.nan))
    return pd.DataFrame(rows).sort_values("hire_rate", ascending=False)
