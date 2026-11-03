import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from core import data, ui

BLIND_SPOTS = [  # text, what a human reads, the label a human would give
    ("Mujhe lagta hai ye offer kaafi kam hai, abhi doosre options bhi dekh raha hoon.", "Negative (Hinglish)", "Negative"),
    ("Bilkul maza nahi aaya process mein, bahut time lag raha hai.", "Negative (Hinglish)", "Negative"),
    ("Bahut accha laga team se milke, excited hoon aage badhne ke liye!", "Positive (Hinglish)", "Positive"),
    ("Another company has made a better offer. I'm worried about the notice buyout.", "Negative (risk)", "Negative"),
    ("Oh great, another two weeks of waiting. Really impressive speed.", "Negative (sarcasm)", "Negative"),
    ("The offer is not bad at all, I'm quite happy with it.", "Positive (negation)", "Positive"),
    ("I'll have to think about it.", "Cooling (neutral wording)", "Negative"),
    ("I have another offer in hand, will decide by Friday.", "Risk (no emotion words)", "Negative"),
    ("This was sick, the interview was insane!", "Positive (slang)", "Positive"),
]


@st.cache_resource
def _vader():
    return SentimentIntensityAnalyzer()


def _label(v):
    return "Positive" if v >= 0.05 else "Negative" if v <= -0.05 else "Neutral"


def render():
    ui.header("Lab", "How the signals are built, where they fail, and how much the model can be trusted.")
    vader = _vader()

    st.markdown("#### Try VADER")
    text = st.text_area("Type a candidate message", "Thanks for the update, but I'm disappointed with the hike.",
                        height=90, key="lab_text")
    s = vader.polarity_scores(text)
    a, b = st.columns([1, 2])
    a.metric("Compound", f"{s['compound']:+.2f}", _label(s["compound"]), delta_color="off")
    with b:
        fig = go.Figure(go.Bar(x=[s["neg"], s["neu"], s["pos"]], y=["Negative", "Neutral", "Positive"], orientation="h",
                               marker=dict(color=[ui.CRITICAL, ui.MUTED, ui.GOOD]),
                               hovertemplate="%{y}: %{x:.0%} of the text<extra></extra>"))
        fig.update_xaxes(range=[0, 1], tickformat=".0%")
        fig.update_yaxes(autorange="reversed")
        ui.show(fig, 150)

    st.markdown("#### VADER blind spots")
    rows = []
    for t, truth, expect in BLIND_SPOTS:
        c = vader.polarity_scores(t)["compound"]
        rows.append({"Message": t, "A human reads": truth, "VADER": c,
                     "Verdict": "Caught" if _label(c) == expect else "Missed"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"VADER": st.column_config.NumberColumn(format="%+.2f", width="small"),
                                "Verdict": st.column_config.TextColumn(width="small")})
    st.caption("VADER is an English lexicon. Hinglish, sarcasm and understated risk slip through, which is why the "
               "model also uses reply silence and structured offer terms.")

    m = data.metrics()
    best = {v["name"]: v for v in m["variants"]}
    full = best["All signals"]
    st.markdown("#### Model")
    ui.kpi_row([
        ("Historical offers", f"{m['n_offers']:,}", f"{m['dropout_rate']:.0%} dropped out"),
        ("AUC, all signals", f"{full['auc_mean']:.3f}", f"± {full['auc_std']:.3f} across 5 folds"),
        ("Log loss", f"{full['logloss']:.3f}", "lower is better"),
        ("Brier score", f"{full['brier']:.3f}", "calibration error"),
    ])
    names = [v["name"] for v in m["variants"]]
    fig = go.Figure(go.Bar(
        x=names, y=[v["auc_mean"] for v in m["variants"]],
        error_y=dict(type="data", array=[v["auc_std"] for v in m["variants"]], color=ui.MUTED),
        marker=dict(color=[ui.ACCENT if n != "Logistic regression (all signals)" else ui.MUTED for n in names]),
        hovertemplate="%{x}<br>AUC %{y:.3f}<extra></extra>"))
    fig.update_yaxes(range=[0.5, max(v["auc_mean"] for v in m["variants"]) + 0.05], title="5-fold CV AUC")
    ui.show(fig, 320)
    gain = full["auc_mean"] - best["Structured only"]["auc_mean"]
    lr_gap = best["Logistic regression (all signals)"]["auc_mean"] - full["auc_mean"]
    note = (" A plain logistic regression matches or beats the boosted trees here, so the trees add little on this "
            "synthetic data." if lr_gap >= -0.01 else " The boosted trees beat a plain logistic regression.")
    st.caption(f"Adding reply silence and message tone lifts AUC by {gain:.3f} over offer terms alone.{note}")

    st.markdown("#### Caveats")
    ui.html(ui.card("Read before trusting a score", (
        "The data is synthetic: the dropout label and the message tone are generated from a shared hidden intent, "
        "so real-world lift will be lower.<br>"
        "Sensitive attributes (gender, age, religion, caste, region of origin, disability) are excluded from the "
        "model and should stay excluded.<br>"
        "Scores prioritise human follow-up. They must never be used to auto-reject a candidate or withdraw an offer."),
        "Caveats", "warning"))
