"""Styling, plotly template and small HTML components."""
import html as _html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BG, SURFACE, SURFACE_2 = "#0d0d0d", "#1a1a19", "#242422"
INK, INK_2, MUTED = "#f0efec", "#c3c2b7", "#898781"
ACCENT = "#3987e5"
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
TONES = {"good": GOOD, "warning": WARNING, "serious": SERIOUS, "critical": CRITICAL, "accent": ACCENT,
         "muted": MUTED, None: MUTED}
FONT = "Inter, -apple-system, 'Segoe UI', sans-serif"

pio.templates["pipelineai"] = go.layout.Template(layout=go.Layout(
    font=dict(family=FONT, color=INK_2, size=12),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=[ACCENT, "#7fb2f0", WARNING, SERIOUS, GOOD, CRITICAL],
    xaxis=dict(gridcolor="#262625", zerolinecolor="#262625", linecolor="#2f2f2d", tickfont=dict(color=MUTED)),
    yaxis=dict(gridcolor="#262625", zerolinecolor="#262625", linecolor="#2f2f2d", tickfont=dict(color=MUTED)),
    hoverlabel=dict(bgcolor=SURFACE_2, bordercolor="#3a3a37", font=dict(family=FONT, color=INK, size=12)),
    margin=dict(l=8, r=8, t=28, b=8), legend=dict(font=dict(color=INK_2)),
))
pio.templates.default = "pipelineai"
PLOT_CONFIG = {"displayModeBar": False}

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp *:not([data-testid="stIconMaterial"]):not(.material-icons) {{ font-family: {FONT}; }}
[data-testid="stIconMaterial"] {{ font-family: "Material Symbols Rounded" !important; }}
.stApp {{ background: {BG}; color: {INK}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid #262625; }}
h1, h2, h3 {{ letter-spacing: -0.02em; color: {INK}; }}
.pa-brand {{ font-weight: 700; font-size: 1.25rem; letter-spacing: -0.02em; color: {INK}; }}
.pa-brand span {{ color: {ACCENT}; }}
.pa-muted {{ color: {MUTED}; font-size: .82rem; }}
.pa-card {{ background: {SURFACE}; border: 1px solid #262625; border-radius: 16px; padding: 16px 18px; margin-bottom: 12px; }}
.pa-card h4 {{ margin: 0 0 6px 0; font-size: .95rem; color: {INK}; font-weight: 600; }}
.pa-card p {{ margin: 0 0 4px 0; color: {INK_2}; font-size: .85rem; line-height: 1.45; }}
.pa-tag {{ font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }}
.pa-kpi {{ background: {SURFACE}; border: 1px solid #262625; border-radius: 16px; padding: 14px 18px; }}
.pa-kpi .l {{ color: {MUTED}; font-size: .78rem; }}
.pa-kpi .v {{ color: {INK}; font-size: 1.9rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1.2; }}
.pa-kpi .s {{ color: {MUTED}; font-size: .78rem; }}
.pa-chip {{ display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: .72rem; font-weight: 600;
  margin-right: 6px; border: 1px solid; }}
.pa-bar {{ background: #2a2a28; border-radius: 999px; height: 6px; overflow: hidden; margin: 6px 0; }}
.pa-bar > div {{ height: 100%; border-radius: 999px; }}
.pa-kcol {{ color: {INK}; font-weight: 600; font-size: .9rem; margin-bottom: 8px; }}
.pa-kcard {{ background: {SURFACE}; border: 1px solid #262625; border-radius: 14px; padding: 10px 12px; margin-bottom: 4px; }}
.pa-kcard.sel {{ border-color: {ACCENT}; }}
.pa-kcard .n {{ color: {INK}; font-weight: 600; font-size: .86rem; }}
.pa-kcard .r {{ color: {MUTED}; font-size: .74rem; margin-bottom: 6px; }}
.stButton > button {{ border-radius: 10px; border: 1px solid #333330; background: {SURFACE_2}; color: {INK}; }}
.stButton > button:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
[data-testid="stMetricValue"] {{ font-weight: 700; }}
"""


def inject():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def e(text):
    return _html.escape(str(text))


def html(markup):
    st.html(markup)


def chip(text, tone="muted"):
    c = TONES[tone]
    return f'<span class="pa-chip" style="color:{c};border-color:{c}55;background:{c}1a">{e(text)}</span>'


def bar(pct, tone="accent"):
    pct = max(0, min(100, pct))
    return f'<div class="pa-bar"><div style="width:{pct:.0f}%;background:{TONES[tone]}"></div></div>'


def card(title, body, tag=None, tone="accent", min_h=0):
    t = f'<div class="pa-tag" style="color:{TONES[tone]}">{e(tag)}</div>' if tag else ""
    return f'<div class="pa-card" style="min-height:{min_h}px">{t}<h4>{e(title)}</h4><p>{body}</p></div>'


def kpi(label, value, sub="", tone=None):
    color = f"color:{TONES[tone]};" if tone else ""
    return (f'<div class="pa-kpi"><div class="l">{e(label)}</div><div class="v" style="{color}">{e(value)}</div>'
            f'<div class="s">{e(sub)}</div></div>')


def kpi_row(items):
    for col, it in zip(st.columns(len(items)), items):
        with col:
            html(kpi(*it))


def header(title, sub=""):
    st.markdown(f"## {title}")
    if sub:
        st.markdown(f'<div class="pa-muted" style="margin:-8px 0 14px 0">{e(sub)}</div>', unsafe_allow_html=True)


def style(fig, height=320):
    fig.update_layout(height=height, template="pipelineai")
    return fig


def show(fig, height=320):
    st.plotly_chart(style(fig, height), width="stretch", config=PLOT_CONFIG)


def gauge(score):
    tone = GOOD if score >= 75 else WARNING if score >= 55 else SERIOUS if score >= 40 else CRITICAL
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(font=dict(size=44, color=INK), valueformat=".0f"),
        gauge=dict(axis=dict(range=[0, 100], tickcolor=MUTED, tickfont=dict(color=MUTED, size=10)),
                   bar=dict(color=tone, thickness=0.28), bgcolor="#2a2a28", borderwidth=0)))
    return fig
