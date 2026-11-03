import streamlit as st

st.set_page_config(page_title="PipelineAI", page_icon=":material/insights:", layout="wide")

from core import data, ui  # noqa: E402
from views import ask, briefing, candidates, lab, offers, pipeline  # noqa: E402

ui.inject()

PAGES = {
    "briefing": st.Page(briefing.render, title="Briefing", icon=":material/wb_sunny:", url_path="briefing", default=True),
    "pipeline": st.Page(pipeline.render, title="Pipeline", icon=":material/filter_alt:", url_path="pipeline"),
    "candidates": st.Page(candidates.render, title="Candidates", icon=":material/groups:", url_path="candidates"),
    "offers": st.Page(offers.render, title="Offers", icon=":material/mark_email_read:", url_path="offers"),
    "ask": st.Page(ask.render, title="Ask", icon=":material/forum:", url_path="ask"),
    "lab": st.Page(lab.render, title="Lab", icon=":material/science:", url_path="lab"),
}
st.session_state["_pages"] = PAGES
nav = st.navigation(list(PAGES.values()))

D = data.get_data()
with st.sidebar:
    st.markdown('<div class="pa-brand">Pipeline<span>AI</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pa-muted">{D["company"]} hiring copilot</div>', unsafe_allow_html=True)
    st.selectbox("Viewing as", [data.ALL_RECRUITERS] + sorted(D["roles"].recruiter.unique()), key="viewer")
    st.markdown(f'<div class="pa-muted">{data.now_ist():%a, %d %b %Y · %I:%M %p} IST<br>'
                'Demo data, dates shifted to today.</div>', unsafe_allow_html=True)

nav.run()
