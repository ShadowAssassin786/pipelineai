# PipelineAI

A hiring-copilot dashboard for recruiting teams. It shows what needs attention today, which offers are likely to fall through and why, which roles will miss their deadline, and lets you ask questions in plain language.

All data is synthetic. All machine learning runs offline, so the deployed app only loads saved files and stays light (well under 1 GB RAM).

## Pages

| Page | What it does |
|---|---|
| Briefing | IST greeting, pipeline health gauge, KPIs, forecast, and "Needs you today" cards |
| Pipeline | Funnel, past-SLA heatmap by role and stage, source hire rate and cost, time to hire |
| Candidates | Kanban sorted by urgency, tone timeline, SHAP bar, what-if levers, follow-up draft |
| Offers | Live dropout risk table with reasons, plus global feature importance |
| Ask | Question pills and chat; rule-based router with an optional LLM fallback |
| Lab | Live VADER box, VADER blind spots, model metrics, ablation chart, caveats |

The sidebar "Viewing as" selector scopes every page to one recruiter.

## Run locally

Python 3.12 is the deploy target. Local runs were also checked on 3.14.

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements-scripts.txt
python scripts/generate_data.py
python scripts/precompute.py
streamlit run app.py
```

The generated files in `data/raw` and `data/processed` are committed, so you can skip the two scripts and just run `streamlit run app.py` with `pip install -r requirements.txt`.

## How it works

- `scripts/generate_data.py` (seed 42): 12 roles, 600 candidates walked through Applied, Screened, Interview, Offer, Accepted and Joined, candidate messages, interviewer feedback, and 1,800 historical offers. Dropout depends on nonlinear offer terms plus a hidden intent that also drives message tone arcs and reply silence, so engagement signals carry real information. About 11% of feedback notes disagree with their rating on purpose.
- `scripts/precompute.py`: VADER scores for every message and note, mismatch flags (note ≥ 0.6 with rating ≤ 2, or note ≤ -0.4 with rating ≥ 4), per-candidate features, a 5-fold CV ablation, the final XGBoost model (`model.json`) and `metrics.json` with mean |SHAP|.
- `core/data.py`: shifts every date by (today in IST minus the anchor date) so the demo always looks live, applies SLA rules (Applied 5, Screened 5, Interview 7, Offer 5 days), scores Offer and Accepted candidates with the saved booster, and builds the role forecast and health score.
- Risk levels: below 30 Low, below 50 Medium, below 70 High, otherwise Critical.
- Health score = 0.4 × SLA compliance + 0.3 × offer safety + 0.3 × roles on track.

## Model metrics

5-fold stratified cross-validation on 1,800 historical offers (35% dropped out). XGBoost uses depth 2, 300 trees, learning rate 0.03.

| Variant | AUC (mean ± std) | Log loss |
|---|---|---|
| XGBoost, structured offer terms only | 0.647 ± 0.018 | 0.618 |
| XGBoost, + reply silence | 0.688 ± 0.009 | 0.595 |
| XGBoost, all signals (used in the app) | 0.726 ± 0.024 | 0.574 |
| Logistic regression, all signals | 0.739 ± 0.010 | 0.564 |

Reply silence and message tone add about 0.08 AUC over offer terms alone. Logistic regression matches or slightly beats the boosted trees on this data, so the trees are chosen for per-candidate SHAP explanations rather than accuracy. Because the label and the tone are generated from the same hidden intent, expect lower lift on real data.

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository, including `data/raw` and `data/processed`.
2. Go to [share.streamlit.io](https://share.streamlit.io) and choose **Create app**.
3. Select the repository and branch, and set the main file to `app.py`.
4. Under **Advanced settings**, choose Python 3.12.
5. Deploy. Dependencies come from `requirements.txt`; `scikit-learn` is only in `requirements-scripts.txt` because the scripts need it and the app does not.

### Optional: language-model fallback on the Ask page

Questions the rule-based router cannot place can go to the Anthropic API. In the app's **Settings > Secrets** (or a local `.streamlit/secrets.toml`) add:

```toml
ANTHROPIC_API_KEY = "your-key"
# ANTHROPIC_MODEL = "claude-haiku-5-5"   # optional override
```

Without a key the Ask page replies with the list of questions it can answer.

## Responsible use

- The data is synthetic.
- Sensitive attributes (gender, age, religion, caste, region of origin, disability) are excluded from the model and should stay excluded.
- Risk scores prioritise human follow-up. Never use them to auto-reject a candidate or withdraw an offer.
