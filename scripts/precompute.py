"""Score text with VADER, build features, train and evaluate the dropout model."""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parent.parent
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
SEED = 42

STRUCT = ["hike_pct", "notice_days", "competing_offer", "relocation", "source_quality", "ctc_current",
          "days_in_process"]
SILENCE = ["days_since_last_reply"]
SENT = ["sentiment_latest", "sentiment_change"]
ALL = STRUCT + SILENCE + SENT
SOURCE_QUALITY = {"LinkedIn": 1.00, "Naukri": 0.80, "Instahyre": 1.05, "Careers page": 0.90, "Referral": 1.35,
                  "Agency": 1.15}

vader = SentimentIntensityAnalyzer()


def compound(text):
    return vader.polarity_scores(text)["compound"]


def sentiment_features(scores):
    """scores: compounds in chronological order -> (latest, change)."""
    n = len(scores)
    if n == 0:
        return np.nan, np.nan
    latest = float(np.mean(scores[-2:]))
    if n >= 3:
        change = latest - float(np.mean(scores[:-2]))
    elif n == 2:
        change = scores[1] - scores[0]
    else:
        change = 0.0
    return latest, change


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    meta = json.loads((RAW / "meta.json").read_text())
    anchor = date.fromisoformat(meta["anchor"])

    # ---- text scoring
    msgs = pd.read_csv(RAW / "messages.csv", parse_dates=["date"])
    msgs["compound"] = msgs.text.map(compound)
    msgs.to_csv(PROC / "messages_scored.csv", index=False)

    fb = pd.read_csv(RAW / "feedback.csv")
    fb["compound"] = fb.note.map(compound)
    fb["mismatch"] = ((fb.compound >= 0.6) & (fb.rating <= 2)) | ((fb.compound <= -0.4) & (fb.rating >= 4))
    fb.to_csv(PROC / "feedback_scored.csv", index=False)
    print(f"messages={len(msgs)} feedback={len(fb)} mismatches={int(fb.mismatch.sum())} "
          f"({fb.mismatch.mean():.1%})")

    # ---- live candidate features (anchor frame; runtime shifts dates, not these gaps)
    cands = pd.read_csv(RAW / "candidates.csv", parse_dates=["applied_date"])
    anchor_ts = pd.Timestamp(anchor)
    rows = []
    grouped = {k: g.sort_values("date") for k, g in msgs.groupby("candidate_id")}
    for c in cands.itertuples():
        g = grouped.get(c.candidate_id)
        latest, change, silence, n = np.nan, np.nan, np.nan, 0
        if g is not None:
            latest, change = sentiment_features(g.compound.tolist())
            silence, n = (anchor_ts - g.date.max()).days, len(g)
        price = c.offer_ctc if pd.notna(c.offer_ctc) else c.expected_ctc
        rows.append(dict(
            candidate_id=c.candidate_id, hike_pct=(price / c.ctc_current - 1) * 100, notice_days=c.notice_days,
            competing_offer=c.competing_offer, relocation=c.relocation,
            source_quality=SOURCE_QUALITY[c.source], ctc_current=c.ctc_current,
            days_in_process=(anchor_ts - c.applied_date).days, days_since_last_reply=silence,
            sentiment_latest=latest, sentiment_change=change, n_messages=n))
    pd.DataFrame(rows).to_csv(PROC / "candidate_features.csv", index=False)

    # ---- historical offers -> training frame
    offers = pd.read_csv(RAW / "offers_history.csv")
    om = pd.read_csv(RAW / "offer_messages.csv")
    om["compound"] = om.text.map(compound)
    feats = {}
    for oid, g in om.groupby("offer_id"):
        g = g.sort_values("days_before_snapshot", ascending=False)  # oldest first
        lat, chg = sentiment_features(g.compound.tolist())
        feats[oid] = (lat, chg, int(g.days_before_snapshot.min()))
    offers["sentiment_latest"] = offers.offer_id.map(lambda o: feats[o][0])
    offers["sentiment_change"] = offers.offer_id.map(lambda o: feats[o][1])
    offers["days_since_last_reply"] = offers.offer_id.map(lambda o: feats[o][2])
    offers.to_csv(PROC / "offers_features.csv", index=False)

    y = offers.dropout.values
    params = dict(max_depth=2, n_estimators=300, learning_rate=0.03, random_state=SEED, n_jobs=1,
                  eval_metric="logloss", tree_method="hist")
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)

    def cv(name, cols, make):
        X = offers[cols]
        aucs, ll, br = [], [], []
        for tr, te in skf.split(X, y):
            m = make().fit(X.iloc[tr], y[tr])
            p = m.predict_proba(X.iloc[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
            ll.append(log_loss(y[te], p))
            br.append(brier_score_loss(y[te], p))
        return dict(name=name, features=cols, auc_mean=float(np.mean(aucs)), auc_std=float(np.std(aucs)),
                    logloss=float(np.mean(ll)), brier=float(np.mean(br)))

    xgbm = lambda: xgb.XGBClassifier(**params)
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    variants = [
        cv("Structured only", STRUCT, xgbm),
        cv("+ Reply silence", STRUCT + SILENCE, xgbm),
        cv("All signals", ALL, xgbm),
        cv("Logistic regression (all signals)", ALL, lr),
    ]
    for v in variants:
        print(f"{v['name']:<38} AUC {v['auc_mean']:.3f} ± {v['auc_std']:.3f}  logloss {v['logloss']:.3f}")

    final = xgb.XGBClassifier(**params).fit(offers[ALL], y)
    booster = final.get_booster()
    booster.save_model(PROC / "model.json")
    contribs = booster.predict(xgb.DMatrix(offers[ALL], feature_names=ALL), pred_contribs=True)
    shap = np.abs(contribs[:, :-1]).mean(axis=0)
    metrics = dict(n_offers=int(len(offers)), dropout_rate=float(y.mean()), features=ALL, variants=variants,
                   shap_importance={f: float(s) for f, s in zip(ALL, shap)})
    (PROC / "metrics.json").write_text(json.dumps(metrics, indent=2))
    top = sorted(metrics["shap_importance"].items(), key=lambda kv: -kv[1])[:4]
    print("top |SHAP|:", ", ".join(f"{k}={v:.2f}" for k, v in top))


if __name__ == "__main__":
    main()
