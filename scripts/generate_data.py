"""Generate the synthetic hiring dataset (seed 42) into data/raw."""
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ANCHOR = date(2025, 6, 2)
COMPANY = "Lumora Technologies"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

STAGES = ["Applied", "Screened", "Interview", "Offer", "Accepted", "Joined"]
SOURCES = {  # name: (application share, quality multiplier)
    "LinkedIn": (0.27, 1.00),
    "Naukri": (0.22, 0.80),
    "Instahyre": (0.13, 1.05),
    "Careers page": (0.15, 0.90),
    "Referral": (0.12, 1.35),
    "Agency": (0.11, 1.15),
}
PASS_P = {"Applied": 0.66, "Screened": 0.64, "Interview": 0.45}
DWELL_MEAN = {"Applied": 3.5, "Screened": 4.0, "Interview": 6.0, "Offer": 4.0}
RECRUITERS = ["Aarti Nair", "Rohan Mehta", "Sneha Iyer", "Vikram Rao"]
INTERVIEWERS = ["Kabir Sethi", "Neha Bhatia", "Arjun Malhotra", "Divya Menon", "Sanjay Kulkarni", "Ritu Ahluwalia"]
ROLES = [  # title, dept, city, base CTC (LPA), weight
    ("Senior Backend Engineer", "Engineering", "Bengaluru", 32, 1.4),
    ("Frontend Engineer", "Engineering", "Pune", 22, 1.2),
    ("Data Scientist", "Data", "Hyderabad", 30, 1.0),
    ("Data Engineer", "Data", "Bengaluru", 26, 1.0),
    ("Product Manager", "Product", "Gurugram", 40, 0.9),
    ("UX Designer", "Design", "Mumbai", 24, 0.8),
    ("DevOps Engineer", "Engineering", "Chennai", 24, 0.9),
    ("Sales Manager", "Sales", "Mumbai", 28, 0.9),
    ("Account Executive", "Sales", "Noida", 18, 1.1),
    ("Marketing Lead", "Marketing", "Gurugram", 38, 0.6),
    ("HR Business Partner", "People", "Pune", 22, 0.6),
    ("Finance Analyst", "Finance", "Hyderabad", 14, 0.7),
]
MANAGERS = ["Meera Kapoor", "Aditya Joshi", "Pooja Deshmukh", "Karan Bansal", "Ishita Rao", "Nikhil Verma",
            "Tanvi Shah", "Gaurav Mishra", "Lakshmi Pillai", "Harsh Vora", "Simran Gill", "Yash Agarwal"]
FIRST = ["Aarav", "Vivaan", "Aditya", "Ishaan", "Arjun", "Sai", "Reyansh", "Krishna", "Dev", "Rahul", "Ananya", "Diya",
         "Saanvi", "Aadhya", "Myra", "Kiara", "Pari", "Riya", "Navya", "Tara", "Rohit", "Manish", "Priya", "Neha",
         "Siddharth", "Varun", "Pranav", "Kavya", "Shreya", "Mohit", "Zoya", "Imran", "Farhan", "Gurpreet", "Harleen",
         "Joseph", "Anjali", "Deepak", "Swati", "Tushar"]
LAST = ["Sharma", "Verma", "Iyer", "Nair", "Reddy", "Patel", "Gupta", "Singh", "Khan", "Mehta", "Das", "Bose", "Menon",
        "Pillai", "Joshi", "Kulkarni", "Desai", "Chopra", "Malhotra", "Sethi", "Bhatt", "Rao", "Naidu", "Banerjee",
        "Chatterjee", "Kaur", "Thakur", "Yadav", "Jain", "Agarwal"]

MSG = {
    "pos": [
        "Thanks for the update! I'm really excited about the {role} role and happy to move ahead.",
        "Great, that works for me. Looking forward to the next round!",
        "Thank you so much for the quick turnaround. This sounds like a fantastic opportunity.",
        "Happy with the discussion so far, the team seems wonderful. Please share the next steps.",
        "Bahut accha laga team se milke, excited hoon aage badhne ke liye!",
        "Sounds great, I'm keen to join and will share the documents today.",
        "Love the details, thanks for being so transparent. Really appreciate it!",
        "Awesome, thank you! I'm confident this is a good fit and I'm very interested.",
    ],
    "neu": [
        "Received your email. I will go through it and revert.",
        "Can we reschedule the call to Thursday afternoon?",
        "Please share the job description and the compensation structure.",
        "Noted. I'll get back to you by end of week.",
        "Mujhe thoda time chahiye, main weekend tak batata hoon.",
        "Could you confirm the location and the reporting manager?",
        "Ok, sending my updated resume as requested.",
        "I am available on Monday or Tuesday after 4 pm.",
    ],
    "neg": [
        "I'm not sure this is the right fit anymore, the compensation is disappointing.",
        "Honestly I'm frustrated with the delays, and I have other offers waiting.",
        "The process has been too slow and I'm losing interest.",
        "Sorry, I'm no longer comfortable with this timeline. I need to rethink.",
        "Mujhe lagta hai ye offer kaafi kam hai, abhi doosre options bhi dekh raha hoon.",
        "Bilkul maza nahi aaya process mein, bahut time lag raha hai.",
        "Another company has made a better offer. I'm worried about the notice buyout.",
        "Not happy with the hike offered, I was expecting much more. This is upsetting.",
    ],
}
FB = {
    "pos": [
        "Excellent problem solving and a great communicator. Strong hire, impressive depth and a wonderful attitude.",
        "Outstanding technical skills, brilliant approach to design. Really impressive and a pleasure to interview.",
        "Very strong candidate, clear thinker and confident. Definitely recommend, great culture fit.",
    ],
    "neu": [
        "Decent fundamentals. Average depth on system design, communication was okay.",
        "Met the bar on the basics but nothing stood out. Mixed signals on ownership.",
        "Reasonable answers overall, some gaps in domain knowledge. Could go either way.",
    ],
    "neg": [
        "Poor grasp of fundamentals and weak communication. Struggled badly, terrible fit for the team.",
        "Disappointing interview, could not solve the basic problem. Not a fit, would not hire.",
        "Weak answers throughout and very unprepared. Bad attitude towards feedback, I have serious concerns.",
    ],
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dropout_prob(hike, notice, comp, reloc, srcq, ctc, days_in, intent):
    """Nonlinear offer dropout probability; `intent` is a hidden engagement trait."""
    z = (-3.1
         + 2.2 * sigmoid((12 - hike) / 3.0)
         + 1.6 * (notice / 90.0) ** 2.2
         + comp * (0.9 + 1.4 * sigmoid((18 - hike) / 4.0))
         + reloc * (0.2 + 0.5 * (notice >= 60))
         - 1.6 * (srcq - 1.0)
         + 0.04 * np.maximum(days_in - 35, 0)
         + 0.02 * np.maximum(ctc - 30, 0)
         - 1.9 * intent)
    return sigmoid(z)


def main():
    rng = np.random.default_rng(SEED)
    RAW.mkdir(parents=True, exist_ok=True)
    iso = lambda d: d.isoformat()

    def pick(lst):
        return lst[int(rng.integers(len(lst)))]

    def arc_for(intent):
        r = rng.normal(intent, 0.5)
        if r > 0.7:
            return "engaged"
        if r > 0.0:
            return "warming" if rng.random() < 0.45 else "flat"
        if r > -0.7:
            return "cooling" if rng.random() < 0.5 else "flat"
        return "cooling"

    def tone_probs(arc, t):
        if arc == "engaged":
            return [0.70, 0.25, 0.05]
        if arc == "flat":
            return [0.15, 0.70, 0.15]
        if arc == "cooling":
            return [0.70 * (1 - t) + 0.05, 0.27, 0.70 * t + 0.03]
        return [0.70 * t + 0.05, 0.27, 0.70 * (1 - t) + 0.03]

    def make_messages(arc, n, role):
        out = []
        for i in range(n):
            t = i / max(n - 1, 1)
            p = np.array(tone_probs(arc, t))
            tone = ["pos", "neu", "neg"][int(rng.choice(3, p=p / p.sum()))]
            out.append(pick(MSG[tone]).format(role=role))
        return out

    def silence_for(intent):
        return int(rng.exponential(1.0 + 9.0 * sigmoid(-1.6 * intent)))

    # ---- roles
    roles = []
    for i, (title, dept, city, base, w) in enumerate(ROLES):
        roles.append(dict(role_id=f"R{i + 1:02d}", title=title, dept=dept, city=city,
                          recruiter=RECRUITERS[i % 4], hiring_manager=MANAGERS[i], base_ctc=base,
                          target_close=iso(ANCHOR + timedelta(days=int(rng.integers(12, 71)))), weight=w))
    stage_mult = {r["role_id"]: {s: float(rng.lognormal(0, 0.3)) for s in DWELL_MEAN} for r in roles}
    for rid in rng.choice(list(stage_mult), 4, replace=False):  # deliberate bottlenecks
        stage_mult[rid][str(pick(["Screened", "Interview", "Offer"]))] *= 1.9

    # ---- candidates
    n_total = 600
    wts = np.array([r["weight"] for r in roles])
    counts = np.floor(wts / wts.sum() * n_total).astype(int)
    counts[: n_total - counts.sum()] += 1
    src_names = list(SOURCES)
    src_p = np.array([SOURCES[s][0] for s in src_names])
    cands, msgs, fbs = [], [], []
    used = set()
    for role, cnt in zip(roles, counts):
        for _ in range(cnt):
            cid = f"C{len(cands) + 1:04d}"
            while True:
                name = f"{pick(FIRST)} {pick(LAST)}"
                if name not in used:
                    used.add(name)
                    break
            src = src_names[int(rng.choice(len(src_names), p=src_p / src_p.sum()))]
            q = SOURCES[src][1]
            applied = ANCHOR - timedelta(days=int(min(rng.exponential(32), 120)))
            ctc = round(role["base_ctc"] * float(rng.lognormal(0, 0.2)), 1)
            hike = float(np.clip(rng.normal(19, 8), 2, 55))
            notice = int(rng.choice([15, 30, 60, 90], p=[0.25, 0.4, 0.25, 0.1]))
            comp = int(rng.random() < 0.28)
            reloc = int(rng.random() < 0.22)
            intent = float(rng.normal())
            expected = round(ctc * float(rng.uniform(1.1, 1.5)), 1)
            offer_ctc = None

            hist, status, t, noshow, end = [], "Active", applied, False, None
            for stage in STAGES:
                entered = t
                if stage == "Joined":
                    hist.append(dict(stage=stage, entered=iso(entered), exited=None))
                    status, end = "Joined", entered
                    break
                if stage == "Offer":
                    offer_ctc = round(ctc * (1 + hike / 100), 1)
                if stage == "Accepted":
                    dwell = max(7, notice + int(rng.integers(-3, 4)))
                else:
                    mean = DWELL_MEAN[stage] * stage_mult[role["role_id"]][stage]
                    dwell = max(1, int(round(rng.gamma(2.0, mean / 2.0))))
                exit_t = entered + timedelta(days=dwell)
                if stage == "Accepted" and noshow:
                    wd = entered + timedelta(days=int(rng.integers(2, max(dwell - 1, 3))))
                    if wd <= ANCHOR:
                        hist.append(dict(stage=stage, entered=iso(entered), exited=iso(wd)))
                        status, end = "Withdrawn", wd
                        break
                    hist.append(dict(stage=stage, entered=iso(entered), exited=None))
                    break
                if exit_t > ANCHOR:
                    hist.append(dict(stage=stage, entered=iso(entered), exited=None))
                    break
                hist.append(dict(stage=stage, entered=iso(entered), exited=iso(exit_t)))
                t = exit_t
                if stage == "Offer":
                    days_in = (entered - applied).days
                    if rng.random() < dropout_prob(hike, notice, comp, reloc, q, ctc, days_in, intent):
                        if rng.random() < 0.65:
                            status, end = "Declined", exit_t
                            break
                        noshow = True
                elif stage in PASS_P and rng.random() > min(PASS_P[stage] * q, 0.95):
                    status = "Withdrawn" if rng.random() < 0.15 else "Rejected"
                    end = exit_t
                    break
            cur = hist[-1]["stage"]
            reached = {h["stage"] for h in hist}
            cands.append(dict(
                candidate_id=cid, name=name, role_id=role["role_id"], source=src, applied_date=iso(applied),
                current_stage=cur, status=status, ctc_current=ctc, expected_ctc=expected,
                offer_ctc=offer_ctc if "Offer" in reached else None, notice_days=notice,
                competing_offer=comp, relocation=reloc, stage_history=json.dumps(hist)))

            # messages
            if "Screened" in reached:
                s0 = next(date.fromisoformat(h["entered"]) for h in hist if h["stage"] == "Screened")
                stop = end if end else ANCHOR
                last = max(stop - timedelta(days=silence_for(intent)), s0)
                span = (last - s0).days
                n = int(rng.integers(2, 7))
                offs = np.sort(rng.integers(0, span + 1, n))
                for off, text in zip(offs, make_messages(arc_for(intent), n, role["title"])):
                    msgs.append(dict(candidate_id=cid, date=iso(s0 + timedelta(days=int(off))), text=text))

            # interviewer feedback
            if "Interview" in reached:
                ih = next(h for h in hist if h["stage"] == "Interview")
                i0 = date.fromisoformat(ih["entered"])
                if "Offer" in reached:
                    mean_r = 4.1
                elif status == "Active" and cur == "Interview":
                    mean_r = 3.4
                else:
                    mean_r = 2.6
                rounds = 2 if ("Offer" in reached and rng.random() < 0.7) else 1
                for rd in range(1, rounds + 1):
                    d = i0 + timedelta(days=int(rng.integers(0, 4)))
                    if d > ANCHOR:
                        continue
                    rating = int(np.clip(round(rng.normal(mean_r, 0.8)), 1, 5))
                    tone = "pos" if rating >= 4 else "neu" if rating == 3 else "neg"
                    if rating != 3 and rng.random() < 0.14:  # deliberate tone/rating mismatch
                        tone = "neg" if rating >= 4 else "pos"
                    fbs.append(dict(candidate_id=cid, round=rd, interviewer=pick(INTERVIEWERS), date=iso(d),
                                    rating=rating, note=pick(FB[tone])))

    cdf = pd.DataFrame(cands)
    # headcount = secured + open seats
    secured = cdf[(cdf.status == "Joined") | ((cdf.status == "Active") & (cdf.current_stage == "Accepted"))]
    sec = secured.groupby("role_id").size()
    for r in roles:
        r["secured"] = int(sec.get(r["role_id"], 0))
        r["open_seats"] = int(rng.integers(1, 5))
        r["headcount"] = r["secured"] + r["open_seats"]

    # ---- historical offers
    n_off = 1800
    off_rows, off_msgs = [], []
    for k in range(n_off):
        oid = f"H{k + 1:04d}"
        src = src_names[int(rng.choice(len(src_names), p=src_p / src_p.sum()))]
        q = SOURCES[src][1]
        hike = float(np.clip(rng.normal(19, 8), 2, 55))
        notice = int(rng.choice([15, 30, 60, 90], p=[0.25, 0.4, 0.25, 0.1]))
        comp, reloc = int(rng.random() < 0.28), int(rng.random() < 0.22)
        ctc = round(float(rng.lognormal(np.log(26), 0.35)), 1)
        days_in = int(rng.integers(12, 76))
        intent = float(rng.normal())
        drop = int(rng.random() < dropout_prob(hike, notice, comp, reloc, q, ctc, days_in, intent))
        off_rows.append(dict(offer_id=oid, source=src, source_quality=q, ctc_current=ctc, hike_pct=round(hike, 1),
                             notice_days=notice, competing_offer=comp, relocation=reloc, days_in_process=days_in,
                             dropout=drop))
        n = int(rng.integers(2, 7))
        sil = silence_for(intent)
        gaps = np.cumsum(rng.integers(1, 12, n - 1)) if n > 1 else []
        offsets = [sil] + [sil + int(g) for g in gaps]  # newest first
        for off, text in zip(offsets, reversed(make_messages(arc_for(intent), n, "role"))):
            off_msgs.append(dict(offer_id=oid, days_before_snapshot=off, text=text))

    pd.DataFrame(roles).drop(columns=["weight"]).to_csv(RAW / "roles.csv", index=False)
    cdf.to_csv(RAW / "candidates.csv", index=False)
    pd.DataFrame(msgs).to_csv(RAW / "messages.csv", index=False)
    pd.DataFrame(fbs).to_csv(RAW / "feedback.csv", index=False)
    pd.DataFrame(off_rows).to_csv(RAW / "offers_history.csv", index=False)
    pd.DataFrame(off_msgs).to_csv(RAW / "offer_messages.csv", index=False)
    (RAW / "meta.json").write_text(json.dumps({"anchor": iso(ANCHOR), "company": COMPANY, "seed": SEED}))

    act = cdf[cdf.status == "Active"]
    print(f"roles={len(roles)} candidates={len(cdf)} messages={len(msgs)} feedback={len(fbs)} offers={n_off}")
    print("status:", cdf.status.value_counts().to_dict())
    print("active by stage:", act.current_stage.value_counts().to_dict())
    print(f"historical dropout rate: {pd.DataFrame(off_rows).dropout.mean():.3f}")


if __name__ == "__main__":
    main()
