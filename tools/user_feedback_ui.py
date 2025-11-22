# tools/user_feedback_ui.py
import os
import sys
import time
import subprocess
import pandas as pd
import yaml
import streamlit as st
import csv
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

FEEDBACK_CSV = os.path.join(DATA_DIR, "user_feedback.csv")
SUGGESTIONS_CSV = os.path.join(DATA_DIR, "merchant_suggestions.csv")
RETRAIN_SCRIPT = os.path.join(DATA_DIR, "retrain_with_feedback.py")
RETRAIN_LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(RETRAIN_LOG_DIR, exist_ok=True)

from data.predict_transaction import predict_transaction


# ---------- MERCHANT EXTRACTION ----------
UPI_RE = re.compile(r"\b([a-zA-Z0-9._-]{2,})@([a-zA-Z0-9._-]+)\b", flags=re.IGNORECASE)

def normalize(tok):
    if not tok:
        return ""
    tok = tok.lower().strip()
    tok = re.sub(r"[^\w]", "", tok)
    return tok

def extract_merchant(r):
    # 1. best source = matched span
    cand = normalize(r.get("matched_span_text", ""))
    if cand:
        return cand

    # 2. UPI id
    tx = r.get("transaction_text", "")
    m = UPI_RE.search(tx)
    if m:
        prefix = normalize(m.group(1))
        if prefix:
            return prefix

    # 3. context words
    tokens = re.split(r"\s+", tx)
    for i, tok in enumerate(tokens):
        if tok.lower() in ("to", "at", "for", "via", "paid"):
            if i + 1 < len(tokens):
                return normalize(tokens[i + 1])

    # 4. fallback longest alpha word
    alphas = [normalize(t) for t in tokens if re.search("[A-Za-z]", t)]
    if alphas:
        alphas.sort(key=lambda x: -len(x))
        return alphas[0]

    return ""


# ---------- SUGGESTION WRITE ----------
def append_suggestion(merchant, category):
    merchant = normalize(merchant)
    category = normalize(category)

    if not merchant or not category:
        return False

    # read existing to dedupe
    existing = set()
    if os.path.exists(SUGGESTIONS_CSV):
        with open(SUGGESTIONS_CSV, encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                existing.add((r["merchant"], r["suggested_category"]))

    if (merchant, category) in existing:
        return False

    header = ["merchant", "suggested_category", "timestamp", "source"]
    newfile = not os.path.exists(SUGGESTIONS_CSV)

    with open(SUGGESTIONS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if newfile:
            writer.writeheader()
        writer.writerow({
            "merchant": merchant,
            "suggested_category": category,
            "timestamp": pd.Timestamp.now().isoformat(),
            "source": "user"
        })

    return True


def append_feedback_row(row: dict):
    newfile = not os.path.exists(FEEDBACK_CSV)
    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if newfile:
            writer.writeheader()
        writer.writerow(row)


def run_retrain():
    subprocess.Popen([sys.executable, RETRAIN_SCRIPT], cwd=PROJECT_ROOT)


# ---------- UI ----------
st.set_page_config(page_title="User Feedback", layout="wide")
st.title("User Feedback – Predict & Review")

raw_input = st.text_area("Paste transactions (one per line)", height=180)

if st.button("Predict & Review"):
    lines = [l.strip() for l in raw_input.split("\n") if l.strip()]
    if not lines:
        st.warning("Enter text before predicting.")
    else:
        with st.spinner("Predicting..."):
            df = predict_transaction(lines)

        df["true_category"] = ""
        df["accept"] = False
        st.session_state["predicted_df"] = df
        st.success("Predicted. Review below and submit fixes.")


if "predicted_df" in st.session_state:
    st.subheader("Review & Correct")

    edited_df = st.data_editor(
        st.session_state["predicted_df"],
        use_container_width=True,
        num_rows="dynamic"
    )

    if st.button("Submit Feedback"):
        saved = 0
        sug_saved = 0

        for _, r in edited_df.iterrows():
            if r["accept"] and str(r["true_category"]).strip():
                fb_row = {
                    "transaction_text": r["transaction_text"],
                    "predicted_category": r["predicted_category"],
                    "true_category": r["true_category"],
                    "matched_span_text": r["matched_span_text"],
                    "source": "user",
                    "timestamp": pd.Timestamp.now().isoformat()
                }
                append_feedback_row(fb_row)
                saved += 1

                merchant = extract_merchant(r)  # correct string extraction
                if merchant:
                    if append_suggestion(merchant, r["true_category"]):
                        sug_saved += 1

        st.success(f"Saved {saved} correction(s), {sug_saved} new suggestion(s).")
        if saved > 0:
            run_retrain()
            st.info("Retraining started...")


st.markdown("---")
st.write("Feedback CSV:", FEEDBACK_CSV)
st.write("Suggestions CSV:", SUGGESTIONS_CSV)
