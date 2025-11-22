"""
predict_transaction.py - updated with live progress logs
"""

import os
import yaml
import joblib
import pandas as pd
import time
from typing import List, Dict

from data.merchant_lookup import MerchantLookup

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "../models/baseline_model.pkl")
VECT_PATH = os.path.join(BASE_DIR, "../models/tfidf_vectorizer.pkl")
TAX_PATH = os.path.join(BASE_DIR, "../config/taxonomy.yaml")

CONFIDENCE_THRESHOLD = 0.9  # threshold for merchant-based classification

def load_taxonomy(path: str) -> Dict:
    print(f"📂 Loading taxonomy from: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    print("✅ Taxonomy loaded")
    return data


def predict_transaction(texts: List[str]) -> pd.DataFrame:
    print("🚀 Starting prediction pipeline")
    t0 = time.time()

    tax = load_taxonomy(TAX_PATH)
    print("🔧 Initializing Merchant Lookup...")
    ml = MerchantLookup(tax)
    print("🔧 Merchant Lookup Ready!")

    fallback_model = None
    fallback_vect = None
    try:
        print("📦 Loading fallback baseline model...")
        fallback_model = joblib.load(MODEL_PATH)
        fallback_vect = joblib.load(VECT_PATH)
        print("✅ Baseline classifier loaded")
    except Exception:
        print("⚠️ No baseline classifier found, fallback disabled.")

    rows = []

    for idx, t in enumerate(texts):
        print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"🔍 Processing transaction [{idx+1}/{len(texts)}]: {t}")
        st = time.time()

        # matched_* = values found by merchant lookup (kept even if low confidence)
        matched_cat = None
        matched_conf = 0.0
        matched_span = None

        best = ml.best_match_for_text(t, top_k=3)
        if best is not None:
            # best may be tuple like (cat, score, span, alias_text, alias_source, alias_weight)
            matched_cat, matched_conf, matched_span, *_ = best
            print(f"📌 Merchant lookup → category={matched_cat} score={matched_conf:.3f} span={matched_span}")

        final_category = None
        final_conf = matched_conf

        if matched_conf >= CONFIDENCE_THRESHOLD:
            print("🎯 Using merchant-based classification (high confidence)")
            final_category = tax.get("categories", {}).get(matched_cat, {}).get("display_name", matched_cat)
        else:
            # if we had a lookup but it was low confidence, log it but keep matched metadata
            if matched_cat is not None:
                print(f"⚠ Merchant lookup found '{matched_cat}' (span='{matched_span}') but low confidence ({matched_conf:.3f}) -> falling back to classifier")
            else:
                print("🤖 No merchant lookup match with sufficient confidence, falling back to ML classifier...")

            if fallback_model is not None and fallback_vect is not None:
                x = fallback_vect.transform([t])
                probs = fallback_model.predict_proba(x)[0]
                best_idx = probs.argmax()
                final_category = fallback_model.classes_[best_idx]
                final_conf = float(probs[best_idx]) * 0.9
                print(f"📎 Fallback predicted category={final_category}, confidence={final_conf:.3f}")
            else:
                print("⚠️ No fallback classifier available → category = unknown")
                final_category = "unknown"
                final_conf = 0.0

        # clamp final confidence into [0, 1]
        final_conf = max(0.0, min(final_conf, 1.0))

        print(f"⏱ Time taken: {time.time() - st:.2f}s")

        rows.append({
            "transaction_text": t,
            "predicted_category": final_category,
            # keep matched_* fields from lookup even if fallback was used
            "matched_category_key": matched_cat,
            "matched_span_text": matched_span,
            "confidence": round(final_conf, 4),
            "review_required": (final_conf < 0.6)
        })

    df = pd.DataFrame(rows)

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🏁 Prediction complete in {time.time()-t0:.2f} seconds for {len(texts)} transactions")
    print("📄 Returning dataframe...")

    return df


if __name__ == "__main__":
    sample_inputs = [
        "Payment bill upi to mohak",
        "UPI payment to nykkaa@oksbi",
        "Electricity Bill Payment BESCOM",
        "Booking IRCTC Train Ticket",
        "Online purchase Amzon order #445"
    ]
    df = predict_transaction(sample_inputs)
    print(df.to_string(index=False))
