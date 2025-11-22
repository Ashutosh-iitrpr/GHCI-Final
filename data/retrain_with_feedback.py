# retrain_with_feedback.py
"""
Retrain pipeline that:
 - reads config/taxonomy.yaml
 - accepts feedback rows whose true_category exists in taxonomy
 - adds new valid merchant aliases from accepted feedback into taxonomy
 - respects different weights for user vs admin feedback (different augmentation)
 - writes a compact merchant_index.json for debugging / UI
 - optionally triggers model retrain (sample weights)
"""

import os
import joblib
import pandas as pd
import yaml
import json
import datetime
from collections import defaultdict
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(__file__)
FEEDBACK = os.path.join(BASE_DIR, "user_feedback.csv")
TAX_PATH = os.path.join(BASE_DIR, "../config/taxonomy.yaml")
MODEL_PATH = os.path.join(BASE_DIR, "../models/baseline_model.pkl")
VECT_PATH = os.path.join(BASE_DIR, "../models/tfidf_vectorizer.pkl")
MERCHANT_INDEX_OUT = os.path.join(BASE_DIR, "merchant_index.json")

# Hyperparameters - tune to taste
ADMIN_ALIAS_WEIGHT = 1.0
USER_ALIAS_WEIGHT = 0.35
SEED_ALIAS_WEIGHT = 0.8
SYNTH_ADMIN_MULTIPLIER = 6    # generate more synthetic variants for admin aliases
SYNTH_USER_MULTIPLIER  = 2    # fewer for user-created aliases
SYNTH_MAX_VARIANTS = 30

def load_taxonomy(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_taxonomy(tax: Dict[str, Any]) -> None:
    with open(TAX_PATH, 'w', encoding='utf-8') as f:
        yaml.safe_dump(tax, f, sort_keys=False, allow_unicode=True)
    print("🔄 Saved taxonomy.yaml")

def save_merchant_index(taxonomy: Dict[str, Any], out_path: str) -> None:
    cats = taxonomy.get("categories", {})
    compact = {}
    for cat_key, meta in cats.items():
        syns = meta.get("synonyms", [])
        # normalize to list of dicts for UI
        normalized = []
        for s in syns:
            if isinstance(s, str):
                normalized.append({"value": s, "source": "seed", "weight": SEED_ALIAS_WEIGHT})
            elif isinstance(s, dict):
                normalized.append(s)
        compact[cat_key] = {
            "display_name": meta.get("display_name", cat_key),
            "aliases": normalized
        }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(compact, f, indent=2, ensure_ascii=False)
    print(f"📦 Wrote merchant index -> {out_path}")

def _normalize_alias_obj(alias_obj) -> Dict[str, Any]:
    """
    Ensure alias representation is dict with required fields.
    Accepts string or dict.
    """
    now = datetime.datetime.utcnow().isoformat()
    if isinstance(alias_obj, str):
        return {"value": alias_obj, "source": "seed", "weight": SEED_ALIAS_WEIGHT, "created_at": now}
    if isinstance(alias_obj, dict):
        out = {}
        out['value'] = alias_obj.get('value') or alias_obj.get('alias') or ""
        out['source'] = alias_obj.get('source', 'seed')
        out['weight'] = float(alias_obj.get('weight', SEED_ALIAS_WEIGHT))
        out['created_at'] = alias_obj.get('created_at', now)
        out['meta'] = alias_obj.get('meta', {})
        return out
    # fallback
    return {"value": str(alias_obj), "source": "seed", "weight": SEED_ALIAS_WEIGHT, "created_at": now}

def generate_synthetic_variants(alias_text: str, multiplier: int = 2, max_variants: int = 10) -> List[str]:
    """
    Make simple synthetic variants to help TF-IDF classifier and to increase
    likelihood of matching common noisy forms.
    Strategies: small typos, shorten/abbrev, drop vowels, remove punctuation, add upi suffix variants.
    Keep deterministic-ish but reasonably varied.
    """
    out = set()
    base = alias_text.strip()
    if not base:
        return []

    out.add(base)
    # abbreviation by taking first letters of words
    words = base.split()
    if len(words) > 1:
        out.add("".join(w[0] for w in words if w))
    # remove vowels (aggressive)
    out.add(re.sub(r"[aeiou]", "", base))
    # remove spaces
    out.add(base.replace(" ", ""))
    # small character deletions
    if len(base) > 3:
        out.add(base[:-1])
        out.add(base[1:])
    # upi-style variants
    out.add(base + "@ok")
    out.add(base + "@oksbi")
    out.update({base + str(i) for i in range(1, 3)})

    # if multiplier >1 attempt to create more subtle variants
    # small replacements of 'o'->'0', 'l'->'1'
    if multiplier >= 2:
        out.add(base.replace("o", "0"))
        out.add(base.replace("l", "1"))
    if multiplier >= 4:
        out.add(base.replace("a", "@"))
        out.add(base.replace("s", "5"))

    variants = list(out)
    # limit and return
    return variants[:min(max_variants, len(variants))]

import re

def main():
    tax = load_taxonomy(TAX_PATH)
    cats = tax.get("categories", {})

    # migrate synonyms (strings -> dicts) for safety & richer metadata
    migrated = False
    for cat_key, meta in cats.items():
        syns = meta.get("synonyms", []) or []
        new_syns = []
        for s in syns:
            ali = _normalize_alias_obj(s)
            new_syns.append(ali)
        if new_syns != syns:
            cats[cat_key]['synonyms'] = new_syns
            migrated = True
    if migrated:
        tax['categories'] = cats
        save_taxonomy(tax)
        print("🔁 Taxonomy migrated: string synonyms converted to alias objects.")

    # write merchant index for UI even if no feedback
    save_merchant_index(tax, MERCHANT_INDEX_OUT)

    if not os.path.exists(FEEDBACK):
        print("ℹ No feedback file found.")
        return

    df = pd.read_csv(FEEDBACK)
    if df.empty:
        print("ℹ Feedback file empty.")
        return

    # expected columns: transaction_text, predicted_category, matched_category_key,
    # matched_span_text, true_category, accepted, feedback_source (optional: 'user' or 'admin')
    # default source is 'user' unless accepted_by_admin flagged via feedback_source or accepted_by_admin column.
    accepted_rows = df[df.get("accepted", False) == True].copy()
    if accepted_rows.empty:
        print("ℹ No accepted feedback to process.")
        return

    updates = 0
    synthetic_examples = []  # (text, label, sample_weight)
    for _, row in accepted_rows.iterrows():
        true_cat = row.get("true_category")
        span = row.get("matched_span_text") or row.get("merchant_span") or row.get("merchant_span_text")
        # allow feedback to pass a column 'feedback_source' which can be 'admin' or 'user'
        fb_source = str(row.get("feedback_source") or row.get("source") or "").strip().lower()
        if fb_source not in ("admin", "user"):
            # fallback to 'user' unless some accepted_by_admin flag present
            fb_source = "user"

        if pd.isna(span) or not isinstance(span, str) or not span.strip():
            continue

        # normalize text as alias value (it is safe to call MerchantLookup._normalize_text but avoid circular import)
        norm = re.sub(r"[^\w\s]", " ", span.lower()).strip()
        if not norm:
            continue

        if true_cat not in cats:
            print(f"⚠ True category '{true_cat}' not present - skipped alias '{norm}'")
            continue

        # choose weights and multipliers
        if fb_source == "admin":
            weight = ADMIN_ALIAS_WEIGHT
            multiplier = SYNTH_ADMIN_MULTIPLIER
            source_label = "admin"
        else:
            weight = USER_ALIAS_WEIGHT
            multiplier = SYNTH_USER_MULTIPLIER
            source_label = "user"

        # add alias entry if missing
        existing = cats[true_cat].get("synonyms", []) or []
        found = False
        for a in existing:
            # a may be dict or str (but migrated earlier to dicts)
            if isinstance(a, dict):
                if a.get("value") == norm:
                    found = True
                    # if existing alias has lower weight and this is admin, bump it
                    if source_label == "admin" and float(a.get("weight", 0.0)) < weight:
                        a['weight'] = weight
                        a['source'] = 'admin'
                        a['created_at'] = datetime.datetime.utcnow().isoformat()
                        updates += 1
                        print(f"↑ Bumped weight for alias '{norm}' in '{true_cat}' by admin")
                    break
            else:
                # string case (should be migrated) - compare normalized
                if str(a).lower().strip() == norm:
                    found = True
                    break

        if not found:
            alias_obj = {
                "value": norm,
                "source": source_label,
                "weight": float(weight),
                "created_at": datetime.datetime.utcnow().isoformat()
            }
            existing.append(alias_obj)
            cats[true_cat]['synonyms'] = existing
            updates += 1
            print(f"➕ Added alias '{norm}' to '{true_cat}' (source={source_label}, weight={weight})")

        # generate synthetic variants for model augmentation (if you retrain)
        variants = generate_synthetic_variants(norm, multiplier, max_variants=SYNTH_MAX_VARIANTS)
        # assign sample weight proportional to alias weight
        for v in variants:
            sample_weight = float(weight)
            synthetic_examples.append((v, true_cat, sample_weight))

    if updates > 0:
        tax['categories'] = cats
        save_taxonomy(tax)
        print(f"✅ taxonomy updated with {updates} alias changes.")
    else:
        print("ℹ No alias updates necessary.")

    save_merchant_index(tax, MERCHANT_INDEX_OUT)

    # Optional retrain using synthetic_examples:
    # If you wish to retrain your TF-IDF + classifier here, you can use the synthetic_examples
    # to augment your training data and provide sample_weight accordingly.
    if synthetic_examples:
        print(f"🧪 Prepared {len(synthetic_examples)} synthetic examples for retraining (sample-weighted).")
        # Example: call your training function here, or call data/train_baseline.py with augmented CSV.
        # E.g. write synthetic examples to a CSV for manual inspection/training:
        aug_df = pd.DataFrame(synthetic_examples, columns=["transaction_text", "true_category", "sample_weight"])
        aug_out = os.path.join(BASE_DIR, "synthetic_augmented_examples.csv")
        aug_df.to_csv(aug_out, index=False)
        print(f"📝 Wrote augmented examples -> {aug_out}")
        # If you have a training routine that accepts sample weights, you may call it here.
    else:
        print("ℹ No synthetic examples generated (no accepted feedback produced variants).")

    print("🔚 retrain_with_feedback completed.")

if __name__ == "__main__":
    main()
