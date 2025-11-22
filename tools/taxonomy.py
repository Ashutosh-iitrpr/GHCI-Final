# tools/taxonomy.py
import os
import yaml
import json
import shutil
import tempfile
import datetime
from jsonschema import validate, ValidationError
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TAX_PATH = "config/taxonomy.yaml"
BACKUP_DIR = "config/backups"
AUDIT_LOG = "data/taxonomy_changes.log"

SCHEMA = {
    "type": "object",
    "required": ["categories"],
    "properties": {
        "version": {"type": "integer"},
        "updated_by": {"type": "string"},
        "updated_at": {"type": "string"},
        "categories": {
            "type": "object",
            "patternProperties": {
                "^[a-z0-9_\\- ]+$": {
                    "type": "object",
                    "required": ["display_name"],
                    "properties": {
                        "display_name": {"type": "string"},
                        "synonyms": {"type": "array", "items": {"type": "string"}},
                        "description": {"type": "string"}
                    },
                    "additionalProperties": False
                }
            }
        }
    }
}


def load_taxonomy(path=TAX_PATH):
    """ Load and automatically clean synonyms into plain string format """
    if not os.path.exists(path):
        return {"version": 1, "updated_by": "none", "updated_at": datetime.datetime.utcnow().isoformat(), "categories": {}}

    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {"categories": {}}

    # Normalize synonyms to list of strings
    cats = obj.get("categories", {})
    for k, v in cats.items():
        syns = v.get("synonyms", [])
        cleaned = []
        for s in syns:
            if isinstance(s, dict) and "value" in s:
                cleaned.append(str(s["value"]).lower().strip())
            elif isinstance(s, str):
                cleaned.append(s.lower().strip())
        cats[k]["synonyms"] = sorted(list(set(cleaned)))

    obj["categories"] = cats
    return obj


def validate_taxonomy(obj):
    validate(instance=obj, schema=SCHEMA)


def backup_taxonomy(path=TAX_PATH, backup_dir=BACKUP_DIR):
    os.makedirs(backup_dir, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dst = os.path.join(backup_dir, f"taxonomy.{ts}.yaml")
    shutil.copy2(path, dst)
    return dst


def atomic_write_yaml(obj, path=TAX_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="taxonomy_", dir=os.path.dirname(path))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=True)
    os.replace(tmp, path)


def append_audit(user, action, summary, before_path=None, after_obj=None, audit_log=AUDIT_LOG):
    os.makedirs(os.path.dirname(audit_log), exist_ok=True)
    ts = datetime.datetime.utcnow().isoformat()
    entry = {"timestamp": ts, "user": user, "action": action, "summary": summary}
    if before_path:
        entry["before_backup"] = before_path
    if after_obj:
        entry["after_snapshot"] = {"version": after_obj.get("version"), "categories": list(after_obj.get("categories", {}).keys())}
    with open(audit_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def update_taxonomy(new_obj, user="admin", trigger_retrain=False, retrain_cmd=None):
    before_backup = None
    if os.path.exists(TAX_PATH):
        before_backup = backup_taxonomy()

    # increment version
    new_obj = dict(new_obj)
    prev = load_taxonomy()
    new_obj.setdefault("version", prev.get("version", 1) + 1)
    new_obj["updated_by"] = user
    new_obj["updated_at"] = datetime.datetime.utcnow().isoformat()

    # validate uses cleaned synonyms now
    validate_taxonomy(new_obj)

    atomic_write_yaml(new_obj)
    append_audit(user, "update", f"Updated taxonomy ({len(new_obj.get('categories', {}))} categories)", before_path=before_backup, after_obj=new_obj)

    if trigger_retrain and retrain_cmd:
        import subprocess
        subprocess.Popen(retrain_cmd, shell=True)

    return True


# -------------------------
# Pending promotion
# -------------------------
def promote_pending_category(category_name):
    pending = "data/pending_new_categories.csv"
    feedback = "data/user_feedback.csv"
    if not os.path.exists(pending):
        return 0

    prev = pd.read_csv(pending).dropna(how="all", axis=1)
    prev["true_category"] = prev["true_category"].astype(str).str.strip().str.lower()
    move = prev[prev["true_category"] == category_name].copy()
    if move.empty:
        return 0

    move["source"] = "admin"

    if os.path.exists(feedback):
        fb = pd.read_csv(feedback).dropna(how="all", axis=1)
        new_fb = pd.concat([fb, move], ignore_index=True).drop_duplicates(subset=["transaction_text","true_category"], keep="last")
    else:
        new_fb = move

    new_fb.to_csv(feedback, index=False)
    prev[prev["true_category"] != category_name].to_csv(pending, index=False)
    append_audit("admin", "promote", f"Promoted {len(move)} pending rows", after_obj={"categories": {category_name: {}}})
    return len(move)


# -------------------------
# Admin Operations
# -------------------------
def add_category(category_key, display_name=None, synonyms=None, description="", user="admin", trigger_retrain=False, retrain_cmd=None):
    category_key = category_key.strip().lower()
    tax = load_taxonomy()
    cats = tax.get("categories", {})

    if category_key in cats:
        raise ValueError(f"Category '{category_key}' already exists")

    cats[category_key] = {
        "display_name": display_name or category_key.title(),
        "synonyms": [s.lower().strip() for s in (synonyms or [])],
        "description": description
    }

    tax["categories"] = cats
    update_taxonomy(tax, user, trigger_retrain, retrain_cmd)
    return True


def add_merchant(category_key, merchant_name, user="admin", trigger_retrain=False, retrain_cmd=None):
    category_key = category_key.lower().strip()
    merchant_name = merchant_name.lower().strip()
    tax = load_taxonomy()

    if category_key not in tax["categories"]:
        raise ValueError(f"Category '{category_key}' does not exist")

    syns = tax["categories"][category_key].get("synonyms", [])
    if merchant_name in syns:
        return False

    syns.append(merchant_name)
    tax["categories"][category_key]["synonyms"] = sorted(list(set(syns)))
    update_taxonomy(tax, user, trigger_retrain, retrain_cmd)

    append_audit(user, "add_merchant", f"Added '{merchant_name}' → '{category_key}'")
    return True
