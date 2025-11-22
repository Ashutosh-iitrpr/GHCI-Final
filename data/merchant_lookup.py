"""
merchant_lookup.py — improved noisy-match handling (NER + edit/fuzzy/phonetic boosts)

Key changes vs previous:
 - Try spaCy NER (ORG/PRODUCT/PERSON) as candidate source (optional)
 - Add short-token focused fuzzy/edit-distance sweep across alias list
 - Generate simple edit variants for tokens (deletes, transposes, common vowel fixes)
 - Increase weight of lexical fuzzy + edit similarity for short tokens
 - Provide debug prints for top matches per candidate
"""
import re
import os
import numpy as np
from typing import List, Tuple, Optional

# optional libs
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except Exception:
    _HAS_SBERT = False

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

try:
    from rapidfuzz import fuzz, process as rapid_process
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

# spaCy optional for NER
try:
    import spacy
    _HAS_SPACY = True
    # we try to lazily load the small model only when needed
except Exception:
    _HAS_SPACY = False

# sklearn fallback embedding (char ngram)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# difflib fallback for fuzzy matching
import difflib

UPI_HANDLE_RE = re.compile(r"\b[\w.\-]{2,}@\w+\b", flags=re.IGNORECASE)
PUNCT_RE = re.compile(r"[^\w\s]")
EXCLUDE_TOKENS = set(["upi","payment","paid","to","from","txn","ref","order","bill","billpayment","payment", "xfer", "online"])

# helper: simple phonetic -- fallback to lowercased tokens; you can integrate DoubleMetaphone later
def simple_phonetic(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())

def _normalize_alias_item(item):
    if not item:
        return None
    if isinstance(item, str):
        val = item.lower().strip()
        val = PUNCT_RE.sub(" ", val)
        val = " ".join(tok for tok in val.split() if tok and tok not in EXCLUDE_TOKENS)
        return {"value": val, "source": "seed", "weight": 0.8}
    if isinstance(item, dict):
        val = (item.get("value") or item.get("alias") or "").lower().strip()
        val = PUNCT_RE.sub(" ", val)
        val = " ".join(tok for tok in val.split() if tok and tok not in EXCLUDE_TOKENS)
        return {"value": val, "source": item.get("source", "seed"), "weight": float(item.get("weight", 0.8))}
    return None

class MerchantLookup:
    def __init__(self, taxonomy: dict, embed_model_name: str = "all-MiniLM-L6-v2"):
        self.taxonomy = taxonomy
        cats = taxonomy.get("categories", {})

        # Build alias list (category_key, alias_value, weight, source)
        aliases = []
        for cat_key, meta in cats.items():
            syns = meta.get("synonyms", []) or []
            for s in syns:
                ai = _normalize_alias_item(s)
                if ai and ai.get("value"):
                    aliases.append((cat_key, ai["value"], ai.get("weight", 0.8), ai.get("source", "seed")))
            # include display name
            name = meta.get("display_name", cat_key)
            ai = _normalize_alias_item(name)
            if ai and ai.get("value"):
                aliases.append((cat_key, ai["value"], ai.get("weight", 0.8), ai.get("source", "seed")))

        # deduplicate preserving first weight
        seen = {}
        new_aliases = []
        for (cat, val, w, src) in aliases:
            if val in seen:
                continue
            seen[val] = True
            new_aliases.append((cat, val, w, src))
        self.aliases = new_aliases
        self.alias_texts = [a for (_, a, _, _) in self.aliases]
        self.alias_meta = [(cat, w, src) for (cat, _, w, src) in self.aliases]

        # optional SBERT / TF-IDF embedding engine
        self._use_sbert = False
        self.sbert = None
        if _HAS_SBERT:
            try:
                self.sbert = SentenceTransformer(embed_model_name)
                self._use_sbert = True
            except Exception:
                self.sbert = None
                self._use_sbert = False

        self.tfidf = None
        if not self._use_sbert and _HAS_SKLEARN and self.alias_texts:
            self.tfidf = TfidfVectorizer(ngram_range=(1,2), analyzer='char_wb', min_df=1)
            self.tfidf.fit(self.alias_texts)

        # alias embeddings (SBERT or TFIDF)
        self.alias_embeddings = None
        if self._use_sbert and self.alias_texts:
            self.alias_embeddings = self.sbert.encode(self.alias_texts, convert_to_numpy=True, normalize_embeddings=True)
        elif self.tfidf is not None and self.alias_texts:
            self.alias_embeddings = self.tfidf.transform(self.alias_texts).toarray()

        # faiss index optional
        self.faiss_index = None
        if _HAS_FAISS and self.alias_embeddings is not None and self.alias_embeddings.size > 0:
            dim = self.alias_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.alias_embeddings.astype('float32'))

        # lazy spaCy model holder (don't load unless used)
        self._spacy_nlp = None

    # --- normalization & candidate creation ---
    def _normalize_text(self, s: str) -> str:
        if not s:
            return ""
        s = s.lower().strip()
        # extract upi merchant prefix (nyka@oksbi -> nyka)
        upi_handles = UPI_HANDLE_RE.findall(s)
        for h in upi_handles:
            prefix = h.split("@")[0]
            s = s.replace(h, " " + prefix + " ")

        s = PUNCT_RE.sub(" ", s)
        s = " ".join(tok for tok in s.split() if tok and tok not in EXCLUDE_TOKENS)
        return s

    def _ensure_spacy(self):
        if not _HAS_SPACY:
            return None
        if self._spacy_nlp is None:
            # lazy load small model if available
            try:
                self._spacy_nlp = spacy.load("en_core_web_sm")
            except Exception:
                try:
                    # fallback to blank English if no model; less accurate
                    self._spacy_nlp = spacy.blank("en")
                except Exception:
                    self._spacy_nlp = None
        return self._spacy_nlp

    def generate_candidates(self, text: str) -> List[Tuple[int,int,str]]:
        raw = text or ""
        candidates = set()

        # 1) explicit UPI handles
        for m in UPI_HANDLE_RE.finditer(raw):
            span = m.group(0)
            candidates.add((m.start(), m.end(), self._normalize_text(span)))

        # 2) spaCy NER (ORG/PRODUCT) if available
        nlp = self._ensure_spacy()
        if nlp is not None:
            try:
                doc = nlp(raw)
                for ent in doc.ents:
                    # include ORG, PRODUCT, PERSON, GPE etc as possible merchant spans
                    if ent.label_ in ("ORG", "PRODUCT", "PERSON", "GPE", "NORP"):
                        norm = self._normalize_text(ent.text)
                        if norm:
                            candidates.add((ent.start_char, ent.end_char, norm))
            except Exception:
                pass

        # 3) sliding-window tokens (n up to 5) focusing on tokens that look like names
        toks = re.split(r"\s+", raw)
        positions = []
        idx = 0
        for t in toks:
            start = raw.find(t, idx)
            end = start + len(t)
            positions.append((t, start, end))
            idx = end

        n = len(positions)
        for i in range(n):
            for L in range(1, min(6, n-i+1)):
                span_toks = positions[i:i+L]
                span_text = " ".join(p[0] for p in span_toks)
                if len(span_text) < 2:
                    continue
                # skip pure digits
                if span_text.strip().isdigit():
                    continue
                norm = self._normalize_text(span_text)
                if not norm:
                    continue
                start = span_toks[0][1]
                end = span_toks[-1][2]
                # heuristics: prefer spans near "to", "at", "paid", or capitalized spans
                candidates.add((start, end, norm))

        # 4) capitalized spans regex
        cap_spans = re.findall(r"([A-Z][A-Za-z&\.\-]{1,}(?:\s+[A-Z][A-Za-z&\.\-]{1,}){0,4})", raw)
        for cs in cap_spans:
            norm = self._normalize_text(cs)
            if norm:
                start = raw.find(cs)
                end = start + len(cs)
                candidates.add((start, end, norm))

        # convert to sorted list
        cand_list = sorted(list(candidates), key=lambda x: (x[0], -(x[1]-x[0])))
        # Heuristic: return top ~20 candidates to limit work
        return cand_list[:20]

    # --- variant generation for short/noisy tokens ---
    def generate_variants(self, token: str, max_variants: int = 40):
        t = token.strip()
        if not t:
            return []
        variants = set()
        variants.add(t)
        # deletes
        for i in range(len(t)):
            variants.add(t[:i] + t[i+1:])
        # transposes
        for i in range(len(t)-1):
            variants.add(t[:i] + t[i+1] + t[i] + t[i+2:])
        # replace vowels with common ones (help with vowel-drop)
        vowels = "aeiou"
        for v in vowels:
            variants.add(re.sub(r'[aeiou]', v, t))
        # replace o->0, l->1 for char-swap noise
        variants.add(t.replace("o", "0"))
        variants.add(t.replace("l", "1"))
        # add UPI-like suffixes
        variants.add(t + "@ok")
        variants.add(t + "@oksbi")
        # remove spaces and punctuation
        variants.add(t.replace(" ", ""))
        # duplicate last character form e.g. nyka -> nykaa
        variants.add(t + t[-1])
        ret = list(variants)
        return ret[:max_variants]

    # --- embedding / faiss / tfidf wrapper ---
    def _embed_text(self, texts: List[str]):
        if not texts:
            return None
        if self._use_sbert and self.sbert is not None:
            return self.sbert.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        elif self.tfidf is not None:
            return self.tfidf.transform(texts).toarray()
        else:
            # char-hash fallback
            out = []
            for t in texts:
                vec = np.zeros(256, dtype=float)
                for i,ch in enumerate(t):
                    vec[i % 256] += ord(ch)
                out.append(vec)
            return np.vstack(out)

    # --- fuzzy sweep across aliases (fast) ---
    def _best_fuzzy_match(self, candidate: str, top_k:int=3):
        cand = candidate
        # rapidfuzz preferred
        if _HAS_RAPIDFUZZ:
            # get top K fuzzy matches with token_sort_ratio
            res = rapid_process.extract(cand, self.alias_texts, scorer=fuzz.token_sort_ratio, limit=top_k)
            # res entries: (alias_text, score, index)
            out = []
            for alias_text, score, idx in res:
                cat, alias_weight, alias_source = self.alias_meta[idx]
                out.append((cat, score/100.0, alias_text, alias_source, alias_weight))
            return out
        else:
            # difflib fallback: use get_close_matches (no scores)
            choices = difflib.get_close_matches(cand, self.alias_texts, n=top_k, cutoff=0.4)
            out = []
            for ch in choices:
                try:
                    idx = self.alias_texts.index(ch)
                    cat, alias_weight, alias_source = self.alias_meta[idx]
                    # approximate ratio using SequenceMatcher
                    score = difflib.SequenceMatcher(None, cand, ch).ratio()
                    out.append((cat, float(score), ch, alias_source, alias_weight))
                except ValueError:
                    continue
            return out

    def match_candidate(self, candidate_text: str, top_k: int = 3):
        if not candidate_text:
            return []
        candidate_text = candidate_text.strip()
        cand_emb = self._embed_text([candidate_text])
        n_alias = len(self.alias_texts) if self.alias_texts else 0

        results = []

        # 1) embedding-based nearest neighbors (if available)
        if self.faiss_index is not None and cand_emb is not None and cand_emb.shape[1] == self.alias_embeddings.shape[1]:
            q = cand_emb.astype('float32')
            D, I = self.faiss_index.search(q, min(top_k, n_alias))
            for sim, idx in zip(D[0].tolist(), I[0].tolist()):
                if idx < 0: continue
                cat, alias_text, alias_weight, alias_source = self.aliases[idx]
                # fuzzy boost (if rapidfuzz)
                fuzzy = 0.0
                if _HAS_RAPIDFUZZ:
                    fuzzy = fuzz.token_sort_ratio(candidate_text, alias_text) / 100.0
                final = 0.55 * float(sim) + 0.30 * float(fuzzy) + 0.15 * float(alias_weight)
                results.append((cat, final, alias_text, alias_source, alias_weight))

        elif self.alias_embeddings is not None and cand_emb is not None:
            from sklearn.preprocessing import normalize
            try:
                emb_norm = normalize(self.alias_embeddings, axis=1)
                q_norm = normalize(cand_emb, axis=1)
                sims = (q_norm @ emb_norm.T)[0]
                idxs = np.argsort(-sims)[:top_k]
                for idx in idxs:
                    sim = float(sims[idx])
                    cat, alias_text, alias_weight, alias_source = self.aliases[idx]
                    fuzzy = 0.0
                    if _HAS_RAPIDFUZZ:
                        fuzzy = fuzz.token_sort_ratio(candidate_text, alias_text) / 100.0
                    final = 0.55 * sim + 0.30 * fuzzy + 0.15 * float(alias_weight)
                    results.append((cat, final, alias_text, alias_source, alias_weight))
            except Exception:
                pass

        # 2) fuzzy lexical sweep across alias_texts (covers short/noisy tokens well)
        fuzzy_hits = self._best_fuzzy_match(candidate_text, top_k=5)
        for (cat, fuzzy_score, alias_text, alias_source, alias_weight) in fuzzy_hits:
            # estimate embedding sim fallback as 0 if unknown
            emb_sim = 0.0
            # combine: emphasize fuzzy for short tokens
            length_bonus = 0.0
            if len(candidate_text) <= 6:
                # short tokens: give fuzzy more weight
                final = 0.20 * emb_sim + 0.65 * fuzzy_score + 0.15 * float(alias_weight)
            else:
                final = 0.50 * emb_sim + 0.40 * fuzzy_score + 0.10 * float(alias_weight)
            final = min(final, 1.0) 
            results.append((cat, final, alias_text, alias_source, alias_weight))

        # 3) generate small edit variants for very short noisy tokens and check fuzzy
        if len(candidate_text) <= 6:
            variants = self.generate_variants(candidate_text, max_variants=30)
            for v in variants:
                v_hits = self._best_fuzzy_match(v, top_k=2)
                for (cat, fuzzy_score, alias_text, alias_source, alias_weight) in v_hits:
                    final = 0.65 * fuzzy_score + 0.25 * (1.0 if v==candidate_text else 0.0) + 0.10 * float(alias_weight)
                    final = min(final, 1.0)
                    results.append((cat, final, alias_text, alias_source, alias_weight))

        # deduplicate & sort
        if not results:
            return []

        # collapse by alias_text keeping max score
        best_by_alias = {}
        for cat, score, alias_text, alias_source, alias_weight in results:
            key = (alias_text, cat)
            if key not in best_by_alias or score > best_by_alias[key][0]:
                best_by_alias[key] = (score, alias_source, alias_weight)

        collapsed = []
        for (alias_text, cat), (score, alias_source, alias_weight) in best_by_alias.items():
            collapsed.append((cat, float(score), alias_text, alias_source, alias_weight))

        collapsed.sort(key=lambda x: x[1], reverse=True)
        return collapsed[:top_k]

    def best_match_for_text(self, text: str, top_k: int = 3) -> Optional[Tuple[str, float, str]]:
        candidates = self.generate_candidates(text)
        if not candidates:
            return None
        best = None
        for (s,e,c) in candidates:
            matches = self.match_candidate(c, top_k=top_k)
            if not matches:
                continue
            cat, score, alias_text, alias_source, alias_weight = matches[0]

            common = len(set(c.lower()) & set(alias_text.lower()))
            overlap = common / max(len(c), len(alias_text))
            if overlap < 0.50:                # reject completely different words
                continue

            print(f"  [CAND] span='{c}' -> "
                    f"alias='{alias_text}' (cat={cat}) raw={score:.3f} "
                    f"overlap={overlap:.2f}")

            penalty = 0.65 if score < 0.85 else 1.0
            final_score = round(score * alias_weight * penalty, 3)

            pos_bonus = 0.0
            prefix = text[max(0,s-12):s].lower()
            if any(k in prefix for k in (" to ", " at ", " paid to", "pmt to", "paid ")):
                pos_bonus = 0.05
            source_bonus = 0.05 if alias_source in ("admin", "seed") else 0.0
            final_score = min(score + pos_bonus + source_bonus, 1.0)

            # context-aware penalty for short or suspicious tokens
            is_short = len(c) <= 6
            fuzzy_strong = score >= 0.80     # strong fuzzy
            is_known_alias = alias_source in ("admin","seed")

            if is_short and not fuzzy_strong and not is_known_alias:
                # penalize only weak short matches
                final_score -= 0.40  # suppress gibberish

            # discard weak candidates entirely
            if final_score < 0.55:
                continue

            final_score = max(min(final_score, 1.0), 0.0)

            # debug print for visibility
            print(f"  [CAND] span='{c}' -> top_alias='{alias_text}' (cat={cat}) score={score:.3f} src={alias_source} w={alias_weight} final={final_score:.3f}")

            if best is None or final_score > best[1]:
                best = (cat, final_score, c, alias_text, alias_source, alias_weight)

            if best is None or best[1] < 0.75:
                return None

        return best
