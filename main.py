"""
main.py — Core recommendation logic for SHL Assessment Engine
Improvements:
- Preload LLM clients (Gemini & Together)
- LRU cache for constraint extraction
- Robust logging on exceptions
- Startup assertions for data/index consistency
- Docstrings and type hints
- Auto-fallback from Gemini to Together on ResourceExhausted
"""
import os
import re
import json
import logging
from functools import lru_cache
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv

# ——— Configuration & Logging ———
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ——— Paths & Constants ———
DATA_DIR       = os.getenv("DATA_DIR", "./data")
CATALOG_CSV    = os.path.join(DATA_DIR, "cleaned_catalog.csv")
EMB_FILE       = os.path.join(DATA_DIR, "embeddings.npy")
INDEX_FILE     = os.path.join(DATA_DIR, "faiss_index.bin")
SKILL_MAP_FILE = os.path.join(DATA_DIR, "skill_map.json")
TOGETHER_URL   = "https://api.together.xyz/v1/completions"

# ——— Load environment & API keys ———
load_dotenv()

def get_keys_and_configure() -> Tuple[str, str]:
    """
    Load Gemini and Together API keys from environment and configure Gemini client.
    Raises EnvironmentError if missing.
    """
    gemini_key   = os.getenv("GEMINI_API_KEY")
    together_key = os.getenv("TOGETHER_API_KEY")
    if not gemini_key or not together_key:
        raise EnvironmentError("Set GEMINI_API_KEY and TOGETHER_API_KEY in your environment")
    genai.configure(api_key=gemini_key)
    return gemini_key, together_key

GEMINI_KEY, TOGETHER_KEY = get_keys_and_configure()
# Pre-instantiated LLM clients
GEMINI_MODEL = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# ——— Load Data & Models ———
df         = pd.read_csv(CATALOG_CSV)
embeddings = np.load(EMB_FILE)
index      = faiss.read_index(INDEX_FILE)
embedder   = SentenceTransformer("all-MiniLM-L6-v2")

# Startup sanity checks
assert df.shape[0] == embeddings.shape[0], "Catalog rows != embeddings rows"
assert index.ntotal == embeddings.shape[0], "FAISS index size mismatch"

# ——— Load Skill Map ———
with open(SKILL_MAP_FILE) as f:
    SKILL_MAP = {k.lower(): v for k, v in json.load(f).items()}

# ——— Helper Functions ———

def parse_duration(text: str) -> Optional[int]:
    """
    Extract hours/minutes from free-form text.
    Returns total minutes or None if nothing found.
    """
    t = (text or "").lower()
    hrs  = sum(int(h) for h in re.findall(r"(\d+)\s*(?:h|hour)s?", t))
    mins = sum(int(m) for m in re.findall(r"(\d+)\s*(?:m|min)utes?", t))
    total = hrs * 60 + mins
    return total if total > 0 else None


def safe_json_extract(txt: str) -> Dict[str, Any]:
    """
    Find the first JSON object in txt and parse it.
    Returns empty dict on failure.
    """
    try:
        body = txt[txt.find("{"): txt.rfind("}")+1]
        return json.loads(body)
    except Exception:
        logger.exception("Failed to extract JSON from model response")
        return {}


def default_constraints(dur: Optional[int]) -> Dict[str, Any]:
    return {
        "skills": [],
        "max_duration": dur,
        "remote_required": "no",
        "adaptive_required": "no",
        "test_type": None
    }


def merge_constraints(parsed: Dict[str, Any], fallback: Optional[int]) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        return default_constraints(fallback)
    return {
        "skills":            parsed.get("skills", []),
        "max_duration":      parsed.get("max_duration") or fallback,
        "remote_required":   parsed.get("remote_required", "no"),
        "adaptive_required": parsed.get("adaptive_required", "no"),
        "test_type":         parsed.get("test_type")
    }

# ——— Constraint Extraction ———
PROMPT_TMPL = (
    "You are an intelligent assistant helping recruiters choose the right assessments.\n"
    "Given a natural language query about a hiring requirement, extract the following details as a valid JSON object:\n"
    "- \"skills\": A list of technical or soft skills mentioned.\n"
    "- \"max_duration\": Maximum assessment duration in minutes (integer).\n"
    "- \"remote_required\": \"yes\" or \"no\".\n"
    "- \"adaptive_required\": \"yes\" or \"no\".\n"
    "- \"test_type\": e.g., \"technical\", \"cognitive\", etc.\n\n"
    "Respond ONLY with a JSON object. No extra text.\n\n"
    "Query:\n{query}"
)

@lru_cache(maxsize=256)
def extract_constraints(query: str, engine: str = "gemini") -> Dict[str, Any]:
    """
    Parse hiring query into structured constraints via LLM.
    Auto-fallback to Together.ai on Gemini quota exhaustion.
    """
    pre_max = parse_duration(query)
    prompt  = PROMPT_TMPL.format(query=query)

    def call_together() -> str:
        headers = {"Authorization": f"Bearer {TOGETHER_KEY}"}
        payload = {
            "model":       "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "prompt":      prompt,
            "max_tokens":  200,
            "temperature": 0.3
        }
        res = requests.post(TOGETHER_URL, headers=headers, json=payload, timeout=60)
        res.raise_for_status()
        return res.json()["choices"][0]["text"]

    txt = ""
    if engine == "gemini":
        try:
            resp = GEMINI_MODEL.generate_content(prompt)
            txt  = getattr(resp, "text", resp.parts[0].text)
        except ResourceExhausted:
            logger.warning("Gemini quota exhausted; falling back to Together.ai")
            txt = call_together()
        except Exception:
            logger.exception("Gemini constraint extraction failed")
            return default_constraints(pre_max)

    elif engine == "together":
        try:
            txt = call_together()
        except Exception:
            logger.exception("Together.ai constraint extraction failed")
            return default_constraints(pre_max)
    else:
        raise ValueError("Unsupported engine. Use 'gemini' or 'together'.")

    parsed = safe_json_extract(txt)
    return merge_constraints(parsed, pre_max)

# ——— Skill-Map Lookup ———
def by_skill_map(query: str, skills: List[str]) -> pd.DataFrame:
    slugs: List[str] = []
    for s in skills:
        slugs.extend(SKILL_MAP.get(s.lower(), []))
    if not slugs:
        low = query.lower()
        for k, vals in SKILL_MAP.items():
            if k in low:
                slugs.extend(vals)
    slugs = list(dict.fromkeys(slugs))
    if not slugs:
        return pd.DataFrame()

    mask = df["relative_url"].str.contains("|".join(slugs), na=False)
    sub  = df.loc[mask].copy()
    sub.insert(0, "query", query)
    sub["score"] = 1.0
    return sub.reset_index(drop=True)[["query"] + list(df.columns) + ["score"]]

# ——— FAISS Retrieval & Ranking ———
def retrieve_assessments(query: str, cons: Dict[str, Any], top_k: int = 10) -> pd.DataFrame:
    """
    Filter catalog per constraints, then rank via FAISS embeddings.
    Applies up to 4 fallback strategies.
    """
    def filter_df(d: pd.DataFrame, c: Dict[str, Any]) -> pd.DataFrame:
        f = d.copy()
        if c["max_duration"]:
            f = f[f.duration <= c["max_duration"]]
        if c["remote_required"] == "yes":
            f = f[f.remote.fillna("no").str.lower() == "yes"]
        if c["adaptive_required"] == "yes":
            f = f[f.adaptive.fillna("no").str.lower() == "yes"]
        if c["test_type"]:
            types = [c["test_type"]] if isinstance(c["test_type"], str) else c["test_type"]
            f = f[f.test_type.fillna("").str.lower().apply(lambda x: any(t.lower() in x for t in types))]
        return f

    fallback_steps = [
        cons,
        {**cons, "adaptive_required": "no"},
        {**cons, "adaptive_required": "no", "test_type": None},
        {**cons, "adaptive_required": "no", "test_type": None, "remote_required": "no"}
    ]

    for fb in fallback_steps:
        cand = filter_df(df, fb)
        if not cand.empty:
            logger.info(f"Matched with fallback: {fb}")
            qv = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
            D, I = index.search(qv, top_k * 5)
            hits: List[Dict[str, Any]] = []
            for dist, idx in zip(D[0], I[0]):
                if idx in cand.index:
                    rec = cand.loc[idx].to_dict()
                    rec.update({"query": query, "score": dist})
                    hits.append(rec)
                if len(hits) >= top_k:
                    break
            if hits:
                out = pd.DataFrame(hits)
                out["score"] = MinMaxScaler().fit_transform(out[["score"]])
                return out.sort_values("score", ascending=False)[["query"] + list(df.columns) + ["score"]]
    return pd.DataFrame()

# ——— Main API ———
def recommend(query: str, engine: str = "gemini") -> pd.DataFrame:
    """
    Main entry: extract constraints, attempt skill-map lookup, else FAISS retrieval.

    Returns a DataFrame of matching assessments.
    """
    cons   = extract_constraints(query, engine)
    logger.info(f"Constraints: {cons}")
    df_map = by_skill_map(query, cons.get("skills", []))
    if not df_map.empty:
        logger.info("Skill-map matched; returning direct mappings")
        return df_map
    return retrieve_assessments(query, cons)

# ——— Test Run ———
if __name__ == "__main__":
    sample = """
    I need a 45-minute adaptive technical test that can be taken remotely.
    """
    res = recommend(sample, engine="gemini")
    if res.empty:
        logger.warning("No assessments matched your criteria.")
    else:
        print(res.to_markdown(index=False))