import os
import re
import json
import pandas as pd
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import google.generativeai as genai
from dotenv import load_dotenv

# ——— CONFIG & PATHS ———
DATA_DIR       = "./data"
CATALOG_CSV    = os.path.join(DATA_DIR, "cleaned_catalog.csv")
EMB_FILE       = os.path.join(DATA_DIR, "embeddings.npy")
INDEX_FILE     = os.path.join(DATA_DIR, "faiss_index.bin")
SKILL_MAP_FILE = os.path.join(DATA_DIR, "skill_map.json")

# ——— LOAD ENV & API KEYS ———
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
TOGETHER_KEY   = os.getenv("TOGETHER_API_KEY")
if not GEMINI_KEY or not TOGETHER_KEY:
    raise EnvironmentError("Set GEMINI_API_KEY and TOGETHER_API_KEY in your environment")
genai.configure(api_key=GEMINI_KEY)


# ——— LOAD DATA & MODELS ———
df         = pd.read_csv(CATALOG_CSV)
embeddings = np.load(EMB_FILE)
index      = faiss.read_index(INDEX_FILE)
embedder   = SentenceTransformer("all-MiniLM-L6-v2")

# ——— LOAD SKILL MAP ———
with open(SKILL_MAP_FILE) as f:
    SKILL_MAP = {k.lower(): v for k, v in json.load(f).items()}

# ——— HELPERS ———
def parse_duration(text: str) -> int | None:
    t = (text or "").lower()
    hrs  = sum(int(h) for h in re.findall(r"(\d+)\s*(?:h|hour)s?", t))
    mins = sum(int(m) for m in re.findall(r"(\d+)\s*(?:m|min)utes?", t))
    return (hrs * 60 + mins) or None

def safe_json_extract(txt: str) -> dict:
    try:
        body = txt[txt.find("{"): txt.rfind("}")+1]
        return json.loads(body)
    except:
        return {}

def default_constraints(dur: int|None) -> dict:
    return {
        "skills": [], 
        "max_duration": dur,
        "remote_required": "no", 
        "adaptive_required": "no",
        "test_type": None
    }

def merge_constraints(parsed: dict, fallback: int|None) -> dict:
    if not isinstance(parsed, dict):
        return default_constraints(fallback)
    return {
        "skills":            parsed.get("skills", []),
        "max_duration":      parsed.get("max_duration") or fallback,
        "remote_required":   parsed.get("remote_required", "no"),
        "adaptive_required": parsed.get("adaptive_required", "no"),
        "test_type":         parsed.get("test_type")
    }

# ——— CONSTRAINT EXTRACTION ———
PROMPT_TMPL = """
You are an intelligent assistant helping recruiters choose the right assessments.

Given a natural language query about a hiring requirement, extract the following details as a valid JSON object:
- "skills": A list of technical or soft skills mentioned.
- "max_duration": Maximum assessment duration in minutes (integer).
- "remote_required": "yes" or "no".
- "adaptive_required": "yes" or "no".
- "test_type": e.g., "technical", "cognitive", etc.

Respond ONLY with a JSON object. No extra text.

Query:
{query}
""".strip()

def extract_constraints(query: str, engine: str = "gemini") -> dict:
    pre_max = parse_duration(query)
    prompt  = PROMPT_TMPL.format(query=query)

    if engine == "gemini":
        try:
            mdl  = genai.GenerativeModel("models/gemini-1.5-pro-latest")
            resp = mdl.generate_content(prompt)
            txt  = getattr(resp, "text", resp.parts[0].text)
        except:
            return default_constraints(pre_max)

    elif engine == "together":
        try:
            headers = {"Authorization": f"Bearer {TOGETHER_KEY}"}
            data = {
                "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "prompt": prompt,
                "max_tokens": 200,
                "temperature": 0.3
            }
            r = requests.post("https://api.together.xyz/v1/completions", headers=headers, json=data, timeout=60)
            txt = r.json()["choices"][0]["text"]
        except:
            return default_constraints(pre_max)

    else:
        raise ValueError("Unsupported engine. Use 'gemini' or 'together'.")

    return merge_constraints(safe_json_extract(txt), pre_max)



# ——— SKILL MAP LOOKUP WITH QUERY FALLBACK ———
def by_skill_map(query: str, skills: list[str]) -> pd.DataFrame:
    slugs = []
    # 1) explicit skills from LLM
    for s in skills:
        slugs.extend(SKILL_MAP.get(s.lower(), []))

    # 2) fallback: any skill_map key found in the raw query text
    if not slugs:
        qlow = query.lower()
        for key, vals in SKILL_MAP.items():
            if key in qlow:
                slugs.extend(vals)

    slugs = list(dict.fromkeys(slugs))
    if not slugs:
        return pd.DataFrame()

    mask = df["relative_url"].str.contains("|".join(slugs), na=False)
    sub  = df.loc[mask].copy()
    sub.insert(0, "query", query)
    sub["score"] = 1.0
    return sub[["query"] + list(df.columns) + ["score"]].reset_index(drop=True)

# ——— FAISS RETRIEVAL & RANKING ———
def retrieve_assessments(query: str, cons: dict, top_k: int = 10) -> pd.DataFrame:
    def filter_df(df, cons) -> pd.DataFrame:
        filt = df.copy()
        if cons["max_duration"]:
            filt = filt[filt.duration <= cons["max_duration"]]
        if cons["remote_required"] == "yes":
            filt = filt[filt.remote.fillna("no").str.lower() == "yes"]
        if cons["adaptive_required"] == "yes":
            filt = filt[filt.adaptive.fillna("no").str.lower() == "yes"]
        if cons["test_type"]:
            t = [cons["test_type"]] if isinstance(cons["test_type"], str) else cons["test_type"]
            filt = filt[filt.test_type.fillna("").str.lower().apply(
                lambda x: any(tt.lower() in x for tt in t)
            )]
        return filt

    fallback_steps = [
        cons,
        {**cons, "adaptive_required": "no"},
        {**cons, "adaptive_required": "no", "test_type": None},
        {**cons, "adaptive_required": "no", "test_type": None, "remote_required": "no"}
    ]

    for fallback in fallback_steps:
        filt = filter_df(df, fallback)
        if not filt.empty:
            print(f"✅ Matched with fallback: {fallback}")
            qv   = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
            D, I = index.search(qv, top_k * 5)
            hits = []
            for dist, idx in zip(D[0], I[0]):
                if idx in filt.index:
                    rec = filt.loc[idx].to_dict()
                    rec["query"] = query
                    rec["score"] = dist
                    hits.append(rec)
                if len(hits) >= top_k:
                    break
            if hits:
                out = pd.DataFrame(hits)
                out["score"] = MinMaxScaler().fit_transform(out[["score"]])
                return out.sort_values("score", ascending=False)[
                    ["query"] + list(df.columns) + ["score"]
                ]

    return pd.DataFrame()

# ——— MAIN ———
def recommend(query: str, engine: str = "gemini") -> pd.DataFrame:
    cons   = extract_constraints(query, engine)
    skills = cons.get("skills", [])
    print("🧠 Constraints:", cons)
    df_map = by_skill_map(query, skills)
    if not df_map.empty:
        print("🔧 Skill-map matched slugs")
        return df_map
    return retrieve_assessments(query, cons)

if __name__ == "__main__":
    example_query = """
I am looking for a COO for my 
company in China and I want to see 
if they are culturally a right fit for 
our company. Suggest me an 
assessment that they can complete 
in about an hour 
"""
    res = recommend(example_query, engine="gemini")
    if res.empty:
        print("⚠️ No assessments matched your criteria.")
    else:
        print(res.to_markdown(index=False))
