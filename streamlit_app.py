import threading
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import streamlit as st
import pandas as pd
import math
import warnings
import requests
from main import recommend, get_keys_and_configure

warnings.filterwarnings("ignore", category=UserWarning)

# ——— 1. FastAPI server (port 8000) ———
api = FastAPI()

@api.get("/health")
async def health():
    return {"status": "ok"}

@api.post("/recommend")
async def recommend_api(payload: dict):
    query = (payload.get("query") or "").strip()
    if not query:
        return JSONResponse(status_code=400, content={"results": [], "error": "Provide a non-empty 'query'."})
    try:
        df = recommend(query, engine="gemini")
        out = []
        for rec in df.to_dict(orient="records"):
            # sanitize floats
            for k, v in rec.items():
                if isinstance(v, float) and math.isnan(v):
                    rec[k] = None
            # build filtered record
            out.append({
                "assessment_name": rec.get("assessment_name"),
                "duration": rec.get("duration"),
                "remote": rec.get("remote"),
                "adaptive_support": rec.get("adaptive"),  # adaptive_support flag
                "test_type": rec.get("test_type"),
                "url": rec.get("relative_url")
            })
        return {"results": out, "error": None}
    except Exception:
        return JSONResponse(status_code=500, content={"results": [], "error": "Internal processing error."})

def run_api():
    uvicorn.run(api, host="0.0.0.0", port=8000, log_level="warning")

# Launch FastAPI in background
threading.Thread(target=run_api, daemon=True).start()


# ——— 2. Streamlit UI (port 8501) ———
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

# Load API keys (optional)
try:
    GEMINI_KEY, _ = get_keys_and_configure()
    st.sidebar.success("✅ API keys loaded")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load API keys: {e}")

st.sidebar.header("⚙️ Options")
st.sidebar.write("Using Gemini model only")
st.sidebar.checkbox("🌙 Enable Dark Mode", key="dark_mode")

st.title("🔍 SHL Assessment Recommendation System")
query = st.text_input("Enter Job Description")

if st.button("Get Recommendations"):
    if not query:
        st.warning("Please enter a job description.")
    else:
        resp = requests.post(
            "http://localhost:8000/recommend",
            json={"query": query},
            headers={"Content-Type": "application/json"}
        )
        if resp.status_code != 200:
            st.error(f"API Error {resp.status_code}: {resp.json().get('error')}")
        else:
            data = resp.json().get("results", [])
            if not data:
                st.info("No recommendations found.")
            else:
                df = pd.DataFrame(data).rename(columns={
                    "assessment_name": "Assessment Name",
                    "duration": "Duration",
                    "remote": "Remote",
                    "adaptive_support": "Adaptive Support",
                    "test_type": "Test Type",
                    "url": "URL"
                })[[
                    "Assessment Name",
                    "Duration",
                    "Remote",
                    "Adaptive Support",
                    "Test Type",
                    "URL"
                ]]
                # Make URLs clickable
                df["URL"] = df["URL"].apply(lambda u: f"[Link]({u})" if u else "-")
                st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
