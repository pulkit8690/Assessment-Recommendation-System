# shl_recommender_app.py  ── run with:  streamlit run shl_recommender_app.py
import streamlit as st
import pandas as pd
import math
import warnings
from main import recommend, get_keys_and_configure   # ← your own modules

warnings.filterwarnings("ignore", category=UserWarning)

# ────────────────────────────────────────────────────────────────
# 1. Page‑level config
# ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

# ────────────────────────────────────────────────────────────────
# 2. API keys
# ────────────────────────────────────────────────────────────────
try:
    GEMINI_KEY, TOGETHER_KEY = get_keys_and_configure()
    st.sidebar.success("✅ API keys loaded")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load API keys:\n{e}")

# ────────────────────────────────────────────────────────────────
# 3. Sidebar – choose LLM back‑end
# ────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️  Options")
model = st.sidebar.selectbox("Choose Model", ["gemini", "together"])

# ────────────────────────────────────────────────────────────────
# 4. Main UI
# ────────────────────────────────────────────────────────────────
st.title("🔍 SHL Assessment Recommendation System")
query = st.text_area("Enter Job Description", height=200)

if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("⚠️  Please enter a job description.")
        st.stop()

    try:
        # 4‑A. Call recommender
        df_raw = recommend(query, engine=model)

        # 4‑B. Build list of dicts with serial numbers
        records = []
        for idx, rec in enumerate(df_raw.to_dict(orient="records"), start=1):
            for k, v in rec.items():
                if isinstance(v, float) and math.isnan(v):
                    rec[k] = None
            records.append(
                {
                    "#":                idx,
                    "Assessment Name": rec.get("assessment_name"),
                    "Duration":         rec.get("duration"),
                    "Remote":           rec.get("remote"),
                    "Adaptive Support": rec.get("adaptive"),
                    "Test Type":        rec.get("test_type") or "–",
                    "URL":              rec.get("relative_url"),
                }
            )

        if not records:
            st.info("No recommendations found.")
            st.stop()

        # 4‑C. Build HTML table with high‑contrast colours
        table_html = """
        <style>
        /* ------------- dark theme (default in Streamlit desktop) ------------- */
        .shl-table            {width: 100%; border-collapse: collapse; font-size: 0.95rem;}
        .shl-table th         {background: #36393f; color: #ffffff; padding: 10px; text-align: left;}
        .shl-table td         {padding: 9px 10px; color: #eaeaea;}
        .shl-table tr:nth-child(odd)  {background: #1e1e1e;}
        .shl-table tr:nth-child(even) {background: #26282c;}
        .shl-table td, .shl-table th  {border-bottom: 1px solid #4a4a4a;}
        .shl-table td a       {color: #3ea6ff; text-decoration: none;}
        .shl-table td a:hover {text-decoration: underline;}

        /* ------------- light theme override ------------- */
        @media (prefers-color-scheme: light) {
            .shl-table th         {background: #f2f2f2; color: #222;}
            .shl-table td         {color: #222;}
            .shl-table tr:nth-child(odd)  {background: #ffffff;}
            .shl-table tr:nth-child(even) {background: #f7f7f7;}
            .shl-table td, .shl-table th  {border-bottom: 1px solid #dddddd;}
            .shl-table td a       {color: #1a0dab;}
        }
        </style>

        <table class="shl-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Assessment Name</th>
              <th>Duration</th>
              <th>Remote</th>
              <th>Adaptive Support</th>
              <th>Test Type</th>
              <th>URL</th>
            </tr>
          </thead>
          <tbody>
        """

        for row in records:
            table_html += f"""
              <tr>
                <td>{row['#']}</td>
                <td>{row['Assessment Name']}</td>
                <td>{row['Duration']}</td>
                <td>{row['Remote']}</td>
                <td>{row['Adaptive Support']}</td>
                <td>{row['Test Type']}</td>
                <td><a href="{row['URL']}" target="_blank">Link</a></td>
              </tr>
            """

        table_html += "</tbody></table>"

        # 4‑D. Render inside iframe for guaranteed HTML rendering
        st.components.v1.html(table_html, height=600, scrolling=True)

    except Exception as e:
        st.exception(f"🚫 An error occurred:\n{e}")
