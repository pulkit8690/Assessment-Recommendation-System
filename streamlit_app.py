import streamlit as st
import pandas as pd
import math
import warnings
from main import recommend, get_keys_and_configure

warnings.filterwarnings("ignore", category=UserWarning)

# ——— 1. Page config ———
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

# ——— 2. Load API keys (if needed) ———
try:
    GEMINI_KEY, TOGETHER_KEY = get_keys_and_configure()
    st.sidebar.success("✅ API keys loaded")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load API keys: {e}")

# ——— 3. Sidebar: model choice ———
st.sidebar.header("⚙️ Options")
model = st.sidebar.selectbox("Choose Model", ["gemini", "together"])

# ——— 4. Main UI ———
st.title("🔍 SHL Assessment Recommendation System")
query = st.text_area("Enter Job Description", height=200)

if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("⚠️ Please enter a job description.")
    else:
        try:
            # Call your recommend() directly
            df = recommend(query, engine=model)
            
            # Build filtered records
            records = []
            for rec in df.to_dict(orient="records"):
                # sanitize floats
                for k, v in rec.items():
                    if isinstance(v, float) and math.isnan(v):
                        rec[k] = None
                records.append({
                    "Assessment Name": rec.get("assessment_name"),
                    "Duration":       rec.get("duration"),
                    "Remote":         rec.get("remote"),
                    "Adaptive Support": rec.get("adaptive"),
                    "Test Type":      rec.get("test_type") or "–",
                    "URL":            rec.get("relative_url"),
                })

            if not records:
                st.info("No recommendations found.")
            else:
                # ——— HTML Table with Clickable Links ———
                table_html = """
                <style>
                  table {border-collapse: collapse; width:100%;}
                  th, td {padding:8px; border-bottom:1px solid #ddd; text-align:left;}
                  th {background:#f2f2f2;}
                  td a {color:blue; text-decoration: none;}
                </style>
                <table>
                  <thead>
                    <tr>
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
                for r in records:
                    table_html += f"""
                      <tr>
                        <td>{r['Assessment Name']}</td>
                        <td>{r['Duration']}</td>
                        <td>{r['Remote']}</td>
                        <td>{r['Adaptive Support']}</td>
                        <td>{r['Test Type']}</td>
                        <td><a href="{r['URL']}" target="_blank">Link</a></td>
                      </tr>
                    """
                table_html += "</tbody></table>"

                st.markdown(table_html, unsafe_allow_html=True)

        except Exception as e:
            st.exception(f"🚫 An error occurred: {e}")
