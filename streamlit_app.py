# shl_recommender_app.py
import streamlit as st
import pandas as pd
import math
import warnings
from main import recommend, get_keys_and_configure   # <- your own modules

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
st.sidebar.header("⚙️ Options")
model = st.sidebar.selectbox("Choose Model", ["gemini", "together"])

# ────────────────────────────────────────────────────────────────
# 4. Main UI
# ────────────────────────────────────────────────────────────────
st.title("🔍 SHL Assessment Recommendation System")
query = st.text_area("Enter Job Description", height=200)

if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("⚠️ Please enter a job description.")
    else:
        try:
            # 4‑A. Call your recommender
            df_raw = recommend(query, engine=model)

            # 4‑B. Normalise & select columns
            records = []
            for rec in df_raw.to_dict(orient="records"):
                # replace NaN floats with None
                for k, v in rec.items():
                    if isinstance(v, float) and math.isnan(v):
                        rec[k] = None
                records.append(
                    {
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

            # 4‑C. Build a DataFrame
            df_display = pd.DataFrame(records)[
                [
                    "Assessment Name",
                    "Duration",
                    "Remote",
                    "Adaptive Support",
                    "Test Type",
                    "URL",
                ]
            ].fillna("–")

            # 4‑D. Turn URL column into clickable <a>
            df_display["URL"] = df_display["URL"].apply(
                lambda u: f'<a href="{u}" target="_blank">Link</a>' if u else "–"
            )

            # 4‑E. Convert to HTML (escape=False keeps the <a> tags)
            table_html = df_display.to_html(
                index=False,
                escape=False,
                classes="shl-table",
            )

            # 4‑F. Inject a tiny bit of CSS once, then the table
            st.markdown(
                """
                <style>
                .shl-table {
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 0.94rem;
                }
                .shl-table th, .shl-table td {
                    padding: 8px 10px;
                    border-bottom: 1px solid #ddd;
                    text-align: left;
                }
                .shl-table th {background: #f2f2f2;}
                .shl-table td a {color: #1a73e8; text-decoration: none;}
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(table_html, unsafe_allow_html=True)

        except Exception as e:
            st.exception(f"🚫 An error occurred:\n{e}")
