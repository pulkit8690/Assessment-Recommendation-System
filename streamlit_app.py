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
                # Create DataFrame
                df_display = pd.DataFrame(records)

                # Make URL clickable
                df_display["URL"] = df_display["URL"].apply(lambda url: f"[Link]({url})" if url else "–")

                # Optional: reorder columns
                df_display = df_display[[
                    "Assessment Name", "Duration", "Remote", "Adaptive Support", "Test Type", "URL"
                ]].fillna("–")

                # Display as Streamlit table
                st.dataframe(df_display, use_container_width=True)

        except Exception as e:
            st.exception(f"🚫 An error occurred: {e}")
