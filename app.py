import streamlit as st
import pandas as pd
from main import recommend  # Uses your existing recommend() logic
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---- Page Configuration ----
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---- Styling ----
st.markdown("""
    <style>
    .reportview-container {
        background: linear-gradient(to right, #ece9e6, #ffffff);
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        font-weight: 600;
        border-radius: 5px;
        padding: 0.6rem 1rem;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Title ----
st.title("🔍 SHL Assessment Recommendation Engine")
st.caption("Built with 🧠 Gemini, TogetherAI, FAISS & Streamlit")

# ---- Input Form ----
with st.form("query_form"):
    query = st.text_area("Enter your hiring query", 
        placeholder="e.g. I need to hire a software engineer with AI skills",
        height=150)

    model = st.selectbox("Select AI Model", ["gemini", "together"])
    submitted = st.form_submit_button("Find Matching Assessments")

# ---- Query Processing ----
if submitted and query.strip():
    with st.spinner("Finding best assessments for your query..."):
        try:
            df = recommend(query, engine=model)
            if df.empty:
                st.warning("No matching assessments found. Try a broader query.")
            else:
                st.success("Assessments found!")
                display_df = df[["assessment_name", "duration", "remote", "test_type", "relative_url"]]
                display_df = display_df.rename(columns={
                    "assessment_name": "Assessment Name",
                    "duration": "Duration (min)",
                    "remote": "Remote",
                    "test_type": "Test Type",
                    "relative_url": "URL"
                })
                display_df["URL"] = display_df["URL"].apply(lambda x: f"[Link]({x})")
                st.markdown("### 📋 Recommended Assessments")
                st.write(display_df.to_markdown(index=False), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ---- Footer ----
st.markdown("""
---
Made by [Pulkit Arora](https://www.linkedin.com/in/pulkit-arora-731b17227/) using Streamlit 🚀
""")