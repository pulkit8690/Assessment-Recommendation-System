import pandas as pd
import re

# 1. Load the CSV
raw_path = "./data/assessments.csv"
df = pd.read_csv(raw_path)

# 2. Clean column names
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# 3. Extract numeric duration from assessment_length with a flexible regex
def extract_minutes(text):
    if pd.isna(text):
        return None
    s = str(text).lower()
    # pattern "minutes = 17" or "minutes:17"
    m = re.search(r"minutes?[\s:=]*\s*(\d+)", s)
    if m:
        return int(m.group(1))
    # fallback pattern "17 minutes"
    m = re.search(r"(\d+)\s*minutes?", s)
    if m:
        return int(m.group(1))
    return None

df["duration"] = df["assessment_length"].apply(extract_minutes)

# 4. Treat any zero‐minute values as missing (optional)
df.loc[df["duration"] == 0, "duration"] = None

# 5. Impute missing durations by median within each test_type, then overall
#    (so that we don’t default everything to one huge median)
overall_median = df["duration"].median()
df["duration"] = (
    df["duration"]
      .fillna(df.groupby("test_type")["duration"].transform("median"))
      .fillna(overall_median)
      .astype(int)
)

# 6. Normalize remote/adaptive fields (unchanged)
df["remote"]   = df["remote_testing"].astype(str).str.strip().str.lower().fillna("no")
df["adaptive"] = df["adaptive/irt"].   astype(str).str.strip().str.lower().fillna("no")

# 7. Clean 'test_type' and 'assessment_name' (unchanged)
df["test_type"]       = df["test_type"].astype(str).str.strip().str.lower().fillna("unspecified")
df["assessment_name"] = df["assessment_name"].fillna("unnamed test")
df["assessment_length"] = df["assessment_length"].fillna("")

# 8. Build text field for semantic embedding (unchanged)
df["raw_text"] = (
    df["assessment_name"] + " " +
    df["test_type"] + " " +
    df["assessment_length"]
)

# 9. Select and save cleaned catalog
final_df = df[[
    "assessment_name",
    "relative_url",
    "duration",
    "remote",
    "adaptive",
    "test_type",
    "raw_text"
]]
final_df.to_csv("./data/cleaned_catalog.csv", index=False)
print("✅ Cleaned data saved to ./data/cleaned_catalog.csv")
