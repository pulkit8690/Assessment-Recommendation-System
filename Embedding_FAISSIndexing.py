import os
import json
import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer

DATA_DIR    = "./data"
CLEANED_CSV = os.path.join(DATA_DIR, "cleaned_catalog.csv")
EMB_FILE    = os.path.join(DATA_DIR, "embeddings.npy")
INDEX_FILE  = os.path.join(DATA_DIR, "faiss_index.bin")
ID_MAP_FILE = os.path.join(DATA_DIR, "id_map.json")

# 1) Load only the columns we need
df = pd.read_csv(CLEANED_CSV, usecols=["assessment_name","relative_url","raw_text"])

# 2) Setup the model on GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# 3) Encode in one shot (batched & normalized)
texts = df["raw_text"].tolist()
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")

# 4) Save embeddings
np.save(EMB_FILE, embeddings, allow_pickle=False)

# 5) Build id_map with a single vectorized call
#    orient="index" gives {row_idx: {col: val, ...}, ...}
id_map = df[["assessment_name","relative_url"]] \
           .to_dict(orient="index")
with open(ID_MAP_FILE, "w") as f:
    json.dump(id_map, f, indent=2)

# 6) Build a FAISS index using all threads
faiss.omp_set_num_threads(os.cpu_count())
dim   = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # inner-product on L2-normalized => cosine
index.add(embeddings)

# 7) Persist the index
faiss.write_index(index, INDEX_FILE)

print("✅ Done: embeddings, id_map, and FAISS index saved.")
