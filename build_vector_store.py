import os
import ssl
import json
import sys
import time
import torch

# ── SSL bypass for corporate/restricted networks ──────────────────────────────
os.environ["CURL_CA_BUNDLE"]               = ""
os.environ["REQUESTS_CA_BUNDLE"]           = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
ssl._create_default_https_context          = ssl._create_unverified_context

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = r"D:\New folder (6)\Maintience NTI"
JSON_PATH  = os.path.join(BASE_DIR, "final_knowledge_base.json")
INDEX_DIR  = os.path.join(BASE_DIR, "faiss_bge_small_index")
MODEL      = "BAAI/bge-small-en-v1.5"   # 384-dim, cached locally, ~130MB
BATCH_SIZE = 64                           # GPU can handle larger batches

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load Knowledge Base
# ─────────────────────────────────────────────────────────────────────────────
print("[1/4] Loading knowledge base...")
if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(f"Cannot find {JSON_PATH}. Complete Phase 1 first.")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    raw_chunks = json.load(f)

print(f"      Loaded {len(raw_chunks):,} chunks.\n")

# Wrap into LangChain Documents
docs = [
    Document(
        page_content=c["content"],
        metadata={
            "device":            c.get("device",           "Unknown"),
            "manual":            c.get("manual",           "Unknown"),
            "page":              str(c.get("page",          1)),
            "strategy":          c.get("strategy",         "General"),
            "has_safety_hazard": str(c.get("has_safety_hazard", False)),
        }
    )
    for c in raw_chunks
]

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Load Embedding Model
# ─────────────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[2/4] Loading model: {MODEL}  [{device.upper()}]")
print(f"      First run downloads ~1.3GB — subsequent runs use local cache.\n")

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL,
    model_kwargs={"device": device},
    encode_kwargs={
        "normalize_embeddings": True,   # L2-normalize → cosine = dot product
        "batch_size": BATCH_SIZE,
        "prompt": "Represent this sentence for searching relevant passages: ",
    },
)
print(f"      Embedding dimension: 1024\n")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Build FAISS Index
# ─────────────────────────────────────────────────────────────────────────────
print(f"[3/4] Building FAISS index for {len(docs):,} documents...")
print(f"      Batch size: {BATCH_SIZE} | This may take 10–20 min on CPU.")
print(f"      Progress is shown every 500 documents.\n")

start = time.time()

# Build in sub-batches with progress reporting
REPORT_EVERY = 500
vector_db = None

for i in range(0, len(docs), REPORT_EVERY):
    batch = docs[i : i + REPORT_EVERY]
    if vector_db is None:
        vector_db = FAISS.from_documents(batch, embeddings)
    else:
        vector_db.add_documents(batch)

    processed = min(i + REPORT_EVERY, len(docs))
    elapsed   = time.time() - start
    rate      = processed / elapsed if elapsed > 0 else 1
    eta       = (len(docs) - processed) / rate
    pct       = processed / len(docs) * 100
    print(f"      [{pct:5.1f}%]  {processed:,}/{len(docs):,} docs  |  "
          f"{rate:.0f} docs/s  |  ETA: {eta/60:.1f} min")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Save Index to Disk
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(INDEX_DIR, exist_ok=True)
vector_db.save_local(INDEX_DIR)
total_time = time.time() - start

print(f"\n[4/4] FAISS index saved to: {INDEX_DIR}")
print(f"\n{'='*65}")
print(f"[DONE] Phase 2 Indexing Complete!")
print(f"       Total chunks indexed  : {len(docs):,}")
print(f"       Embedding model       : {MODEL}")
print(f"       Embedding dimensions  : 1024")
print(f"       Similarity metric     : Cosine (L2-normalized dot product)")
print(f"       Device used           : {device.upper()}")
print(f"       Total time            : {total_time/60:.1f} min")
print(f"       Index location        : {INDEX_DIR}")
print(f"{'='*65}")
print(f"\nNext step: run smart_retriever.py to test semantic queries.")
