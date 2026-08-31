import os
import re
import sys
import ssl
import torch

# ── SSL bypass for corporate/restricted networks ──────────────────────────────
os.environ["CURL_CA_BUNDLE"]                  = ""
os.environ["REQUESTS_CA_BUNDLE"]              = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
ssl._create_default_https_context             = ssl._create_unverified_context

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR  = r"D:\New folder (6)\Maintience NTI"
INDEX_DIR = os.path.join(BASE_DIR, "faiss_bge_small_index")
MODEL     = "BAAI/bge-small-en-v1.5"

# ─────────────────────────────────────────────────────────────────────────────
# Load Model + Index (done once at module load)
# ─────────────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INIT] Loading embedding model: {MODEL} [{device.upper()}]")
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL,
    model_kwargs={"device": device},
    encode_kwargs={
        "normalize_embeddings": True,
        "prompt": "Represent this sentence for searching relevant passages: ",
    },
)

print(f"[INIT] Loading FAISS index from: {INDEX_DIR}")
vector_store = FAISS.load_local(
    INDEX_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)
print("[INIT] Ready.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Fault Code Pattern
# ─────────────────────────────────────────────────────────────────────────────
CODE_PATTERN = re.compile(
    r'\b(E\d{1,4}|F\d{1,4}|ERR[-_]?\d+|CODE\s*\d+|ALARM\s*\d+|A\d{3,4}|0x[0-9A-Fa-f]+)\b',
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────────────────────
# Strict Hard-Filtered Retriever (Zero Cross-Equipment Bleeding)
# ─────────────────────────────────────────────────────────────────────────────
def retrieve_solution(
    device_name: str = None,
    technician_input: str = None,
    top_k: int = 3,
    candidate_pool: int = 50,
    **kwargs
) -> dict:
    """
    Performs hybrid search combining exact code matching, semantic search,
    and 100% STRICT device isolation.
    """
    # Flexibility: handle if called with positional arguments in either order
    if device_name and len(device_name) > 40 and (technician_input is None or len(technician_input) < 40):
        device_name, technician_input = technician_input, device_name

    query = technician_input or ""

    # 1. Exact Regex code matching
    code_match = CODE_PATTERN.search(query)
    detected_code = code_match.group(0).upper() if code_match else None

    # 2. Construct search prompt
    if detected_code:
        search_query = f"Error Code: {detected_code} | Symptoms: {query}"
    else:
        search_query = query

    # 3. Dense similarity retrieval (retrieve broad candidate set)
    candidates = vector_store.similarity_search(search_query, k=candidate_pool)

    # 4. Strict Isolation 100% by Device Key
    if device_name and device_name.strip().upper() not in ["ALL DEVICES", "ALL", ""]:
        # Extract the distinguishing device model keyword (e.g. 'g40', '6002', 'v24', 'skyra', 'cios')
        device_key = device_name.lower()
        for noise in [
            "patient monitor", "service manual", "user manual", "copy of", 
            "instructions for use", "quick manual", "manual", "system", "monitor"
        ]:
            device_key = device_key.replace(noise, "")
        device_key = device_key.strip()

        isolated_results = []
        for doc in candidates:
            doc_device = str(doc.metadata.get("device", "")).lower()
            doc_manual = str(doc.metadata.get("manual", "")).lower()

            # Verify that the manual/device metadata strictly contains the device key
            if device_key and (device_key in doc_device or device_key in doc_manual):
                isolated_results.append(doc)
                if len(isolated_results) == top_k:
                    break

        final_docs = isolated_results
        isolation_enforced = True
    else:
        final_docs = candidates[:top_k]
        isolation_enforced = False

    # 5. Prevent ANY fallback to other equipment
    return {
        "detected_code": detected_code,
        "search_query":  search_query,
        "device_filter": device_name or "ALL DEVICES",
        "isolation_enforced": isolation_enforced,
        "matched_count": len(final_docs),
        "top_chunks": [
            {
                "manual":            doc.metadata.get("manual"),
                "page":              doc.metadata.get("page"),
                "strategy":          doc.metadata.get("strategy"),
                "has_safety_hazard": doc.metadata.get("has_safety_hazard", False),
                "content":           doc.page_content,
            }
            for doc in final_docs
        ]
    }

if __name__ == "__main__":
    print("Testing G40 Patient Monitor strict isolation...")
    r = retrieve_solution(device_name="G40 Patient Monitor", technician_input="screen blank alarm sound", top_k=3)
    print(f"Device: {r['device_filter']}")
    print(f"Matched count: {r['matched_count']}")
    for i, c in enumerate(r['top_chunks'], 1):
        print(f"  [{i}] {c['manual']} (Page {c['page']})")
