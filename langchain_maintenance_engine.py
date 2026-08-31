import os
import sys
import ssl
import torch
from typing import List, Dict, Any

# ── SSL bypass for corporate/restricted networks ──────────────────────────────
os.environ["CURL_CA_BUNDLE"]                  = ""
os.environ["REQUESTS_CA_BUNDLE"]              = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
ssl._create_default_https_context             = ssl._create_unverified_context

sys.stdout.reconfigure(encoding='utf-8')

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# ─────────────────────────────────────────────────────────────
# 1. Path & Hardware Configuration
# ─────────────────────────────────────────────────────────────
BASE_DIR = r"D:\New folder (6)\Maintience NTI"
INDEX_DIR = os.path.join(BASE_DIR, "faiss_bge_small_index")
device = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────
# 2. Embedding Model & Vector Store Initialization
# ─────────────────────────────────────────────────────────────
print(f"📦 [LangChain Engine] Loading FAISS index with BGE on [{device.upper()}]...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": device},
    encode_kwargs={
        "normalize_embeddings": True,
        "prompt": "Represent this sentence for searching relevant passages: "
    }
)

vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

# ─────────────────────────────────────────────────────────────
# 3. LLM Configuration via Ollama
# ─────────────────────────────────────────────────────────────
print("🤖 [LangChain Engine] Initializing ChatOllama (Qwen 2.5 7B, temp=0.1)...")
llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.1,
    num_ctx=2048,
    num_predict=180
)

# ─────────────────────────────────────────────────────────────
# 4. Prompt Template with Conversational Memory & Safety Rules
# ─────────────────────────────────────────────────────────────
system_prompt_template = """You are an expert Biomedical Equipment Field Support Assistant.
You are assisting a technician in troubleshooting: '{device}'.

CRITICAL CONSTRAINTS & OPERATIONAL GUIDELINES:
1. Zero Cross-Equipment Bleeding: Rely EXCLUSIVELY on the authorized manual context for '{device}'. Never reference another machine.
2. Safety First: If any [HIGH RISK HAZARD] is flagged in the context, your FIRST sentence MUST be a prominent SAFETY WARNING.
3. Exhaustive Hardware Troubleshooting:
   - Provide direct, numbered sequential actions (e.g. 1. cable check -> 2. board reseating -> 3. component replacement).
   - If hardware replacement is indicated, state the replacement part and cited removal page.
   - Keep answers direct and concise (under 80 words) for immediate spoken audio feedback.
4. Grounding: Do not invent part numbers or procedures not found in the manual context.
5. Traceability: Cite manual and page number for each action.

Manual Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_template),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# ─────────────────────────────────────────────────────────────
# 5. Helper Functions
# ─────────────────────────────────────────────────────────────
def format_docs_with_metadata(docs: List) -> tuple[str, List[str], bool]:
    """Formats retrieved chunks with citations and hazard indicators."""
    formatted_chunks = []
    citations = []
    has_any_hazard = False

    for doc in docs:
        manual = doc.metadata.get("manual", "Manual")
        page = doc.metadata.get("page", "N/A")
        hazard = str(doc.metadata.get("has_safety_hazard", False)).lower() in ["true", "1", "yes"]
        if hazard:
            has_any_hazard = True
            hazard_label = "⚠️ [HIGH RISK HAZARD]"
        else:
            hazard_label = "ℹ️ [STANDARD PROCEDURE]"

        citation_str = f"{manual} (Page {page})"
        if citation_str not in citations:
            citations.append(citation_str)
        
        chunk_str = f"{hazard_label} [Source: {citation_str}]\n{doc.page_content}"
        formatted_chunks.append(chunk_str)

    return "\n\n".join(formatted_chunks), citations, has_any_hazard

def create_isolated_retriever(device_name: str, top_k: int = 5):
    """Creates a device-isolated retriever preventing cross-manual leaks."""
    if not device_name or device_name.strip().upper() in ["ALL DEVICES", "ALL", ""]:
        return vectorstore.as_retriever(search_kwargs={"k": top_k})

    device_key = device_name.lower()
    for noise in [
        "patient monitor", "service manual", "user manual", "copy of",
        "instructions for use", "quick manual", "manual", "system", "monitor"
    ]:
        device_key = device_key.replace(noise, "")
    device_key = device_key.strip()
    
    def filter_func(metadata: dict) -> bool:
        doc_device = str(metadata.get("device", "")).lower()
        doc_manual = str(metadata.get("manual", "")).lower()
        return bool(device_key and (device_key in doc_device or device_key in doc_manual))

    return vectorstore.as_retriever(
        search_kwargs={
            "k": top_k,
            "filter": filter_func
        }
    )

# ─────────────────────────────────────────────────────────────
# 6. Session History Store (For Multi-Turn Calls/Chats)
# ─────────────────────────────────────────────────────────────
session_store: Dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

# ─────────────────────────────────────────────────────────────
# 7. Main Invocation Function
# ─────────────────────────────────────────────────────────────
def ask_assistant(
    device_name: str, 
    question: str, 
    session_id: str = "default_session",
    stream: bool = False
):
    """
    Executes LangChain LCEL RAG pipeline with session history,
    strict device isolation, and multi-turn contextual memory.
    """
    retriever = create_isolated_retriever(device_name, top_k=5)
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        msg = f"I could not locate troubleshooting procedures for '{device_name}' matching that issue in the authorized service manual."
        if stream:
            def empty_gen():
                yield msg
            return empty_gen(), {"citations": [], "has_safety_hazard": False}
        return msg

    formatted_context, citations, has_hazard = format_docs_with_metadata(retrieved_docs)
    metadata = {
        "citations": citations,
        "has_safety_hazard": has_hazard,
        "device": device_name
    }

    # Build the LCEL Chain
    chain = (
        RunnablePassthrough.assign(
            context=lambda _: formatted_context,
            device=lambda _: device_name
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Wrap chain with session history
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )

    config = {"configurable": {"session_id": session_id}}

    if stream:
        token_stream = chain_with_history.stream({"question": question}, config=config)
        return token_stream, metadata

    response = chain_with_history.invoke({"question": question}, config=config)
    return response

# ─────────────────────────────────────────────────────────────
# Multi-Turn Unit Test Execution
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    target_device = "G40 Patient Monitor"
    test_session = "tech_call_001"

    print("\n" + "=" * 70)
    print(f"🏥 TESTING LANGCHAIN MULTI-TURN RAG FOR: {target_device}")
    print("=" * 70)

    print("\n--- [Turn 1: Initial Complaint] ---")
    query_1 = "The LCD screen is completely blank. What should I check first?"
    print(f"Technician: {query_1}")
    res_1 = ask_assistant(target_device, query_1, session_id=test_session)
    print(f"\nAssistant:\n{res_1}\n")

    print("--- [Turn 2: Follow-up relying on Memory] ---")
    query_2 = "I checked the cable and reseated it, but the screen is still black. What is the next step?"
    print(f"Technician: {query_2}")
    res_2 = ask_assistant(target_device, query_2, session_id=test_session)
    print(f"\nAssistant:\n{res_2}\n")
