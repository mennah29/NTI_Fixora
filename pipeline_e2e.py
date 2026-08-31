import os
import json
import sys
from smart_retriever import retrieve_solution
from phase3_llm_engine import MaintenanceLLMEngine

sys.stdout.reconfigure(encoding='utf-8')

def run_biomedical_assistant(technician_input: str, device_name: str = None, top_k: int = 3):
    print("=" * 70)
    print(f"🔧 INCOMING TECHNICIAN QUERY: '{technician_input}'")
    if device_name:
        print(f"📟 SELECTED EQUIPMENT       : {device_name}")
    print("=" * 70)

    # 1. Phase 2: Hybrid FAISS Retrieval
    print("\n🔍 [Step 1: FAISS Semantic Retrieval] Searching manuals...")
    retrieval_res = retrieve_solution(
        technician_input=technician_input,
        device_name=device_name,
        top_k=top_k
    )

    detected_code = retrieval_res["detected_code"]
    top_chunks = retrieval_res["top_chunks"]

    print(f"   Detected Code : {detected_code or 'None'}")
    print(f"   Chunks Found  : {len(top_chunks)}")
    for i, c in enumerate(top_chunks, 1):
        hazard_icon = "⚠️" if str(c.get("has_safety_hazard")).lower() in ["true", "1"] else "📄"
        print(f"   {hazard_icon} [{i}] {c.get('manual')} (Page {c.get('page')})")

    # 2. Phase 3: Strict LLM Inference via Ollama (Qwen 2.5 7B)
    print("\n🧠 [Step 2: LLM Synthesis via Qwen 2.5 7B (Ollama)] Processing...")
    engine = MaintenanceLLMEngine(model_name="qwen2.5:7b", temperature=0.1)
    
    final_payload = engine.generate_response(
        device_name=device_name or "Biomedical Equipment",
        technician_query=technician_input,
        retrieved_chunks=top_chunks
    )

    # 3. Output payload for Audio TTS & UI
    print("\n" + "=" * 70)
    print("📢 [Phase 3 Final Output Payload]:")
    print("=" * 70)
    print(json.dumps(final_payload, indent=2, ensure_ascii=False))

    return final_payload

if __name__ == "__main__":
    test_query = "The monitor has high voltage alarm, what are the safety precautions before opening?"
    target_device = "Siemens Ag"
    run_biomedical_assistant(technician_input=test_query, device_name=target_device, top_k=3)
