import streamlit as st
import os
import time
import sys
import uuid

# Set UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

from audio_production import ProductionAudioEngine
from phase3_llm_engine import MaintenanceLLMEngine
from smart_retriever import retrieve_solution
from langchain_maintenance_engine import ask_assistant

st.set_page_config(page_title="Biomedical Live Voice & Diagnostic Assistant", layout="wide", page_icon="🏥")

@st.cache_resource
def load_all_engines():
    audio = ProductionAudioEngine()
    llm = MaintenanceLLMEngine(model_name="qwen2.5:7b", temperature=0.1)
    return audio, llm

with st.spinner("⏳ Loading AI & Audio Engines (Whisper + Kokoro + Qwen 2.5 + LangChain)..."):
    audio_engine, llm_engine = load_all_engines()

if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"

# ─────────────────────────────────────────────────────────────
# Sidebar Setup
# ─────────────────────────────────────────────────────────────
st.sidebar.title("🛠️ Device & Mode Settings")

# Real Canonical Devices from Knowledge Base
EQUIPMENT_OPTIONS = [
    "G40 Patient Monitor",
    "Philips V24, V25 Agilent Monitor",
    "Sc 6002Xl Patient Monitor",
    "Siemens Ag 2017",
    "SIEMENS CIOS SELECT",
    "SIEMENS MAGNETOM SKYRA",
    "SIEMENS SLIDING GANTRY",
    "SIEMENS SOMATOM SCOPE",
    "philips Big Bore Family",
    "philips Ingenuity CT Family",
    "philips CT Rembra RT",
    "Ge Healthcare Ct Scanners",
    "ACUSON Freestyle Ultrasound",
    "ACUSON Origin ICE Ultrasound",
    "Epatch Sensor",
    "ALL DEVICES (Unrestricted)",
    "Other / Custom Equipment"
]

selected_device_choice = st.sidebar.selectbox("Target Medical Equipment:", EQUIPMENT_OPTIONS)
if selected_device_choice == "Other / Custom Equipment":
    selected_device = st.sidebar.text_input("Enter Custom Equipment Name:", value="Ventilator")
elif selected_device_choice == "ALL DEVICES (Unrestricted)":
    selected_device = None
else:
    selected_device = selected_device_choice

mode = st.sidebar.radio("Operating Mode:", ["💬 Diagnostic Chatbot", "📞 Live Voice Call"])
rag_engine_type = st.sidebar.radio("RAG Architecture:", [
    "🦜 LangChain LCEL (Multi-Turn Memory)",
    "⚡ Native Engine (Direct Streaming)"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔒 Safety & Isolation")
if selected_device:
    st.sidebar.success(f"🛡️ **Strict Boundary:** Chunks restricted exclusively to `{selected_device}`.")
else:
    st.sidebar.warning("⚠️ **Open Search:** Cross-equipment matching enabled.")

st.sidebar.markdown("### ⚡ System Stack")
st.sidebar.info(
    "**Framework:** LangChain LCEL + Ollama\n\n"
    "**Memory:** RunnableWithMessageHistory\n\n"
    "**STT:** Whisper Turbo (CUDA)\n\n"
    "**RAG:** FAISS + BGE-Small (Strict Filter)\n\n"
    "**LLM:** Qwen 2.5 7B (Ollama Local)\n\n"
    "**TTS:** Kokoro-82M Neural Speech"
)

def process_query_native(user_text: str):
    search_data = retrieve_solution(device_name=selected_device, technician_input=user_text, top_k=5)
    return llm_engine.generate_response(
        device_name=selected_device or "Biomedical Equipment",
        technician_query=user_text,
        retrieved_chunks=search_data["top_chunks"]
    )

# ─────────────────────────────────────────────────────────────
# MODE 1: DIAGNOSTIC CHATBOT
# ─────────────────────────────────────────────────────────────
if mode == "💬 Diagnostic Chatbot":
    st.title("💬 Interactive Service Chatbot")
    disp_dev = selected_device if selected_device else "All Devices"
    st.caption(f"Diagnostic Session: **{disp_dev}** | Session ID: `{st.session_state.session_id}`")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        if st.button("🔄 Reset Conversation"):
            st.session_state.chat_history = []
            st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
            st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg.get("has_safety_hazard"):
                st.error("🚨 HIGH RISK / SAFETY HAZARD FLAGGED")
            st.markdown(msg["content"])
            if msg.get("citations"):
                st.caption(f"📄 Reference: {', '.join(msg['citations'])}")

    if prompt := st.chat_input("Enter fault symptoms, or follow up on previous steps..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if "LangChain" in rag_engine_type:
                # LangChain LCEL with Multi-turn Memory
                stream_gen, meta = ask_assistant(
                    device_name=selected_device or "Biomedical Equipment",
                    question=prompt,
                    session_id=st.session_state.session_id,
                    stream=True
                )
                if meta.get("has_safety_hazard"):
                    st.error("⚠️ HIGH RISK WARNING — ADHERE TO SAFETY DIRECTIVES")

                full_response = st.write_stream(stream_gen)
                if meta.get("citations"):
                    st.caption(f"📄 Reference: {', '.join(meta['citations'])}")
                hazard_flag = meta.get("has_safety_hazard", False)
                cit_list = meta.get("citations", [])
            else:
                # Native Engine
                search_data = retrieve_solution(device_name=selected_device, technician_input=prompt, top_k=5)
                chunks = search_data["top_chunks"]
                stream_gen, meta = llm_engine.generate_response(
                    device_name=selected_device or "Biomedical Equipment",
                    technician_query=prompt,
                    retrieved_chunks=chunks,
                    stream=True
                )
                if meta.get("has_safety_hazard"):
                    st.error("⚠️ HIGH RISK WARNING — ADHERE TO SAFETY DIRECTIVES")

                full_response = st.write_stream(stream_gen)
                if meta.get("citations"):
                    st.caption(f"📄 Reference: {', '.join(meta['citations'])}")
                hazard_flag = meta.get("has_safety_hazard", False)
                cit_list = meta.get("citations", [])

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_response,
            "has_safety_hazard": hazard_flag,
            "citations": cit_list
        })

# ─────────────────────────────────────────────────────────────
# MODE 2: LIVE VOICE CALL
# ─────────────────────────────────────────────────────────────
else:
    st.title("📞 Live Field Voice Channel")
    disp_dev = selected_device if selected_device else "All Devices"
    st.caption(f"Active Audio Session: **{disp_dev}** | Session: `{st.session_state.session_id}`")

    col_mic, col_resp = st.columns([1, 1], gap="large")

    with col_mic:
        st.subheader("🎙️ Voice Input")
        st.write("Speak clearly describing the issue or following up on previous repair steps.")
        audio_stream = st.audio_input("Press record and describe the issue:")

        if audio_stream:
            t0 = time.time()
            temp_in = "temp_input_voice.wav"
            with open(temp_in, "wb") as f:
                f.write(audio_stream.read())

            with st.spinner("Transcribing with Whisper..."):
                transcript = audio_engine.transcribe(temp_in)
            st_time = time.time() - t0

            st.success(f"🗣️ **Technician:** \"{transcript}\"")
            st.caption(f"⏱️ STT Latency: `{st_time:.2f}s`")

            with st.spinner("Processing diagnosis with conversational memory..."):
                t_llm0 = time.time()
                if "LangChain" in rag_engine_type:
                    spoken_text = ask_assistant(
                        device_name=selected_device or "Biomedical Equipment",
                        question=transcript,
                        session_id=st.session_state.session_id,
                        stream=False
                    )
                    has_hazard = any(w in spoken_text.upper() for w in ["WARNING", "DANGER", "CAUTION", "HIGH RISK"])
                    citations = [f"{selected_device} Manual"]
                else:
                    rag_out = process_query_native(transcript)
                    spoken_text = rag_out["spoken_text"]
                    has_hazard = rag_out.get("has_safety_hazard", False)
                    citations = rag_out.get("citations", [])

                out_audio_file = audio_engine.synthesize(spoken_text, "temp_out.wav")
                total_time = time.time() - t0

            with col_resp:
                st.subheader("🔊 Assistant Response")
                if has_hazard:
                    st.error("🚨 CRITICAL SAFETY WARNING DETECTED — ADHERE TO DIRECTIVES")

                st.audio(out_audio_file, format="audio/wav", autoplay=True)
                st.markdown(f"### 📋 Spoken Diagnostic Protocol\n{spoken_text}")
                
                if citations:
                    st.markdown("#### 📄 Authorized Manual Citations")
                    for citation in citations:
                        st.markdown(f"- `{citation}`")
                        
                st.caption(f"⚡ Total End-to-End Latency: `{total_time:.2f}s`")
