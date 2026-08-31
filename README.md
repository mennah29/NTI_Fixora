# 🏥 Fixora — Biomedical Field Service AI Assistant

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red?logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)
![Ollama](https://img.shields.io/badge/Qwen_2.5_7B-Ollama-orange)
![FAISS](https://img.shields.io/badge/FAISS-CUDA-blueviolet)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**An intelligent RAG-powered AI assistant for biomedical equipment field technicians**  
*Built with Qwen 2.5 7B · FAISS BGE Retrieval · Kokoro-82M Neural TTS · Whisper STT*

</div>

---

## ✨ Overview

**Fixora** is a production-grade biomedical AI assistant that helps field technicians troubleshoot and repair medical equipment in real time. It combines:

- 📚 **RAG (Retrieval-Augmented Generation)** — Grounded answers from 13,105 indexed OEM service manual chunks across 32 biomedical equipment models
- 🧠 **Qwen 2.5 7B** — Local, offline LLM inference via Ollama (no cloud dependency)
- 🔊 **Kokoro-82M Neural TTS** — Natural speech synthesis (RTF ≈ 0.18, faster than real-time)
- 🎙️ **Whisper Turbo STT** — Hands-free voice query transcription
- 💬 **LangChain Conversational Memory** — Multi-turn context across troubleshooting sessions
- 🛡️ **Zero Cross-Equipment Bleeding** — Strict per-device OEM manual isolation

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      FIXORA ARCHITECTURE                        │
├──────────────┬──────────────────────────────────────────────────┤
│  Frontend    │  Streamlit App (fixora_app.py) — Port 8501       │
│              │  • Device selector, Chat UI, Voice Call Mode     │
│              │  • Gemini-style persistent session history       │
│              │  • Dark / Light Mode with eye-comfort themes     │
├──────────────┼──────────────────────────────────────────────────┤
│  Backend API │  FastAPI Voice Server (fixora_live_voice.py)     │
│              │  → Port 8000 — /api/chat endpoint               │
├──────────────┼──────────────────────────────────────────────────┤
│  RAG Engine  │  Phase 1: FAISS BGE Retrieval (smart_retriever) │
│              │  Phase 2: Qwen 2.5 7B via Ollama (phase3_llm)   │
│              │  Phase 3: LangChain LCEL + Memory (langchain_*) │
├──────────────┼──────────────────────────────────────────────────┤
│  Speech      │  Whisper Turbo (STT) + Kokoro-82M (TTS)         │
│              │  → Base64 WAV streaming via API response        │
└──────────────┴──────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) with `qwen2.5:7b` pulled
- NVIDIA GPU (4GB+ VRAM recommended) or CPU fallback
- Git

### 2. Clone & Install

```bash
git clone https://github.com/mennah29/NTI_Fixora.git
cd NTI_Fixora
pip install -r requirements.txt
```

### 3. Pull the LLM

```bash
ollama pull qwen2.5:7b
```

### 4. Build the Vector Knowledge Base

> Place your OEM service manual PDFs inside a `valid_manuals/` folder, then run:

```bash
python build_vector_store.py
```

This will create `final_knowledge_base.json` and the `faiss_bge_small_index/` directory.

### 5. Launch the System

**Terminal 1 — Voice Backend:**
```bash
python fixora_live_voice.py
# → Uvicorn running on http://0.0.0.0:8000
```

**Terminal 2 — Streamlit Frontend:**
```bash
streamlit run fixora_app.py --server.port 8501
# → Open http://localhost:8501
```

---

## 📁 Project Structure

```
NTI_Fixora/
├── fixora_app.py                 # Streamlit UI (chat, history, voice call)
├── fixora_live_voice.py          # FastAPI backend + Kokoro TTS + Whisper STT
├── langchain_maintenance_engine.py  # LangChain LCEL + multi-turn memory
├── phase3_llm_engine.py          # Native Ollama RAG inference engine
├── smart_retriever.py            # FAISS + BGE hybrid retrieval
├── build_vector_store.py         # Knowledge base builder (PDF → FAISS)
├── audio_production.py           # Kokoro-82M TTS pipeline
├── pipeline_e2e.py               # End-to-end pipeline test runner
├── app_production.py             # Alternative production app entry point
├── Menna_Biomedical_AI_Resume.md # Author resume
├── .gitignore
└── README.md
```

---

## 🔧 Supported Equipment

The system supports 32+ biomedical device models including:

| Category | Devices |
|----------|---------|
| **Patient Monitors** | G40, Philips V24/V25 Agilent |
| **Imaging / X-Ray** | Siemens CIOS Select, GE Healthcare CT |
| **Ultrasound** | ACUSON Freestyle, ACUSON Origin ICE |
| **Ventilators / ICU** | Philips CT Rembra RT |
| **Diagnostics** | Epatch Sensor |
| **All Devices** | Unrestricted cross-manual search mode |

---

## 📊 Performance Benchmarks

| Metric | Value |
|--------|-------|
| Knowledge Base Chunks | 13,105 |
| OEM Manuals Indexed | 32 |
| Safety Hazard Chunks | 1,312 (10.01%) |
| FAISS Retrieval Latency | ~25–235 ms |
| Equipment Isolation Purity | **100%** (0% cross-bleeding) |
| Safety Compliance Rate | **100%** |
| Kokoro TTS Real-Time Factor | **0.18** (5.5× faster than real-time) |
| Voice Call End-to-End Latency | ~16–29 s |

---

## 🛡️ Safety Features

- ⚠️ **Mandatory Safety Warnings** — Any high-voltage or hazardous procedure triggers a safety mandate before any diagnostic steps
- 🔒 **Zero Cross-Equipment Contamination** — Responses are strictly limited to the selected device's OEM manual
- 📖 **Cited Page References** — Every diagnostic step includes the exact manual name and page number
- 🚫 **Hallucination Guard** — System refuses to invent part numbers or procedures not grounded in the manual context

---

## 💬 Voice Call Mode

The Live Call feature uses **WebRTC-style push-to-talk**:

1. Click the glowing **Orb** to activate voice recording
2. Speak your fault description naturally
3. Whisper Turbo transcribes in real time
4. Qwen 2.5 7B generates a grounded diagnostic response
5. Kokoro-82M synthesizes audio and plays it back automatically

---

## 👩‍💻 Author

**Menna Ashraf**  
Biomedical Engineering · AI & Medical Devices Specialist  
NTI Graduate Training Program — Biomedical AI Track

---

## 📄 License

This project is licensed under the MIT License.
