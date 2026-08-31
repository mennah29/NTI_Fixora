import os
import sys
import time
import base64
import re
import uuid
import json
from typing import Optional, List
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Set UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

from audio_production import ProductionAudioEngine
from phase3_llm_engine import MaintenanceLLMEngine
from smart_retriever import retrieve_solution
from langchain_maintenance_engine import ask_assistant

app = FastAPI(title="Fixora Live Voice Telemetry API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AI & Audio Engines
print("⏳ Initializing Fixora Voice & AI Engines...")
audio_engine = ProductionAudioEngine()
llm_engine = MaintenanceLLMEngine(model_name="qwen2.5:7b", temperature=0.1)
print("✅ Fixora Live Voice Backend is Ready.")

EQUIPMENT_CATALOG = [
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
    "ALL DEVICES (Unrestricted)"
]

class ChatQuery(BaseModel):
    query: str
    device: str = "G40 Patient Monitor"
    session_id: Optional[str] = "live_call_001"
    use_langchain: bool = True

def parse_steps_from_text(raw_text: str):
    lines = [l.strip() for l in raw_text.strip().split("\n") if l.strip()]
    hazard_warning = None
    steps = []
    overview_lines = []
    step_regex = re.compile(r'^(?:\d+\.|\bStep\s*\d+:?)\s*(.+)', re.IGNORECASE)

    for line in lines:
        if any(h in line.upper() for h in ["WARNING:", "DANGER:", "CAUTION:", "HIGH VOLTAGE", "HIGH RISK"]):
            if not hazard_warning:
                hazard_warning = line.replace("[HIGH RISK HAZARD]", "").replace("⚠️", "").strip()
                continue
        match = step_regex.match(line)
        if match:
            steps.append(match.group(1).strip())
        elif not steps:
            overview_lines.append(line)

    return hazard_warning, " ".join(overview_lines), steps

@app.post("/api/chat")
async def chat_endpoint(payload: ChatQuery):
    t0 = time.time()
    user_query = payload.query.strip()
    device = payload.device
    session_id = payload.session_id

    if not user_query:
        return JSONResponse({"error": "Empty query"}, status_code=400)

    # 1. Diagnostic Inference (LangChain or Native)
    if payload.use_langchain:
        spoken_text = ask_assistant(device_name=device, question=user_query, session_id=session_id, stream=False)
        search_data = retrieve_solution(device_name=device, technician_input=user_query, top_k=3)
        citations = [f"{c['manual']} (Page {c['page']})" for c in search_data["top_chunks"] if c.get("page")]
    else:
        search_data = retrieve_solution(device_name=device, technician_input=user_query, top_k=5)
        rag_out = llm_engine.generate_response(device_name=device, technician_query=user_query, retrieved_chunks=search_data["top_chunks"], stream=False)
        spoken_text = rag_out["spoken_text"]
        citations = rag_out.get("citations", [])

    # 2. Neural Audio Synthesis (Kokoro-82M)
    out_audio_file = f"voice_{uuid.uuid4().hex[:6]}.wav"
    audio_path = audio_engine.synthesize(spoken_text, out_audio_file)
    
    # Encode Audio to Base64
    audio_base64 = ""
    if os.path.exists(audio_path):
        with open(audio_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")
        try:
            os.remove(audio_path)
        except:
            pass

    hazard_warning, overview, steps = parse_steps_from_text(spoken_text)
    # Persist voice conversation to history DB
    try:
        hist_path = os.path.join(r"D:\New folder (6)\Maintience NTI", "fixora_saved_sessions.json")
        sessions = {}
        if os.path.exists(hist_path):
            with open(hist_path, "r", encoding="utf-8") as f:
                sessions = json.load(f)
        cur = sessions.get(session_id, {
            "id": session_id,
            "title": f"Voice: {user_query[:20]}...",
            "device": device,
            "timestamp": time.strftime("%Y-%m-%d %H:%M"),
            "messages": []
        })
        cur["messages"].append({"role": "user", "content": user_query})
        cur["messages"].append({"role": "assistant", "content": spoken_text, "citations": citations})
        cur["timestamp"] = time.strftime("%Y-%m-%d %H:%M")
        sessions[session_id] = cur
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error persisting voice session: {e}")

    total_time = round(time.time() - t0, 2)

    return {
        "spoken_text": spoken_text,
        "hazard_warning": hazard_warning,
        "overview": overview,
        "steps": steps,
        "citations": citations,
        "audio_base64": audio_base64,
        "latency_sec": total_time,
        "device": device
    }

@app.get("/", response_class=HTMLResponse)
async def live_call_ui():
    options_html = "".join([f"<option value='{d}'>{d}</option>" for d in EQUIPMENT_CATALOG])
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fixora — Live Voice Telemetry Call</title>
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}
            body {{
                font-family: 'Inter', sans-serif;
                background-color: #ffffff;
                color: #1e293b;
                display: flex;
                height: 100vh;
                overflow: hidden;
            }}

            /* Zone A: Left Sidebar */
            .sidebar {{
                width: 280px;
                background-color: #f8fafc;
                border-right: 1px solid #e2e8f0;
                display: flex;
                flex-direction: column;
                padding: 20px 16px;
                flex-shrink: 0;
            }}
            .sidebar-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 20px;
            }}
            .logo-wrap {{
                display: flex;
                align-items: center;
                gap: 8px;
                font-weight: 700;
                font-size: 1.2rem;
                color: #0b496b;
            }}
            .new-chat-btn {{
                background-color: #0b496b;
                color: #ffffff;
                border: none;
                border-radius: 20px;
                padding: 10px 16px;
                font-size: 0.9rem;
                font-weight: 600;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 6px;
                margin-bottom: 24px;
                transition: all 0.2s ease;
            }}
            .new-chat-btn:hover {{
                background-color: #083750;
                transform: translateY(-1px);
            }}
            .sidebar-label {{
                font-size: 0.72rem;
                font-weight: 700;
                color: #64748b;
                letter-spacing: 0.06em;
                margin-bottom: 8px;
                text-transform: uppercase;
            }}
            .recent-session-pill {{
                background: #eef6fc;
                border: 1px solid #cbe3f7;
                color: #0b496b;
                border-radius: 20px;
                padding: 8px 14px;
                font-size: 0.85rem;
                font-weight: 600;
                margin-bottom: 24px;
                cursor: pointer;
            }}
            .device-select {{
                width: 100%;
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 0.85rem;
                color: #1e293b;
                margin-bottom: 20px;
                outline: none;
            }}
            .sidebar-footer {{
                margin-top: auto;
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 12px;
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
            }}
            .avatar {{
                width: 36px;
                height: 36px;
                border-radius: 50%;
                background: #0b496b;
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 700;
                font-size: 0.85rem;
            }}

            /* Zone B: Main Stage */
            .main-content {{
                flex: 1;
                display: flex;
                flex-direction: column;
                overflow-y: auto;
                padding: 24px 36px;
                position: relative;
            }}
            .call-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                width: 100%;
                max-width: 860px;
                margin: 0 auto 20px auto;
            }}
            .back-btn {{
                background: #0b496b;
                color: white;
                border: none;
                border-radius: 20px;
                padding: 6px 18px;
                font-size: 0.85rem;
                font-weight: 600;
                cursor: pointer;
            }}
            .end-call-btn {{
                background-color: #D92D20;
                color: white;
                border: none;
                border-radius: 20px;
                padding: 6px 20px;
                font-size: 0.85rem;
                font-weight: 700;
                cursor: pointer;
                transition: all 0.2s ease;
            }}
            .end-call-btn:hover {{
                background-color: #b42318;
            }}

            /* Concentric Telemetry Orb */
            .orb-stage {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                margin: 20px 0;
            }}
            .orb-container {{
                position: relative;
                width: 260px;
                height: 260px;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            @keyframes pulse-orb {{
                0% {{ transform: scale(0.96); box-shadow: 0 0 0 0 rgba(24, 138, 114, 0.5); }}
                70% {{ transform: scale(1.05); box-shadow: 0 0 0 35px rgba(24, 138, 114, 0); }}
                100% {{ transform: scale(0.96); box-shadow: 0 0 0 0 rgba(24, 138, 114, 0); }}
            }}
            @keyframes ripple {{
                0% {{ transform: scale(0.85); opacity: 0.8; }}
                100% {{ transform: scale(1.6); opacity: 0; }}
            }}
            .voice-orb {{
                width: 180px;
                height: 180px;
                border-radius: 50%;
                background: radial-gradient(circle at 35% 35%, #25b89a 0%, #188a72 55%, #0a5244 100%);
                box-shadow: 0 0 45px rgba(24, 138, 114, 0.45);
                animation: pulse-orb 2.8s infinite ease-in-out;
                z-index: 2;
                transition: all 0.3s ease;
            }}
            .voice-orb.speaking {{
                animation: pulse-orb 1.2s infinite ease-in-out;
                background: radial-gradient(circle at 35% 35%, #34d399 0%, #10b981 55%, #065f46 100%);
                box-shadow: 0 0 65px rgba(16, 185, 129, 0.6);
            }}
            .voice-orb.thinking {{
                animation: pulse-orb 1s infinite ease-in-out;
                background: radial-gradient(circle at 35% 35%, #60a5fa 0%, #0b496b 60%, #082f49 100%);
                box-shadow: 0 0 50px rgba(11, 73, 107, 0.5);
            }}
            .pulse-ring {{
                position: absolute;
                width: 180px;
                height: 180px;
                border-radius: 50%;
                border: 2px solid rgba(24, 138, 114, 0.4);
                animation: ripple 2.8s infinite cubic-bezier(0.2, 0.8, 0.2, 1);
                z-index: 1;
            }}
            .ring-2 {{ animation-delay: 0.9s; }}
            .ring-3 {{ animation-delay: 1.8s; }}

            .status-pill {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background: rgba(24, 138, 114, 0.1);
                color: #188a72;
                border: 1px solid rgba(24, 138, 114, 0.25);
                border-radius: 20px;
                padding: 6px 18px;
                font-size: 0.85rem;
                font-weight: 700;
                letter-spacing: 0.06em;
                margin-top: 14px;
            }}
            .status-dot {{
                width: 9px;
                height: 9px;
                border-radius: 50%;
                background-color: #188a72;
                box-shadow: 0 0 10px #188a72;
            }}

            /* Transcript & Procedure Card */
            .display-stage {{
                width: 100%;
                max-width: 860px;
                margin: 20px auto;
            }}
            .transcript-bubble {{
                background-color: #eef6fc;
                border: 1px solid #cbe3f7;
                border-radius: 16px;
                padding: 12px 18px;
                color: #0c324c;
                font-size: 0.95rem;
                margin-bottom: 16px;
                display: none;
            }}

            .gpt-proc-card {{
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 16px rgba(11, 73, 107, 0.06);
                margin-top: 14px;
                display: none;
            }}
            .gpt-proc-header {{
                background: #0b496b;
                color: #ffffff;
                padding: 10px 18px;
                font-size: 0.8rem;
                font-weight: 700;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .gpt-proc-body {{
                padding: 16px 20px;
            }}
            .gpt-proc-step {{
                display: flex;
                align-items: flex-start;
                gap: 12px;
                margin-bottom: 12px;
                font-size: 0.92rem;
                line-height: 1.5;
            }}
            .step-badge {{
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.8rem;
                font-weight: 700;
                background: #eef6fc;
                color: #0b496b;
                border: 1px solid #bfdbfe;
                border-radius: 6px;
                padding: 2px 7px;
                flex-shrink: 0;
            }}
            .gpt-proc-footer {{
                background: #f8fafc;
                border-top: 1px solid #e2e8f0;
                padding: 9px 18px;
                font-size: 0.78rem;
                color: #64748b;
                display: flex;
                justify-content: space-between;
            }}

            .safety-banner {{
                background-color: #fef2f2;
                border: 1px solid #fecaca;
                border-left: 5px solid #D92D20;
                border-radius: 8px;
                padding: 12px 16px;
                margin-bottom: 12px;
                color: #991b1b;
                font-size: 0.88rem;
                display: none;
            }}

            /* Floating Bottom Mic Button */
            .mic-controls {{
                position: fixed;
                bottom: 30px;
                left: calc(50% + 140px);
                transform: translateX(-50%);
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 8px;
                z-index: 10;
            }}
            .mic-btn {{
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background: #ffffff;
                border: 2px solid #188a72;
                box-shadow: 0 4px 18px rgba(24, 138, 114, 0.25);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.6rem;
                cursor: pointer;
                transition: all 0.2s ease;
                outline: none;
            }}
            .mic-btn.recording {{
                background: #188a72;
                color: white;
                transform: scale(1.08);
                box-shadow: 0 0 25px rgba(24, 138, 114, 0.6);
            }}
            .mic-hint {{
                font-size: 0.8rem;
                color: #64748b;
                background: rgba(255,255,255,0.9);
                padding: 3px 10px;
                border-radius: 12px;
            }}
        </style>
    </head>
    <body>
        <!-- ZONE A: SIDEBAR -->
        <div class="sidebar">
            <div class="sidebar-header">
                <div class="logo-wrap">
                    <svg width="32" height="32" viewBox="0 0 100 100" fill="none">
                        <polygon points="50,6 88,28 88,72 50,94 12,72 12,28" fill="#0b496b" stroke="#062638" stroke-width="3"/>
                        <polygon points="50,15 80,32 80,68 50,85 20,68 20,32" fill="#072d42"/>
                        <circle cx="50" cy="50" r="22" stroke="#188a72" stroke-width="3.5" stroke-dasharray="5 3"/>
                        <circle cx="50" cy="50" r="11" fill="#188a72"/>
                        <circle cx="50" cy="50" r="4.5" fill="#ffffff"/>
                    </svg>
                    <span>Fixora</span>
                </div>
                <span style="font-size: 1.1rem; color: #64748b;">🔍</span>
            </div>

            <button class="new-chat-btn" onclick="resetCall()">✚ New chat</button>

            <div class="sidebar-label">Recent Sessions</div>
            <div class="recent-session-pill">💬 Welcome Session</div>

            <div class="sidebar-label">Target Equipment Manual</div>
            <select class="device-select" id="deviceSelect">
                {options_html}
            </select>

            <div class="sidebar-footer">
                <div class="avatar">ME</div>
                <div>
                    <div style="font-size: 0.86rem; font-weight: 600; color: #0b496b;">Menna</div>
                    <div style="font-size: 0.72rem; color: #64748b;">Biomedical Engineer</div>
                </div>
            </div>
        </div>

        <!-- ZONE B: MAIN STAGE -->
        <div class="main-content">
            <div class="call-header">
                <button class="back-btn" onclick="window.location.href='http://localhost:8501'">← Back to Chat</button>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.2rem; color: #64748b;">←</span>
                    <h2 style="color: #0b496b; font-size: 1.35rem; font-weight: 700;">Fixora Voice</h2>
                </div>
                <button class="end-call-btn" onclick="endCall()">END CALL</button>
            </div>

            <!-- Pulsing Concentric Orb -->
            <div class="orb-stage">
                <div class="orb-container">
                    <div class="pulse-ring ring-3"></div>
                    <div class="pulse-ring ring-2"></div>
                    <div class="pulse-ring"></div>
                    <div class="voice-orb" id="voiceOrb"></div>
                </div>
                <div class="status-pill" id="statusPill">
                    <span class="status-dot" id="statusDot"></span>
                    <span id="statusText">LIVE</span>
                </div>
            </div>

            <!-- Transcript & Procedure Output -->
            <div class="display-stage">
                <div class="transcript-bubble" id="userBubble"></div>
                <div class="safety-banner" id="safetyBanner"></div>
                
                <div class="gpt-proc-card" id="procCard">
                    <div class="gpt-proc-header">⚙️ RECOMMENDED MAINTENANCE PROCEDURE</div>
                    <div class="gpt-proc-body" id="procBody"></div>
                    <div class="gpt-proc-footer">
                        <span id="procCitation"></span>
                        <span id="procStepCount" style="font-family: 'IBM Plex Mono', monospace; font-weight: 700; color: #0b496b;"></span>
                    </div>
                </div>
            </div>

            <!-- Floating Mic Controller -->
            <div class="mic-controls">
                <button class="mic-btn" id="micBtn" onclick="toggleVoiceDuplex()">🎙️</button>
                <span class="mic-hint" id="micHint">Click mic to start live conversation</span>
            </div>
        </div>

        <script>
            let isRecording = false;
            let recognition = null;
            let currentAudio = null;
            let silenceTimer = null;
            let lastSpeechTime = Date.now();

            const voiceOrb = document.getElementById('voiceOrb');
            const statusPill = document.getElementById('statusPill');
            const statusText = document.getElementById('statusText');
            const micBtn = document.getElementById('micBtn');
            const micHint = document.getElementById('micHint');
            const userBubble = document.getElementById('userBubble');
            const safetyBanner = document.getElementById('safetyBanner');
            const procCard = document.getElementById('procCard');
            const procBody = document.getElementById('procBody');
            const procCitation = document.getElementById('procCitation');
            const procStepCount = document.getElementById('procStepCount');
            const deviceSelect = document.getElementById('deviceSelect');

            // Initialize Web Speech Recognition
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (SpeechRecognition) {{
                recognition = new SpeechRecognition();
                recognition.continuous = false;
                recognition.interimResults = true;
                recognition.lang = 'en-US';

                recognition.onstart = function() {{
                    isRecording = true;
                    micBtn.classList.add('recording');
                    micBtn.innerHTML = '🛑';
                    statusText.innerText = 'LISTENING...';
                    micHint.innerText = 'Speak now — stops automatically on silence';
                    voiceOrb.className = 'voice-orb';
                }};

                recognition.onresult = function(event) {{
                    let transcript = '';
                    for (let i = event.resultIndex; i < event.results.length; ++i) {{
                        transcript += event.results[i][0].transcript;
                    }}
                    userBubble.style.display = 'block';
                    userBubble.innerHTML = '🗣️ "' + transcript + '"';
                    lastSpeechTime = Date.now();
                }};

                recognition.onspeechend = function() {{
                    statusText.innerText = 'THINKING...';
                    voiceOrb.className = 'voice-orb thinking';
                }};

                recognition.onend = function() {{
                    isRecording = false;
                    micBtn.classList.remove('recording');
                    micBtn.innerHTML = '🎙️';
                    
                    const queryText = userBubble.innerText.replace('🗣️ "', '').replace('"', '').trim();
                    if (queryText && queryText.length > 2) {{
                        sendVoiceQuery(queryText);
                    }} else {{
                        statusText.innerText = 'LIVE';
                        voiceOrb.className = 'voice-orb';
                        micHint.innerText = 'Click to speak again';
                    }}
                }};

                recognition.onerror = function(event) {{
                    console.log('Speech error:', event.error);
                    statusText.innerText = 'LIVE';
                    voiceOrb.className = 'voice-orb';
                    micBtn.classList.remove('recording');
                    micBtn.innerHTML = '🎙️';
                }};
            }} else {{
                alert('Your browser does not support Speech Recognition. Please use Google Chrome or Microsoft Edge.');
            }}

            function toggleVoiceDuplex() {{
                if (currentAudio && !currentAudio.paused) {{
                    currentAudio.pause();
                }}
                if (isRecording) {{
                    recognition.stop();
                }} else {{
                    userBubble.innerText = '';
                    userBubble.style.display = 'none';
                    recognition.start();
                }}
            }}

            async function sendVoiceQuery(text) {{
                statusText.innerText = 'DIAGNOSING...';
                voiceOrb.className = 'voice-orb thinking';
                micHint.innerText = 'Fixora AI reasoning...';

                try {{
                    const res = await fetch('/api/chat', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            query: text,
                            device: deviceSelect.value,
                            session_id: 'session_live_call'
                        }})
                    }});

                    const data = await res.json();
                    
                    // Render Procedure Card
                    if (data.hazard_warning) {{
                        safetyBanner.style.display = 'block';
                        safetyBanner.innerHTML = '<strong>⚠️ CRITICAL SAFETY MANDATE:</strong> ' + data.hazard_warning;
                    }} else {{
                        safetyBanner.style.display = 'none';
                    }}

                    if (data.steps && data.steps.length > 0) {{
                        let stepsHtml = '';
                        data.steps.forEach((s, idx) => {{
                            const num = String(idx + 1).padStart(2, '0');
                            stepsHtml += `<div class="gpt-proc-step"><span class="step-badge">${{num}}</span><span>${{s}}</span></div>`;
                        }});
                        procBody.innerHTML = stepsHtml;
                        procCitation.innerText = '📄 ' + (data.citations[0] || data.device + ' Service Manual');
                        procStepCount.innerText = data.steps.length + ' STEPS';
                        procCard.style.display = 'block';
                    }}

                    // Play Neural Voice (Kokoro Autoplay)
                    if (data.audio_base64) {{
                        statusText.innerText = 'SPEAKING...';
                        voiceOrb.className = 'voice-orb speaking';
                        micHint.innerText = 'Listening automatically after response...';

                        if (currentAudio) {{ currentAudio.pause(); }}
                        currentAudio = new Audio("data:audio/wav;base64," + data.audio_base64);
                        currentAudio.play();

                        // Loop: As soon as assistant finishes speaking, automatically listen again!
                        currentAudio.onended = function() {{
                            statusText.innerText = 'LISTENING...';
                            voiceOrb.className = 'voice-orb';
                            setTimeout(() => {{
                                if (!isRecording) {{
                                    recognition.start();
                                }}
                            }}, 600);
                        }};
                    }} else {{
                        statusText.innerText = 'LIVE';
                        voiceOrb.className = 'voice-orb';
                    }}

                }} catch (e) {{
                    console.error(e);
                    statusText.innerText = 'ERROR';
                    voiceOrb.className = 'voice-orb';
                }}
            }}

            function endCall() {{
                if (currentAudio) {{ currentAudio.pause(); }}
                if (isRecording && recognition) {{ recognition.stop(); }}
                window.location.href = 'http://localhost:8501';
            }}

            function resetCall() {{
                if (currentAudio) {{ currentAudio.pause(); }}
                userBubble.style.display = 'none';
                safetyBanner.style.display = 'none';
                procCard.style.display = 'none';
                statusText.innerText = 'LIVE';
                voiceOrb.className = 'voice-orb';
            }}
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
