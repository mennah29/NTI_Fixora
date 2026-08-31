import streamlit as st
import os
import sys
import time
import uuid
import re
import textwrap

# Set UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# ── Safe imports — graceful fallback for Streamlit Cloud (no GPU/heavy models) ─
try:
    from audio_production import ProductionAudioEngine
    AUDIO_AVAILABLE = True
except Exception as _e:
    AUDIO_AVAILABLE = False
    ProductionAudioEngine = None

try:
    from phase3_llm_engine import MaintenanceLLMEngine
    LLM_AVAILABLE = True
except Exception as _e:
    LLM_AVAILABLE = False
    MaintenanceLLMEngine = None

try:
    from smart_retriever import retrieve_solution
    RETRIEVER_AVAILABLE = True
except Exception as _e:
    RETRIEVER_AVAILABLE = False
    def retrieve_solution(*args, **kwargs):
        return {"top_chunks": [], "detected_code": None}

try:
    from langchain_maintenance_engine import ask_assistant
    LANGCHAIN_AVAILABLE = True
except Exception as _e:
    LANGCHAIN_AVAILABLE = False
    def ask_assistant(*args, **kwargs):
        return "⚠️ AI engine not available in cloud demo mode. Please run locally with a GPU for full functionality."

# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fixora — Biomedical Field Service AI",
    layout="wide",
    page_icon="🔧",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────
if "technician_name" not in st.session_state:
    st.session_state.technician_name = "Menna"
if "selected_device" not in st.session_state:
    st.session_state.selected_device = "G40 Patient Monitor"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"fixora_{uuid.uuid4().hex[:8]}"
if "live_call_active" not in st.session_state:
    st.session_state.live_call_active = False
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "light"

# ─────────────────────────────────────────────────────────────
# Fixora Custom Inline SVG Logo
# ─────────────────────────────────────────────────────────────
def get_fixora_svg(size=36):
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align: middle;">'
        '<polygon points="50,6 88,28 88,72 50,94 12,72 12,28" fill="#0b496b" stroke="#072d42" stroke-width="3"/>'
        '<polygon points="50,15 80,32 80,68 50,85 20,68 20,32" fill="#083750"/>'
        '<circle cx="50" cy="50" r="22" stroke="#188a72" stroke-width="3.5" stroke-dasharray="5 3"/>'
        '<circle cx="50" cy="50" r="11" fill="#188a72"/>'
        '<circle cx="50" cy="50" r="4.5" fill="#ffffff"/>'
        '<line x1="50" y1="15" x2="50" y2="28" stroke="#188a72" stroke-width="2.5"/>'
        '<line x1="50" y1="72" x2="50" y2="85" stroke="#188a72" stroke-width="2.5"/>'
        '<line x1="20" y1="50" x2="28" y2="50" stroke="#188a72" stroke-width="2.5"/>'
        '<line x1="72" y1="50" x2="80" y2="50" stroke="#188a72" stroke-width="2.5"/>'
        '</svg>'
    )
# Persistent Chat History Storage (Gemini & ChatGPT Style)
# ─────────────────────────────────────────────────────────────
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "fixora_saved_sessions.json")

def load_all_sessions():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_session(session_id, device, messages):
    if not messages:
        return
    sessions = load_all_sessions()
    first_q = next((m["content"] for m in messages if m.get("role") == "user"), "Diagnostics")
    clean_title = first_q.replace("Power is on, but the monitor screen is completely blank", "Screen Blank Fault")
    clean_title = re.sub(r'[\r\n]+', ' ', clean_title).strip()
    if len(clean_title) > 24:
        clean_title = clean_title[:22] + "..."
    if not clean_title:
        clean_title = f"{device[:14]} Session"

    sessions[session_id] = {
        "id": session_id,
        "title": clean_title,
        "device": device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "messages": messages
    }
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving session: {e}")

def delete_session(session_id):
    sessions = load_all_sessions()
    if session_id in sessions:
        del sessions[session_id]
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(sessions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error deleting session: {e}")


# ─────────────────────────────────────────────────────────────
# CSS Styling (Clean, Zero Markdown Indentation Bugs)
# ─────────────────────────────────────────────────────────────
def inject_signature_styles():
    is_dark = st.session_state.theme_mode == "dark"
    # Relaxing, eye-friendly palette: Soft warm porcelain (#f8fafc) for light mode, deep obsidian (#080c14) for dark mode
    bg_main = "#080c14" if is_dark else "#f8fafc"
    text_color = "#f8fafc" if is_dark else "#1e293b"
    sub_color = "#94a3b8" if is_dark else "#64748b"
    sidebar_bg = "#04070d" if is_dark else "#f1f5f9"
    card_bg = "#0d1522" if is_dark else "#ffffff"
    border_col = "#1e293b" if is_dark else "#e2e8f0"
    user_bubble_bg = "#132337" if is_dark else "#f0f9ff"
    user_bubble_border = "#1e3a5f" if is_dark else "#bae6fd"
    user_bubble_text = "#f8fafc" if is_dark else "#0369a1"

    # Button styles differentiated between Dark & Relaxed Light
    tile_bg = "#0e1726" if is_dark else "#ffffff"
    tile_border = "#1e2d42" if is_dark else "#e2e8f0"
    tile_text = "#f8fafc" if is_dark else "#0f2942"
    tile_shadow = "0 4px 14px rgba(0, 0, 0, 0.3)" if is_dark else "0 2px 8px rgba(11, 73, 107, 0.05)"

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: {text_color};
    }}

    /* Global Full App Background & Header */
    .stApp {{
        background-color: {bg_main} !important;
        color: {text_color} !important;
    }}
    header[data-testid="stHeader"] {{
        background-color: {bg_main} !important;
    }}
    .stDeployButton, #MainMenu {{
        display: none !important;
        visibility: hidden !important;
    }}

    .main .block-container {{
        max-width: 900px !important;
        padding-top: 1rem !important;
        padding-bottom: 5rem !important;
    }}

    /* Sidebar Refinement */
    section[data-testid="stSidebar"] {{
        width: 270px !important;
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {border_col} !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: {text_color};
    }}

    /* Suggestion & Action Buttons: Relaxing Light Tiles vs Dark Obsidian */
    .stButton>button {{
        background: {tile_bg} !important;
        color: {tile_text} !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        border: 1.5px solid {tile_border} !important;
        padding: 12px 18px !important;
        transition: all 0.22s ease !important;
        box-shadow: {tile_shadow} !important;
        text-align: left !important;
    }}
    .stButton>button:hover {{
        background: {'#0b496b' if is_dark else '#f0fdf4'} !important;
        border-color: #188a72 !important;
        color: {'#ffffff' if is_dark else '#0b496b'} !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(24, 138, 114, {'0.35' if is_dark else '0.15'}) !important;
    }}

    /* Sidebar Primary CTA (New Chat) */
    section[data-testid="stSidebar"] div:has(> button:first-child) > button:first-child {{
        background: #0b496b !important;
        color: #ffffff !important;
        border: 1px solid #083750 !important;
        border-radius: 20px !important;
        text-align: center !important;
    }}
    section[data-testid="stSidebar"] div:has(> button:first-child) > button:first-child:hover {{
        background: #083750 !important;
        box-shadow: 0 4px 14px rgba(11, 73, 107, 0.25) !important;
    }}

    /* Sidebar Session Buttons */
    section[data-testid="stSidebar"] .stButton > button {{
        background-color: {tile_bg} !important;
        border: 1px solid {tile_border} !important;
        color: {tile_text} !important;
        border-radius: 14px !important;
        padding: 7px 14px !important;
        font-size: 0.81rem !important;
        font-weight: 500 !important;
        box-shadow: {tile_shadow} !important;
        text-align: left !important;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        background-color: {'#0b496b' if is_dark else '#f8fafc'} !important;
        border-color: #188a72 !important;
        color: {'#ffffff' if is_dark else '#0b496b'} !important;
    }}

    /* Delete Session Icon Button */
    section[data-testid="stSidebar"] div[data-testid="column"]:nth-child(2) .stButton > button {{
        background: transparent !important;
        border: 1px solid rgba(239, 68, 68, 0.35) !important;
        color: #ef4444 !important;
        border-radius: 10px !important;
        padding: 5px 6px !important;
        font-size: 0.8rem !important;
        min-width: 32px !important;
        text-align: center !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="column"]:nth-child(2) .stButton > button:hover {{
        background: rgba(239, 68, 68, {'0.2' if is_dark else '0.08'}) !important;
        border-color: #ef4444 !important;
        color: #dc2626 !important;
    }}

    /* End Call Red Button */
    .end-call-btn button {{
        background-color: #D92D20 !important;
        border-color: #ef4444 !important;
        color: #ffffff !important;
        border-radius: 20px !important;
        padding: 6px 18px !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        text-align: center !important;
    }}

    /* Top Back To Chat Pill */
    .back-chat-btn button {{
        background: #0b496b !important;
        border-color: #188a72 !important;
        color: #ffffff !important;
        border-radius: 22px !important;
        padding: 6px 20px !important;
        font-size: 0.85rem !important;
        text-align: center !important;
    }}

    /* Selectbox Styling */
    div[data-baseweb="select"] > div {{
        background-color: {card_bg} !important;
        border: 1.5px solid {border_col} !important;
        color: {text_color} !important;
        border-radius: 10px !important;
    }}
    div[data-baseweb="select"] span {{
        color: {text_color} !important;
    }}

    /* Fix Bottom Sticky Container */
    div[data-testid="stBottom"],
    div[data-testid="stBottom"] > div,
    div[data-testid="stBottomBlockContainer"],
    .stChatFloatingInputContainer,
    footer {{
        background-color: {bg_main} !important;
        background: {bg_main} !important;
        border-top: 1px solid {border_col} !important;
    }}

    /* Chat Input Styling */
    div[data-testid="stChatInput"] {{
        background-color: {card_bg} !important;
        border: 1.5px solid {'#1e293b' if is_dark else '#cbd5e1'} !important;
        border-radius: 26px !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, {'0.4' if is_dark else '0.05'}) !important;
    }}
    div[data-testid="stChatInput"]:focus-within {{
        border-color: #188a72 !important;
        box-shadow: 0 0 14px rgba(24, 138, 114, {'0.35' if is_dark else '0.2'}) !important;
    }}
    div[data-testid="stChatInput"] textarea {{
        color: {text_color} !important;
        background: transparent !important;
    }}
    div[data-testid="stChatInput"] button {{
        background-color: {'#1e293b' if is_dark else '#0b496b'} !important;
        color: {'#38bdf8' if is_dark else '#ffffff'} !important;
        border-radius: 50% !important;
    }}
    div[data-testid="stChatInput"] button:hover {{
        background-color: #188a72 !important;
        color: #ffffff !important;
    }}

    /* User Message Bubble */
    .user-bubble-container {{
        display: flex;
        justify-content: flex-end;
        margin: 12px 0;
    }}
    .user-bubble {{
        background-color: {user_bubble_bg};
        border: 1px solid {user_bubble_border};
        border-radius: 18px 18px 4px 18px;
        padding: 12px 18px;
        max-width: 82%;
        color: {user_bubble_text};
        font-size: 0.94rem;
        line-height: 1.5;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }}

    /* Procedure Execution Card (.gpt-proc-card) */
    .gpt-proc-card {{
        background: {card_bg};
        border: 1px solid {border_col};
        border-radius: 12px;
        overflow: hidden;
        margin: 14px 0 16px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }}
    .gpt-proc-header {{
        background: linear-gradient(135deg, #0b496b 0%, #072d42 100%);
        color: #ffffff;
        padding: 11px 18px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid rgba(24, 138, 114, 0.3);
    }}
    .gpt-proc-body {{
        padding: 16px 20px;
        background: {card_bg};
    }}
    .gpt-proc-step {{
        display: flex;
        align-items: flex-start;
        gap: 14px;
        margin-bottom: 14px;
        font-size: 0.92rem;
        line-height: 1.55;
        color: {text_color};
    }}
    .gpt-proc-step:last-child {{
        margin-bottom: 0;
    }}
    .gpt-step-badge {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        font-weight: 700;
        background: {user_bubble_bg};
        color: {'#34d399' if is_dark else '#0b496b'};
        border: 1px solid {'rgba(52, 211, 153, 0.3)' if is_dark else '#bfdbfe'};
        border-radius: 6px;
        padding: 3px 8px;
        flex-shrink: 0;
        margin-top: 1px;
    }}
    .gpt-proc-footer {{
        background: {sidebar_bg};
        border-top: 1px solid {border_col};
        padding: 10px 18px;
        font-size: 0.78rem;
        color: {sub_color};
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}

    .safety-critical {{
        background-color: rgba(217, 45, 32, 0.12);
        border: 1px solid rgba(217, 45, 32, 0.35);
        border-left: 5px solid #D92D20;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 10px 0;
        color: #f87171;
        font-size: 0.88rem;
        line-height: 1.45;
        display: flex;
        align-items: flex-start;
        gap: 10px;
    }}
    .citation-chip {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(11, 73, 107, 0.2);
        border: 1px solid rgba(11, 73, 107, 0.4);
        color: #38bdf8;
        font-size: 0.76rem;
        font-weight: 600;
        border-radius: 16px;
        padding: 3px 10px;
        margin: 4px 6px 4px 0;
    }}
    </style>
    """
    st.markdown(textwrap.dedent(css), unsafe_allow_html=True)

inject_signature_styles()

# Load Backend Engines
@st.cache_resource
def load_fixora_backends():
    audio = ProductionAudioEngine()
    llm = MaintenanceLLMEngine(model_name="qwen2.5:7b", temperature=0.1)
    return audio, llm

with st.spinner("Initializing Fixora Telemetry Core..."):
    audio_engine, llm_engine = load_fixora_backends()

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

# ─────────────────────────────────────────────────────────────
# PROCEDURE PARSER & CARD RENDERER
# ─────────────────────────────────────────────────────────────
def clean_and_parse_procedure(raw_text: str):
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

    overview = " ".join(overview_lines)
    return hazard_warning, overview, steps

def render_procedure_card(raw_text: str, citations: list, device: str):
    hazard_warning, overview, steps = clean_and_parse_procedure(raw_text)

    if hazard_warning:
        hazard_html = (
            f'<div class="safety-critical">'
            f'<span style="font-size: 1.15rem;">⚠️</span>'
            f'<div><strong>CRITICAL SAFETY DIRECTIVE</strong><br>{hazard_warning}</div>'
            f'</div>'
        )
        st.markdown(hazard_html, unsafe_allow_html=True)

    if overview:
        st.markdown(f"<div style='font-size: 0.94rem; margin: 8px 0 12px 0; line-height: 1.5;'>{overview}</div>", unsafe_allow_html=True)

    if steps and len(steps) >= 2:
        steps_html = ""
        for i, step in enumerate(steps, 1):
            badge = f"{i:02d}"
            clean_step = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', step)
            steps_html += f'<div class="gpt-proc-step"><span class="gpt-step-badge">{badge}</span><span>{clean_step}</span></div>'

        citation_text = citations[0] if citations else f"{device} Authorized Manual"
        card_html = (
            f'<div class="gpt-proc-card">'
            f'<div class="gpt-proc-header">⚙️ RECOMMENDED MAINTENANCE PROCEDURE</div>'
            f'<div class="gpt-proc-body">{steps_html}</div>'
            f'<div class="gpt-proc-footer"><span>📄 {citation_text}</span>'
            f'<span style="font-family: \'IBM Plex Mono\', monospace; font-weight: 700; color: #0b496b;">{len(steps)} STEPS</span></div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)
    elif not overview and not steps:
        st.markdown(raw_text)

    if citations:
        chips_html = "".join([f"<span class='citation-chip'>📑 {c}</span>" for c in citations])
        st.markdown(chips_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ZONE A: LEFT SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    sidebar_top = (
        f'<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px;">'
        f'<div style="display: flex; align-items: center; gap: 8px;">'
        f'{get_fixora_svg(32)}'
        f'<span style="font-weight: 700; font-size: 1.15rem; color: #0b496b; letter-spacing: -0.02em;">Fixora</span>'
        f'</div>'
        f'<span style="font-size: 1.1rem; color: #64748b; cursor: pointer;">🔍</span>'
        f'</div>'
    )
    st.markdown(sidebar_top, unsafe_allow_html=True)

    if st.button("✚ New chat", use_container_width=True):
        if st.session_state.chat_history:
            save_session(st.session_state.session_id, st.session_state.selected_device, st.session_state.chat_history)
        st.session_state.chat_history = []
        st.session_state.live_call_active = False
        st.session_state.session_id = f"fixora_{uuid.uuid4().hex[:8]}"
        st.rerun()

    st.markdown("<p style='font-size: 0.72rem; font-weight: 700; color: #64748b; letter-spacing: 0.05em; margin: 18px 0 6px 0;'>RECENT SESSIONS</p>", unsafe_allow_html=True)
    saved_db = load_all_sessions()
    if saved_db:
        sorted_s = sorted(saved_db.values(), key=lambda x: x.get("timestamp", ""), reverse=True)
        for s in sorted_s[:7]:
            s_id = s["id"]
            s_title = s.get("title", "Saved Session")
            is_active = (s_id == st.session_state.session_id)
            c_sess, c_del = st.columns([5, 1])
            with c_sess:
                lbl = f"💬 {s_title}" if not is_active else f"👉 {s_title}"
                if st.button(lbl, key=f"btn_{s_id}", use_container_width=True):
                    st.session_state.session_id = s_id
                    st.session_state.chat_history = s.get("messages", [])
                    st.session_state.selected_device = s.get("device", st.session_state.selected_device)
                    st.session_state.live_call_active = False
                    st.rerun()
            with c_del:
                if st.button("✕", key=f"del_{s_id}", help="Delete chat session"):
                    delete_session(s_id)
                    if is_active:
                        st.session_state.chat_history = []
                        st.session_state.session_id = f"fixora_{uuid.uuid4().hex[:8]}"
                    st.rerun()
    else:
        st.caption("No previous sessions.")

    st.markdown("<p style='font-size: 0.72rem; font-weight: 700; color: #64748b; letter-spacing: 0.05em; margin: 18px 0 4px 0;'>TARGET EQUIPMENT MANUAL</p>", unsafe_allow_html=True)
    selected_device_idx = EQUIPMENT_CATALOG.index(st.session_state.selected_device) if st.session_state.selected_device in EQUIPMENT_CATALOG else 0
    new_dev = st.selectbox("Equipment:", EQUIPMENT_CATALOG, index=selected_device_idx, label_visibility="collapsed")
    if new_dev != st.session_state.selected_device:
        st.session_state.selected_device = new_dev
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    if not st.session_state.live_call_active:
        if st.button("🎙️ Enter Voice Call Mode", use_container_width=True):
            st.session_state.live_call_active = True
            st.rerun()
    else:
        if st.button("💬 Return to Chatbot", use_container_width=True):
            st.session_state.live_call_active = False
            st.rerun()

    theme_label = "Switch Theme ( ☀️ Light)" if st.session_state.theme_mode == "dark" else "Switch Theme ( 🌙 Dark)"
    if st.button(theme_label, use_container_width=True):
        st.session_state.theme_mode = "light" if st.session_state.theme_mode == "dark" else "dark"
        st.rerun()

    if st.button("👤 Switch Technician", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.live_call_active = False
        st.rerun()

    profile_html = (
        f'<div style="display: flex; align-items: center; gap: 10px; margin-top: 24px; padding: 10px; background: rgba(11, 73, 107, 0.06); border-radius: 12px;">'
        f'<div style="width: 34px; height: 34px; border-radius: 50%; background: #0b496b; color: white; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.82rem;">ME</div>'
        f'<div>'
        f'<div style="font-size: 0.85rem; font-weight: 600; color: #0b496b;">{st.session_state.technician_name}</div>'
        f'<div style="font-size: 0.72rem; color: #64748b;">Biomedical Engineer</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(profile_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# VIEW A: SIGNATURE LIVE VOICE CALL MODE (ZERO RECORDER WIDGET)
# ─────────────────────────────────────────────────────────────
if st.session_state.live_call_active:
    import streamlit.components.v1 as components

    nav_c1, nav_c2, nav_c3 = st.columns([1, 1, 1])
    with nav_c2:
        st.markdown('<div class="back-chat-btn">', unsafe_allow_html=True)
        if st.button("← Back to Chat", use_container_width=True):
            st.session_state.live_call_active = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    header_l, header_r = st.columns([3, 1])
    with header_l:
        call_head = (
            f'<div style="display: flex; align-items: center; gap: 8px; margin-top: 6px;">'
            f'<span style="font-size: 1.1rem; color: #64748b;">←</span>'
            f'<h2 style="margin: 0; color: #0b496b; font-size: 1.3rem; font-weight: 700;">Fixora Voice Telemetry</h2>'
            f'</div>'
        )
        st.markdown(call_head, unsafe_allow_html=True)
    with header_r:
        st.markdown('<div class="end-call-btn">', unsafe_allow_html=True)
        if st.button("END CALL", use_container_width=True):
            st.session_state.live_call_active = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Dynamic Theme Variables for Live Call Component
    is_dark = st.session_state.theme_mode == "dark"
    call_card_bg = "#0d1522" if is_dark else "#ffffff"
    call_text = "#f8fafc" if is_dark else "#1e293b"
    call_border = "#1e293b" if is_dark else "#cbd5e1"
    call_bubble = "#132337" if is_dark else "#eef6fc"
    call_bubble_border = "#1e3a5f" if is_dark else "#cbe3f7"
    call_footer_bg = "#04070d" if is_dark else "#f8fafc"
    call_sub = "#94a3b8" if is_dark else "#64748b"

    # 100% Widget-Free ChatGPT Voice Experience (Themed)
    live_call_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Inter', sans-serif;
                background: transparent;
                color: {call_text};
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
                overflow: hidden;
            }}
            .orb-wrapper {{
                position: relative;
                width: 240px;
                height: 240px;
                margin: 20px auto 14px auto;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
            }}
            @keyframes pulse-orb {{
                0% {{ transform: scale(0.96); box-shadow: 0 0 0 0 rgba(24, 138, 114, 0.6); }}
                70% {{ transform: scale(1.05); box-shadow: 0 0 0 36px rgba(24, 138, 114, 0); }}
                100% {{ transform: scale(0.96); box-shadow: 0 0 0 0 rgba(24, 138, 114, 0); }}
            }}
            @keyframes ripple {{
                0% {{ transform: scale(0.85); opacity: 0.85; }}
                100% {{ transform: scale(1.58); opacity: 0; }}
            }}
            .voice-orb {{
                width: 175px;
                height: 175px;
                border-radius: 50%;
                background: radial-gradient(circle at 35% 35%, #25b89a 0%, #188a72 55%, #0a5244 100%);
                box-shadow: 0 0 50px rgba(24, 138, 114, 0.55);
                animation: pulse-orb 2.8s infinite ease-in-out;
                z-index: 2;
                transition: all 0.3s ease;
            }}
            .voice-orb.listening {{
                animation: pulse-orb 1.4s infinite ease-in-out;
                background: radial-gradient(circle at 35% 35%, #34d399 0%, #10b981 55%, #065f46 100%);
                box-shadow: 0 0 70px rgba(16, 185, 129, 0.7);
            }}
            .voice-orb.thinking {{
                animation: pulse-orb 0.9s infinite ease-in-out;
                background: radial-gradient(circle at 35% 35%, #60a5fa 0%, #0b496b 60%, #082f49 100%);
                box-shadow: 0 0 60px rgba(11, 73, 107, 0.6);
            }}
            .voice-orb.speaking {{
                animation: pulse-orb 1.2s infinite ease-in-out;
                background: radial-gradient(circle at 35% 35%, #38bdf8 0%, #0284c7 55%, #075985 100%);
                box-shadow: 0 0 70px rgba(2, 132, 199, 0.7);
            }}
            .pulse-ring {{
                position: absolute;
                width: 175px;
                height: 175px;
                border-radius: 50%;
                border: 2px solid rgba(24, 138, 114, 0.45);
                animation: ripple 2.8s infinite cubic-bezier(0.2, 0.8, 0.2, 1);
                z-index: 1;
            }}
            .ring-2 {{ animation-delay: 0.9s; }}
            .ring-3 {{ animation-delay: 1.8s; }}
            .status-pill {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background: rgba(24, 138, 114, 0.16);
                color: #34d399;
                border: 1px solid rgba(52, 211, 153, 0.35);
                border-radius: 20px;
                padding: 6px 18px;
                font-size: 0.84rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                margin-top: 6px;
            }}
            .status-dot {{
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background-color: #34d399;
                box-shadow: 0 0 10px #34d399;
            }}
            .hint-text {{
                font-size: 0.88rem;
                color: {call_sub};
                margin-top: 10px;
                margin-bottom: 16px;
            }}
            .output-container {{
                width: 100%;
                max-width: 840px;
                text-align: left;
                margin-top: 10px;
            }}
            .speech-bubble {{
                background-color: {call_bubble};
                border: 1px solid {call_bubble_border};
                border-radius: 16px;
                padding: 12px 18px;
                color: {call_text};
                font-size: 0.94rem;
                margin-bottom: 12px;
                display: none;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
            }}
            .gpt-proc-card {{
                background: {call_card_bg};
                border: 1px solid {call_border};
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
                display: none;
            }}
            .gpt-proc-header {{
                background: linear-gradient(135deg, #0b496b 0%, #072d42 100%);
                color: #ffffff;
                padding: 11px 18px;
                font-size: 0.8rem;
                font-weight: 700;
                letter-spacing: 0.07em;
                text-transform: uppercase;
                display: flex;
                align-items: center;
                gap: 8px;
                border-bottom: 1px solid rgba(24, 138, 114, 0.3);
            }}
            .gpt-proc-body {{
                padding: 16px 20px;
                background: {call_card_bg};
                color: {call_text};
            }}
            .proc-step {{
                display: flex;
                align-items: flex-start;
                gap: 12px;
                margin-bottom: 12px;
                font-size: 0.92rem;
                line-height: 1.5;
                color: {call_text};
            }}
            .step-badge {{
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.8rem;
                font-weight: 700;
                background: {call_bubble};
                color: #34d399;
                border: 1px solid rgba(52, 211, 153, 0.3);
                border-radius: 6px;
                padding: 2px 7px;
                flex-shrink: 0;
            }}
            .gpt-proc-footer {{
                background: {call_footer_bg};
                border-top: 1px solid {call_border};
                padding: 10px 18px;
                font-size: 0.78rem;
                color: {call_sub};
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
        </style>
    </head>
    <body>
        <div class="orb-wrapper" onclick="toggleSpeech()">
            <div class="pulse-ring ring-3"></div>
            <div class="pulse-ring ring-2"></div>
            <div class="pulse-ring"></div>
            <div class="voice-orb" id="orb"></div>
        </div>

        <div class="status-pill" id="statusPill">
            <span class="status-dot"></span>
            <span id="statusLabel">CLICK ORB TO SPEAK</span>
        </div>

        <p class="hint-text" id="hintLabel">Tap the glowing orb to speak — auto-answers when you pause.</p>

        <div class="output-container">
            <div class="speech-bubble" id="userBubble"></div>
            <div class="safety-banner" id="safetyAlert"></div>
            <div class="gpt-proc-card" id="card">
                <div class="gpt-proc-header">⚙️ RECOMMENDED MAINTENANCE PROCEDURE</div>
                <div class="gpt-proc-body" id="cardBody"></div>
                <div class="gpt-proc-footer">
                    <span id="cardCitation"></span>
                    <span id="cardCount" style="font-family: 'IBM Plex Mono', monospace; font-weight: 700; color: #0b496b;"></span>
                </div>
            </div>
        </div>

        <script>
            let isListening = false;
            let recognition = null;
            let currentAudio = null;

            const orb = document.getElementById('orb');
            const statusLabel = document.getElementById('statusLabel');
            const hintLabel = document.getElementById('hintLabel');
            const userBubble = document.getElementById('userBubble');
            const safetyAlert = document.getElementById('safetyAlert');
            const card = document.getElementById('card');
            const cardBody = document.getElementById('cardBody');
            const cardCitation = document.getElementById('cardCitation');
            const cardCount = document.getElementById('cardCount');
            const targetDevice = "{st.session_state.selected_device}";

            const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (SpeechRec) {{
                recognition = new SpeechRec();
                recognition.continuous = false;
                recognition.interimResults = true;
                recognition.lang = 'en-US';

                recognition.onstart = function() {{
                    isListening = true;
                    statusLabel.innerText = 'LISTENING...';
                    hintLabel.innerText = 'Speak your question naturally — stops automatically when you pause';
                    orb.className = 'voice-orb listening';
                }};

                recognition.onresult = function(event) {{
                    let text = '';
                    for (let i = event.resultIndex; i < event.results.length; ++i) {{
                        text += event.results[i][0].transcript;
                    }}
                    userBubble.style.display = 'block';
                    userBubble.innerHTML = '🗣️ "' + text + '"';
                }};

                recognition.onspeechend = function() {{
                    statusLabel.innerText = 'THINKING...';
                    orb.className = 'voice-orb thinking';
                    hintLabel.innerText = 'Consulting OEM manual for ' + targetDevice + '...';
                }};

                recognition.onend = function() {{
                    isListening = false;
                    const query = userBubble.innerText.replace('🗣️ "', '').replace('"', '').trim();
                    if (query && query.length > 2) {{
                        executeRAG(query);
                    }} else {{
                        statusLabel.innerText = 'CLICK ORB TO SPEAK';
                        orb.className = 'voice-orb';
                        hintLabel.innerText = 'Tap the glowing orb to speak';
                    }}
                }};

                recognition.onerror = function(e) {{
                    statusLabel.innerText = 'CLICK ORB TO SPEAK';
                    orb.className = 'voice-orb';
                    hintLabel.innerText = 'Tap the glowing orb to speak';
                }};
            }}

            function toggleSpeech() {{
                if (currentAudio && !currentAudio.paused) {{
                    currentAudio.pause();
                }}
                if (isListening) {{
                    recognition.stop();
                }} else {{
                    userBubble.style.display = 'none';
                    userBubble.innerText = '';
                    recognition.start();
                }}
            }}

            async function executeRAG(query) {{
                statusLabel.innerText = 'DIAGNOSING...';
                orb.className = 'voice-orb thinking';

                try {{
                    const resp = await fetch('http://localhost:8000/api/chat', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            query: query,
                            device: targetDevice,
                            session_id: 'fixora_voice_stream'
                        }})
                    }});

                    if (!resp.ok) {{
                        const err = await resp.text();
                        console.error('Server error:', resp.status, err);
                        statusLabel.innerText = 'SERVER BUSY / RETRY';
                        orb.className = 'voice-orb';
                        return;
                    }}

                    const data = await resp.json();

                    if (data.hazard_warning) {{
                        safetyAlert.style.display = 'block';
                        safetyAlert.innerHTML = '<strong>⚠️ CRITICAL SAFETY MANDATE:</strong> ' + data.hazard_warning;
                    }} else {{
                        safetyAlert.style.display = 'none';
                    }}

                    if (data.steps && data.steps.length > 0) {{
                        let h = '';
                        data.steps.forEach((s, idx) => {{
                            const num = String(idx + 1).padStart(2, '0');
                            h += `<div class="proc-step"><span class="step-badge">${{num}}</span><span>${{s}}</span></div>`;
                        }});
                        cardBody.innerHTML = h;
                        cardCitation.innerText = '📄 ' + (data.citations[0] || targetDevice + ' Service Manual');
                        cardCount.innerText = data.steps.length + ' STEPS';
                        card.style.display = 'block';
                    }}

                    if (data.audio_base64) {{
                        statusLabel.innerText = 'SPEAKING...';
                        orb.className = 'voice-orb speaking';
                        hintLabel.innerText = 'Assistant is speaking — listening automatically after...';

                        if (currentAudio) {{ currentAudio.pause(); }}
                        currentAudio = new Audio("data:audio/wav;base64," + data.audio_base64);
                        currentAudio.play().catch(err => {{
                            console.warn("Audio autoplay blocked:", err);
                        }});

                        currentAudio.onended = function() {{
                            statusLabel.innerText = 'LISTENING...';
                            orb.className = 'voice-orb listening';
                            hintLabel.innerText = 'Speak now — assistant is listening';
                            setTimeout(() => {{
                                if (!isListening) {{
                                    recognition.start();
                                }}
                            }}, 500);
                        }};
                    }} else {{
                        statusLabel.innerText = 'CLICK ORB TO SPEAK';
                        orb.className = 'voice-orb';
                    }}
                }} catch (e) {{
                    console.error(e);
                    statusLabel.innerText = 'CLICK ORB TO SPEAK';
                    orb.className = 'voice-orb';
                }}
            }}
        </script>
    </body>
    </html>
    """

    components.html(live_call_html, height=720, scrolling=False)

# ─────────────────────────────────────────────────────────────
# VIEW B: CHATBOT MODE (3-ZONE WORKSPACE)
# ─────────────────────────────────────────────────────────────
else:
    if not st.session_state.chat_history:
        first_name = st.session_state.technician_name.split()[0] if st.session_state.technician_name else "Technician"
        hero_html = (
            f'<div style="text-align: center; margin-top: 36px; margin-bottom: 30px;">'
            f'{get_fixora_svg(56)}'
            f'<h1 style="color: #0b496b; font-size: 1.9rem; font-weight: 700; margin-top: 14px; margin-bottom: 6px;">'
            f'What\'s on your mind today, {first_name}?'
            f'</h1>'
            f'<p style="color: #64748b; font-size: 0.95rem;">'
            f'Grounded assistant for <strong>{st.session_state.selected_device}</strong>'
            f'</p>'
            f'</div>'
        )
        st.markdown(hero_html, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🖥️ Screen Blank on Startup\n\nPower LED is on but display does not show waveforms or interface.", use_container_width=True):
                st.session_state.active_prompt = "Power is on, but the monitor screen is completely blank. What are all root causes and replacement steps?"
                st.rerun()

            if st.button("⚠️ High Voltage Safety Protocol\n\nMandatory safety checks before servicing power supply or chassis.", use_container_width=True):
                st.session_state.active_prompt = "What are the high voltage safety precautions and grounding procedures before opening the unit?"
                st.rerun()

        with c2:
            if st.button("🔊 Alarm Speaker & Buzzer Malfunction\n\nAudible alarm tone fails to sound during critical patient alerts.", use_container_width=True):
                st.session_state.active_prompt = "Audible alarms do not sound. What components should be checked or replaced?"
                st.rerun()

            if st.button("🔄 NIBP Pressure Pump & Cuff Failure\n\nNIBP cuff does not inflate or throws pressure leakage errors.", use_container_width=True):
                st.session_state.active_prompt = "The NIBP cuff does not inflate. What are the corrective actions from the manual?"
                st.rerun()

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble-container"><div class="user-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
        else:
            asst_head = (
                f'<div style="display: flex; align-items: center; gap: 8px; margin-top: 14px; margin-bottom: 6px;">'
                f'{get_fixora_svg(20)}'
                f'<span style="font-size: 0.85rem; font-weight: 700; color: #0b496b;">Fixora Telemetry</span>'
                f'</div>'
            )
            st.markdown(asst_head, unsafe_allow_html=True)
            render_procedure_card(msg["content"], msg.get("citations", []), st.session_state.selected_device)

    prompt_to_run = None
    if "active_prompt" in st.session_state and st.session_state.active_prompt:
        prompt_to_run = st.session_state.active_prompt
        del st.session_state.active_prompt

    user_typed = st.chat_input("Ask anything about faults, codes, procedures... ↑")
    if user_typed:
        prompt_to_run = user_typed

    if prompt_to_run:
        st.session_state.chat_history.append({"role": "user", "content": prompt_to_run})
        st.markdown(f'<div class="user-bubble-container"><div class="user-bubble">{prompt_to_run}</div></div>', unsafe_allow_html=True)

        asst_head = (
            f'<div style="display: flex; align-items: center; gap: 8px; margin-top: 14px; margin-bottom: 6px;">'
            f'{get_fixora_svg(20)}'
            f'<span style="font-size: 0.85rem; font-weight: 700; color: #0b496b;">Fixora Telemetry</span>'
            f'</div>'
        )
        st.markdown(asst_head, unsafe_allow_html=True)

        with st.spinner("Accessing verified technical manuals..."):
            search_data = retrieve_solution(device_name=st.session_state.selected_device, technician_input=prompt_to_run, top_k=5)
            rag_out = llm_engine.generate_response(
                device_name=st.session_state.selected_device,
                technician_query=prompt_to_run,
                retrieved_chunks=search_data["top_chunks"],
                stream=False
            )
            response_text = rag_out["spoken_text"]
            citations = rag_out.get("citations", [])

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response_text,
            "citations": citations
        })
        save_session(st.session_state.session_id, st.session_state.selected_device, st.session_state.chat_history)
        st.rerun()
