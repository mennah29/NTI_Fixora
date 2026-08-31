import os
import sys
import torch
import numpy as np
import soundfile as sf
import asyncio

sys.stdout.reconfigure(encoding='utf-8')

# ── Optional heavy imports — graceful fallback for cloud deployment ────────────
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    print("⚠️ edge_tts not available.")

try:
    import ssl
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️ faster_whisper not available (cloud mode — STT disabled).")

try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    print("⚠️ kokoro not available (cloud mode — TTS via edge_tts fallback).")


class ProductionAudioEngine:
    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.compute_type = "float16" if self.device == "cuda" else "int8"

        # Initialize Whisper (optional — disabled on cloud)
        self.stt_model = None
        if WHISPER_AVAILABLE:
            try:
                print(f"🎙️ Loading Whisper Turbo on [{self.device.upper()} - {self.compute_type}]...")
                self.stt_model = WhisperModel(
                    "turbo",
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=4,
                    local_files_only=True
                )
                print("✅ Whisper Turbo loaded from local cache.")
            except Exception:
                try:
                    print("🎙️ Loading cached Whisper STT model...")
                    self.stt_model = WhisperModel(
                        "base",
                        device=self.device,
                        compute_type=self.compute_type,
                        cpu_threads=4
                    )
                    print("✅ Whisper STT engine ready.")
                except Exception as e:
                    print(f"⚠️ Whisper unavailable: {e}")

        # Initialize Kokoro-82M Neural TTS (optional)
        self.tts_pipeline = None
        self.kokoro_ready = False
        if KOKORO_AVAILABLE:
            try:
                print("🔊 Initializing Kokoro-82M TTS Pipeline...")
                self.tts_pipeline = KPipeline(lang_code='a')
                self.kokoro_ready = True
                print("✅ Kokoro-82M TTS initialized successfully.")
            except Exception as e:
                print(f"⚠️ Kokoro init warning: {e}. Falling back to Edge-TTS.")

    def transcribe(self, audio_file_path: str) -> str:
        """Speech-to-text with Whisper (local only)."""
        if self.stt_model is None:
            return ""
        segments, _ = self.stt_model.transcribe(
            audio_file_path,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=True
        )
        return " ".join([seg.text for seg in segments]).strip()

    def synthesize_kokoro(self, text: str, output_path: str = "output_response.wav", voice: str = "af_heart") -> str:
        """Kokoro-82M neural TTS (local/GPU only)."""
        if not self.kokoro_ready or self.tts_pipeline is None:
            return self._edge_tts_fallback(text, output_path)
        generator = self.tts_pipeline(text, voice=voice, speed=1.0, split_pattern=r'\n+')
        audio_chunks = []
        for _, _, audio in generator:
            audio_chunks.append(audio)
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            sf.write(output_path, full_audio, 24000)
            return output_path
        return output_path

    async def _synthesize_edge_async(self, text: str, output_path: str):
        communicate = edge_tts.Communicate(text, "en-US-GuyNeural")
        await communicate.save(output_path)

    def _edge_tts_fallback(self, text: str, output_path: str) -> str:
        """Edge-TTS cloud fallback."""
        if not EDGE_TTS_AVAILABLE:
            print("⚠️ No TTS engine available.")
            return output_path
        edge_out = output_path.replace(".wav", ".mp3")
        try:
            asyncio.run(self._synthesize_edge_async(text, edge_out))
            return edge_out
        except Exception as e:
            print(f"Edge-TTS synthesis error: {e}")
            return output_path

    def synthesize(self, text: str, output_path: str = "output_response.wav") -> str:
        """Unified TTS — Kokoro on GPU, Edge-TTS on cloud."""
        if self.kokoro_ready:
            try:
                return self.synthesize_kokoro(text, output_path)
            except Exception as e:
                print(f"Kokoro synthesis error: {e}. Trying Edge-TTS fallback...")
        return self._edge_tts_fallback(text, output_path)


if __name__ == "__main__":
    print("Testing ProductionAudioEngine initialization...")
    engine = ProductionAudioEngine()
    test_text = "WARNING: Hazardous voltages present. Please disconnect power before opening chassis."
    out_file = engine.synthesize(test_text, "test_speech.wav")
    print(f"Generated test audio at: {out_file}")
