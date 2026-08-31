import os
import ssl
import sys
import torch
import numpy as np
import soundfile as sf
import asyncio
import edge_tts

# ── SSL bypass for corporate/restricted networks ──────────────────────────────
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
ssl._create_default_https_context = ssl._create_unverified_context

sys.stdout.reconfigure(encoding='utf-8')

from faster_whisper import WhisperModel
from kokoro import KPipeline

class ProductionAudioEngine:
    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.compute_type = "float16" if self.device == "cuda" else "int8"

        # Initialize Whisper with intelligent cache detection
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
            print("🎙️ Loading cached Whisper STT model...")
            self.stt_model = WhisperModel(
                "base",
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=4
            )
            print("✅ Whisper STT engine ready.")

        # Initialize Kokoro-82M Neural TTS
        print("🔊 Initializing Kokoro-82M TTS Pipeline...")
        try:
            self.tts_pipeline = KPipeline(lang_code='a')  # 'a' = American English
            self.kokoro_ready = True
            print("✅ Kokoro-82M TTS initialized successfully.")
        except Exception as e:
            print(f"⚠️ Kokoro init warning: {e}. Falling back to Edge-TTS.")
            self.kokoro_ready = False

    def transcribe(self, audio_file_path: str) -> str:
        """
        Sub-500ms speech-to-text with Whisper.
        """
        segments, _ = self.stt_model.transcribe(
            audio_file_path,
            beam_size=1,            # 1 for maximum speed in live calls
            best_of=1,
            temperature=0.0,
            vad_filter=True         # Strips background silence automatically
        )
        text = " ".join([seg.text for seg in segments]).strip()
        return text

    def synthesize_kokoro(self, text: str, output_path: str = "output_response.wav", voice: str = "af_heart") -> str:
        """
        Sub-200ms neural speech synthesis on CPU/GPU with Kokoro-82M.
        Concatenates all audio chunks for complete procedural instructions.
        """
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

    def synthesize(self, text: str, output_path: str = "output_response.wav") -> str:
        """Unified TTS with auto fallback."""
        if self.kokoro_ready:
            try:
                return self.synthesize_kokoro(text, output_path)
            except Exception as e:
                print(f"Kokoro synthesis error: {e}. Trying Edge-TTS fallback...")
        
        # Fallback to Edge-TTS if needed
        edge_out = output_path.replace(".wav", ".mp3")
        try:
            asyncio.run(self._synthesize_edge_async(text, edge_out))
            return edge_out
        except Exception as e:
            print(f"Edge-TTS synthesis error: {e}")
            return output_path

if __name__ == "__main__":
    print("Testing ProductionAudioEngine initialization...")
    engine = ProductionAudioEngine()
    test_text = "WARNING: Hazardous voltages present on the primary power board. Please disconnect power before opening chassis."
    out_file = engine.synthesize(test_text, "test_speech.wav")
    print(f"Generated test audio at: {out_file}")
