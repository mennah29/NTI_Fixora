import json
import sys
import ollama
from typing import List, Dict, Any, Tuple

sys.stdout.reconfigure(encoding='utf-8')

class MaintenanceLLMEngine:
    def __init__(self, model_name: str = "qwen2.5:7b", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature

    def _build_system_prompt(self, device_name: str = "Biomedical Equipment") -> str:
        return (
            f"You are an expert Biomedical Field Service Voice Assistant specifically for '{device_name}'.\n\n"
            "CRITICAL CONSTRAINTS & OPERATIONAL RULES:\n"
            f"1. Zero Cross-Equipment Bleeding: You are STRICTLY FORBIDDEN from mentioning or referencing any equipment other than '{device_name}'.\n"
            f"2. Strict Manual Isolation: Rely EXCLUSIVELY on the authorized service manual context for '{device_name}'.\n"
            "3. Exhaustive Troubleshooting & Escalation:\n"
            "   - Always list ALL possible root causes and their sequential corrective actions from the manual.\n"
            "   - If hardware replacement is required (e.g. replacing LCD Display Assembly, Inverter, Backlight Board, or Main Board Module), explicitly state each replacement action along with the relevant removal section/page.\n"
            "   - Do not omit secondary solutions if the initial inspection or cable reseating fails.\n"
            "4. Safety First: If any hazard (high voltage, heat, radiation) is flagged, your FIRST spoken sentence MUST be a clear safety warning.\n"
            "5. Format: Structure diagnostic steps as clear, numbered, progressive actions (1, 2, 3...).\n"
            "6. Grounding: Never fabricate part numbers or specifications not present in the manual context.\n"
            "7. Citations: Always include the exact manual name and page numbers."
        )

    def _format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[str, bool, List[str]]:
        context_blocks = []
        has_hazard = False
        citations = []

        for idx, chunk in enumerate(retrieved_chunks, 1):
            manual = chunk.get("manual", "Unknown Manual")
            page = chunk.get("page", "N/A")
            is_hazard = str(chunk.get("has_safety_hazard", False)).lower() in ["true", "1", "yes"]
            content = chunk.get("content", "").strip()

            if is_hazard:
                has_hazard = True

            citation_str = f"{manual} (Page {page})"
            if citation_str not in citations:
                citations.append(citation_str)

            block = (
                f"--- [Excerpt {idx} | {citation_str}] ---\n"
                f"Hazard Flag: {'HIGH RISK' if is_hazard else 'NORMAL'}\n"
                f"Text:\n{content}\n"
            )
            context_blocks.append(block)

        return "\n".join(context_blocks), has_hazard, citations

    def generate_response(
        self, 
        device_name: str, 
        technician_query: str, 
        retrieved_chunks: List[Dict[str, Any]],
        stream: bool = False
    ):
        """
        Synthesizes manual context into structured output for UI and Audio.
        If stream=True, returns (generator, metadata) for real-time typing.
        """
        if not retrieved_chunks:
            msg = f"I could not locate troubleshooting procedures for {device_name} matching that issue in the authorized service manual."
            if stream:
                def empty_gen():
                    yield msg
                return empty_gen(), {
                    "has_safety_hazard": False,
                    "citations": [],
                    "device": device_name
                }
            return {
                "spoken_text": msg,
                "has_safety_hazard": False,
                "citations": [],
                "device": device_name
            }

        context_text, has_hazard, citations = self._format_context(retrieved_chunks)

        user_prompt = (
            f"TARGET EQUIPMENT: {device_name}\n"
            f"TECHNICIAN QUERY: {technician_query}\n\n"
            f"REPRESENTATIVE MANUAL CONTEXT:\n{context_text}\n\n"
            "Provide the diagnostic response now following the system rules:"
        )

        metadata = {
            "has_safety_hazard": has_hazard,
            "citations": citations,
            "device": device_name
        }

        # Call local Ollama Qwen 2.5 with optimized context window (fits better in VRAM)
        if stream:
            response_stream = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._build_system_prompt(device_name)},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": self.temperature,
                    "top_p": 0.9,
                    "num_ctx": 2048,   # Optimized for 4GB VRAM
                    "num_predict": 512 # Cap response length for fast generation
                },
                stream=True
            )
            def token_generator():
                for chunk in response_stream:
                    yield chunk["message"]["content"]
            return token_generator(), metadata

        # Non-streaming for Audio TTS
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self._build_system_prompt(device_name)},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": self.temperature,
                "top_p": 0.9,
                "num_ctx": 2048,
                "num_predict": 512
            }
        )

        raw_output = response["message"]["content"]

        return {
            "spoken_text": raw_output,
            "has_safety_hazard": has_hazard,
            "citations": citations,
            "device": device_name
        }

# --- Quick Unit Test Execution ---
if __name__ == "__main__":
    engine = MaintenanceLLMEngine(model_name="qwen2.5:7b")

    # Simulated Context from Phase 2 (FAISS)
    mock_retrieved_data = [
        {
            "manual": "Philips_V24_Service_Manual.pdf",
            "page": 42,
            "has_safety_hazard": True,
            "content": "ERROR E37: Power Supply Overheating.\n"
                       "WARNING: Hazardous voltages present on the primary power board. Disconnect mains power and wait 5 minutes before opening chassis.\n"
                       "Action Steps:\n"
                       "1. Verify rear cooling fan is running.\n"
                       "2. Clean air inlet filters.\n"
                       "3. If fan RPM is below 1800, replace fan assembly Part #M1205-60010."
        }
    ]

    result = engine.generate_response(
        device_name="Philips V24 Monitor",
        technician_query="I have error code E37 and the machine is extremely hot.",
        retrieved_chunks=mock_retrieved_data
    )

    print("=" * 60)
    print("📋 PHASE 3 STRUCTURED OUTPUT PAYLOAD:")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
