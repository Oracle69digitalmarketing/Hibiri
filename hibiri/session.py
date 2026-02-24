# hibiri/session.py
import asyncio
import time
import logging
import torch
import numpy as np # For potential audio manipulation

from typing import Optional

from .audio import AudioProcessor
from .model import HibiriTranslator # Assuming HibiriTranslator is a global/singleton

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global translator instance, loaded once
# This is a simplification. In a real application, this might be managed by
# a dependency injection system or an application factory.
_global_translator: Optional[HibiriTranslator] = None

def get_global_translator() -> HibiriTranslator:
    global _global_translator
    if _global_translator is None:
        # Default parameters for loading the model, adjust as necessary
        _global_translator = HibiriTranslator(device="cuda", bf16=True)
        logger.info("HibiriTranslator model loaded globally.")
    return _global_translator

class CallSession:
    def __init__(self, call_sid: str, sample_rate: int = 16000, telephony_sample_rate: int = 8000):
        self.call_sid = call_sid
        self.speaker_embedding: Optional[torch.Tensor] = None
        self.source_lang: Optional[str] = None
        self.target_lang: str = "en" # Default target language
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        self.translator = get_global_translator()
        self.created_at = time.time()
        self.telephony_sample_rate = telephony_sample_rate
        self.model_sample_rate = sample_rate # Model expects 16kHz
        self.enrollment_buffer = bytearray()
        self.enrollment_duration_s = 5 # As per documentation
        self.is_enrolled = False

        logger.info(f"Call session {self.call_sid} initiated.")

    async def _detect_language(self, audio_bytes: bytes) -> str:
        """
        Placeholder for language detection.
        In a real scenario, this would involve an actual language detection model.
        For now, it returns a hardcoded default or a simple heuristic.
        """
        # Example: if audio_bytes is long enough, assume a language
        # This is highly rudimentary and needs replacement with a proper model
        if len(audio_bytes) > (self.telephony_sample_rate * 2 * 2): # >2 seconds of audio
            # A more robust solution would integrate a dedicated language identification model.
            # For now, let's just return 'es' as a common example.
            logger.warning(f"Language detection placeholder used for call {self.call_sid}. Returning 'es'.")
            return "es"
        return "es" # Default if not enough audio

    async def enroll_speaker(self, audio_chunk_8k: bytes) -> bool:
        """
        Accumulate audio for speaker enrollment and extract embedding.
        Returns True if enrollment is complete, False otherwise.
        """
        if self.is_enrolled:
            return True

        self.enrollment_buffer.extend(audio_chunk_8k)
        
        # Check if enough audio has been collected for enrollment
        # Assuming 16-bit PCM, 2 bytes per sample
        required_bytes = self.telephony_sample_rate * self.enrollment_duration_s * 2

        if len(self.enrollment_buffer) >= required_bytes:
            logger.info(f"Enrollment audio collected for call {self.call_sid}.")
            full_audio_8k = bytes(self.enrollment_buffer)
            self.enrollment_buffer.clear() # Clear buffer after use

            # Resample to model's sample rate before potentially extracting embedding
            full_audio_16k = self.audio_processor.resample(
                full_audio_8k, self.telephony_sample_rate, self.model_sample_rate
            )
            
            # Speaker embedding extraction (currently a NotImplementedError in HibiriTranslator)
            # For now, we will bypass the actual embedding and just proceed with language detection
            # as the current `moshi` model doesn't explicitly expose `encode_speaker` for arbitrary audio.
            # The voice preservation is handled by the model's direct processing of input audio codes.
            # self.speaker_embedding = await self.translator.extract_speaker_embedding(full_audio_16k, self.model_sample_rate)
            logger.warning(f"Speaker embedding extraction is currently a placeholder for call {self.call_sid}.")
            self.speaker_embedding = torch.zeros(1, 1, 1).to(self.translator.device) # Placeholder tensor

            # Auto-detect language
            self.source_lang = await self._detect_language(full_audio_8k)
            self.is_enrolled = True
            logger.info(f"Call {self.call_sid} enrolled. Detected language: {self.source_lang}")
            return True
        return False

    async def process_chunk(self, audio_bytes_8k: bytes) -> Optional[bytes]:
        """
        Main processing pipeline for incoming audio from the telephony provider.
        Assumes input is 8kHz, 16-bit PCM.
        """
        if not self.is_enrolled:
            logger.warning(f"Call {self.call_sid} not yet enrolled. Skipping chunk processing.")
            return None

        # Resample from 8kHz (telephony) to 16kHz (model)
        audio_16k = self.audio_processor.resample(audio_bytes_8k, self.telephony_sample_rate, self.model_sample_rate)

        # Process stream for VAD and utterance accumulation
        utterance_16k_bytes = await self.audio_processor.process_stream(audio_16k)

        if utterance_16k_bytes:
            logger.info(f"Utterance detected for call {self.call_sid}. Translating...")
            
            # Convert bytes to torch.Tensor for translation
            utterance_16k_np = np.frombuffer(utterance_16k_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            utterance_16k_tensor = torch.from_numpy(utterance_16k_np).to(self.translator.device)

            # Translate
            translated_audio_tensor = self.translator.translate(
                utterance_16k_tensor,
                self.source_lang, # type: ignore
                self.target_lang,
                self.speaker_embedding # This is a placeholder for now
            )
            
            if translated_audio_tensor.numel() == 0:
                logger.warning(f"Translator returned empty audio for call {self.call_sid}.")
                return None

            # Convert translated audio tensor back to bytes (assuming float32 output from model)
            translated_audio_np = (translated_audio_tensor.cpu().numpy() * 32768).astype(np.int16)
            translated_audio_bytes_16k = translated_audio_np.tobytes()

            # Resample back to 8kHz for telephone
            translated_audio_8k = self.audio_processor.resample(
                translated_audio_bytes_16k, self.model_sample_rate, self.telephony_sample_rate
            )
            logger.info(f"Translation completed and resampled for call {self.call_sid}.")
            return translated_audio_8k
        
        return None

    async def cleanup(self):
        """Release resources and log metrics."""
        duration = time.time() - self.created_at
        logger.info(f"Call {self.call_sid} ended. Duration: {duration:.2f}s")
        # Further cleanup like clearing GPU memory if applicable for per-session resources
        self.audio_processor.reset_utterance_buffer()
