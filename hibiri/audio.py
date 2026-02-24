# hibiri/audio.py
import numpy as np
import webrtcvad
import librosa
import time
from collections import deque
import logging

# Initialize a logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Basic config, should be configurable later

# Placeholder for MIN_SPEECH_DURATION, will be configurable
MIN_SPEECH_DURATION = 0.3 # seconds

class AudioProcessor:
    def __init__(self, sample_rate: int = 16000, aggressiveness: int = 3):
        self.sample_rate = sample_rate
        # VAD operates on 16kHz, 16-bit PCM mono samples.
        # It expects frames of 10, 20 or 30 ms.
        self.vad = webrtcvad.Vad(aggressiveness) # Aggressiveness level 0-3
        self.buffer = bytearray()
        self.speech_timer = None
        self.frame_duration_ms = 20 # VAD frame duration
        self.bytes_per_frame = (self.sample_rate * self.frame_duration_ms // 1000) * 2 # 2 bytes for 16-bit samples

        # Deque to store incoming raw audio bytes before processing
        self.incoming_audio_queue = deque()

    def resample(self, audio_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
        """Convert between telephone and model sample rates"""
        # Ensure audio_bytes is properly padded or handled for exact 16-bit samples
        if not audio_bytes:
            return b''

        # If it's not a multiple of 2, it's malformed PCM 16-bit
        if len(audio_bytes) % 2 != 0:
            logger.warning("Received audio bytes with odd length, padding with zero.")
            audio_bytes += b'\x00'

        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        if from_rate == to_rate:
            return audio_bytes

        # librosa.resample expects float32
        resampled_audio = librosa.resample(
            audio.astype(np.float32),
            orig_sr=from_rate,
            target_sr=to_rate
        )
        # Convert back to 16-bit PCM
        return (resampled_audio * 32768).astype(np.int16).tobytes()

    def detect_speech(self, chunk: bytes) -> bool:
        """Voice activity detection for a single chunk."""
        # webrtcvad requires 16kHz, 16-bit mono, and specific frame sizes (10, 20, or 30 ms)
        # Assuming `chunk` is already 16kHz 16-bit mono PCM.
        # It also expects the length of `chunk` to correspond to exactly 10, 20 or 30 ms of audio.
        if len(chunk) != self.bytes_per_frame:
            logger.warning(
                f"VAD chunk length {len(chunk)} does not match expected {self.bytes_per_frame} "
                f"for {self.frame_duration_ms}ms at {self.sample_rate}Hz. Skipping VAD for this chunk."
            )
            return False # Cannot reliably run VAD if chunk size is incorrect

        try:
            return self.vad.is_speech(chunk, self.sample_rate)
        except Exception as e:
            logger.error(f"Error during VAD.is_speech: {e}")
            return False

    async def process_stream(self, audio_bytes_16k: bytes) -> Optional[bytes]:
        """
        Accumulate speech until silence.
        This method is designed to be called repeatedly with small audio chunks.
        It manages an internal buffer and returns a complete utterance when silence is detected
        after a period of speech.
        """
        self.incoming_audio_queue.append(audio_bytes_16k)

        # Process the queue in fixed VAD frame durations
        while len(self.buffer) + len(self.incoming_audio_queue[0]) >= self.bytes_per_frame if self.incoming_audio_queue else False:
            if not self.incoming_audio_queue:
                break

            # Fill buffer until it's at least one frame long
            while len(self.buffer) < self.bytes_per_frame and self.incoming_audio_queue:
                self.buffer.extend(self.incoming_audio_queue.popleft())

            if len(self.buffer) < self.bytes_per_frame: # Not enough for a full frame yet
                break

            # Extract a VAD frame
            vad_frame = self.buffer[:self.bytes_per_frame]
            self.buffer = self.buffer[self.bytes_per_frame:] # Remove processed frame

            is_speech = self.detect_speech(vad_frame)

            if is_speech:
                # If speech is detected, extend the current utterance
                if self.speech_timer is None: # Start of a new speech segment
                    self.speech_timer = time.time()
                self._current_utterance.extend(vad_frame) # Accumulate audio for utterance
            elif self.speech_timer is not None:
                # Silence detected after speech, and speech duration was significant
                if time.time() - self.speech_timer >= MIN_SPEECH_DURATION:
                    utterance = self._current_utterance
                    self._current_utterance = bytearray() # Reset for next utterance
                    self.speech_timer = None
                    return bytes(utterance) # Return the complete utterance
                else: # Short silence, reset timer and continue accumulating
                    self.speech_timer = None
                    self._current_utterance.extend(vad_frame) # Continue accumulating for potential longer speech
            else:
                # Continuous silence, or no speech detected yet
                # No speech_timer, so just continue accumulating in case speech starts
                self._current_utterance.extend(vad_frame)


        return None

    # This buffer will store the current utterance being accumulated
    _current_utterance = bytearray()

    def reset_utterance_buffer(self):
        self._current_utterance = bytearray()
        self.speech_timer = None
        self.buffer = bytearray()
        self.incoming_audio_queue.clear()
