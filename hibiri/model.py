# hibiri/model.py
import torch
import sentencepiece
import numpy as np
import random
import math
from moshi.models import LMGen, LMModel, MimiModel, loaders
from moshi.run_inference import get_condition_tensors

def seed_all(seed: int):
    """Seed all relevant random number generators for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

class HibiriTranslator:
    def __init__(self, model_path: str = "kyutai/hibiki-zero", device: str = "cuda", bf16: bool = False):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if bf16 else torch.float16
        self.model_path = model_path
        self.mimi: MimiModel = None
        self.text_tokenizer: sentencepiece.SentencePieceProcessor = None
        self.lm: LMModel = None
        self.lm_gen: LMGen = None
        self.model_type: str = None
        self.lm_gen_config: dict = {}

        self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load the Hibiki-Zero model and its components."""
        hf_repo_parts = model_path.split("@")
        hf_repo_name = hf_repo_parts[0]
        revision = hf_repo_parts[1] if len(hf_repo_parts) > 1 else None

        # Assuming checkpoint_info can be derived from model_path
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
            hf_repo_name,
            revision=revision,
            # Add other necessary parameters as they are defined in the original `run.py`
            # For simplicity, initially assume default loading and expand as needed.
        )
        self.model_type = checkpoint_info.model_type
        self.mimi = checkpoint_info.get_mimi(device=self.device)
        self.text_tokenizer = checkpoint_info.get_text_tokenizer()
        self.lm = checkpoint_info.get_moshi(device=self.device, dtype=self.dtype)
        self.lm_gen_config = checkpoint_info.lm_gen_config

        # Initialize LMGen with a batch_size of 1 for streaming, will be updated per call
        self.lm_gen = self._get_lmgen(self.lm, self.model_type, 1, self.lm_gen_config)
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        # Warmup for immediate use
        self.warmup()

    def _get_lmgen(self, lm: LMModel, model_type: str, batch_size: int, lm_gen_config: dict, cfg_coef: float = 1.0) -> LMGen:
        condition_tensors = get_condition_tensors(
            model_type, lm, batch_size=batch_size, cfg_coef=cfg_coef
        )
        return LMGen(
            lm, cfg_coef=cfg_coef, condition_tensors=condition_tensors, **lm_gen_config
        )

    def warmup(self):
        """Warm up the model to reduce initial latency."""
        frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        with torch.no_grad():
            for _ in range(4): # Similar to inference.py warmup
                chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=self.device)
                codes = self.mimi.encode(chunk)
                for c in range(codes.shape[-1]):
                    tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                    if tokens is None:
                        continue
                    _ = self.mimi.decode(tokens[:, 1:])
        torch.cuda.synchronize()

    def extract_speaker_embedding(self, audio: np.ndarray, sample_rate: int) -> torch.Tensor:
        """Extract speaker embedding from audio."""
        # This functionality is not directly available in the provided snippets.
        # Hibiki-Zero is primarily a text-to-speech model with voice preservation,
        # but the direct method for extracting a *speaker embedding* for *input audio*
        # is not exposed. The current inference flow implies a source audio is encoded
        # then translated.
        # For now, this will be a placeholder or a direct encoding if `mimi.encode` serves this purpose.
        # Based on the documentation, it seems `extract_speaker_embedding` is a distinct step
        # for speaker enrollment.
        # For the current code, the "voice preservation" happens implicitly through the model's
        # ability to generate speech from audio codes.
        # I'll implement a placeholder that just encodes the audio, assuming this is
        # part of the "voice characteristics" until more specific API is found.
        # However, the documentation explicitly states `encode_speaker` from `AutoModelForSpeechSeq2Seq`.
        # This indicates that the `moshi` models might not be directly `AutoModelForSpeechSeq2Seq`.
        # I will leave this as a NotImplementedError for now and address it once the model specifics are clear.
        raise NotImplementedError("Speaker embedding extraction is not yet implemented based on current model types.")

    def translate(self, audio: torch.Tensor, source_lang: str, target_lang: str, speaker_embedding: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, str]:
        """Translate audio with voice preservation."""
        # The current inference process takes audio, encodes it, then generates output.
        # The `speaker_embedding` concept as a separate input to `generate` is from the documentation's
        # `AutoModelForSpeechSeq2Seq` example, not directly visible in the `moshi` framework.
        # The `moshi` inference seems to handle voice preservation by directly encoding the source audio
        # and using that as input to the LM.

        # For the purpose of aligning with the documentation's `translate` signature,
        # I will adapt the `encode_inputs` and `decode_outputs` flow.

        # Assuming `audio` is already at `mimi.sample_rate` and `mono`
        batch_wavs = audio.to(self.device)[None, None, :] # Add batch and channel dimension
        audio_durations = [audio.shape[-1] / self.mimi.sample_rate]

        # Reset streaming state for new translation
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()

        codes, warmup_codes = self._encode_inputs(batch_wavs, self.mimi, self.lm_gen, audio_durations)

        output_text_tokens: list[torch.Tensor] = []
        output_audio_tokens: list[torch.Tensor] = []
        gen_steps: int = codes.shape[-1]

        with torch.no_grad(), self.lm_gen.streaming(batch_wavs.shape[0]):
            # warmup
            for step in range(warmup_codes.shape[-1]):
                _ = self.lm_gen.step(warmup_codes[:, :, step : step + 1])
            # generation
            for step in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, step : step + 1])
                if tokens is None:
                    continue
                output_text_tokens.append(tokens[:, 0, :])
                output_audio_tokens.append(tokens[:, 1:, :])

        if not output_audio_tokens:
            return torch.empty(0, device=self.device), "" # Return empty audio and empty string if no audio generated

        batch_text_tokens: torch.Tensor = torch.concat(output_text_tokens, dim=-1)
        batch_codes: torch.Tensor = torch.concat(output_audio_tokens, dim=-1)

        # _decode_outputs returns a list of (wav, text) tuples.
        # Since we're processing a single audio input, we expect a list with one tuple.
        decoded_outputs = self._decode_outputs(batch_codes, batch_text_tokens, self.mimi, self.text_tokenizer)
        
        if decoded_outputs:
            translated_audio, translated_text = decoded_outputs[0]
            return translated_audio, translated_text
        else:
            # Fallback for unexpected empty decoded_outputs
            return torch.empty(0, device=self.device), ""

    def _add_input_eos(self, codes: torch.Tensor, mimi: MimiModel, audio_durations: list[float]) -> torch.Tensor:
        """Helper from inference.py"""
        other_audio_eos_idx: torch.Tensor = torch.tensor(
            [
                min(math.ceil(duration * mimi.frame_rate), codes.shape[-1] - 1)
                for duration in audio_durations
            ]
        )[:, None, None].to(codes.device)
        codes_like_indexes: torch.Tensor = torch.arange(0, codes.shape[-1])[None, None].to(
            codes.device
        )
        codes_with_input_eos: torch.Tensor = torch.where(
            codes_like_indexes == other_audio_eos_idx,
            torch.full([1], mimi.cardinality, device=codes.device),
            codes,
        )
        return codes_with_input_eos

    def _encode_inputs(self, batch_wavs: torch.Tensor, mimi: MimiModel, lm_gen: LMGen, audio_durations: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
        """Helper from inference.py"""
        frame_size = int(mimi.sample_rate / mimi.frame_rate)
        with torch.no_grad():
            codes: torch.Tensor = mimi.encode(batch_wavs.to(lm_gen.lm_model.device))
            codes_with_input_eos = self._add_input_eos(codes, mimi, audio_durations)
            warmup_wav: torch.Tensor = torch.zeros(
                codes.shape[0],
                1,
                frame_size * lm_gen.max_delay,
                dtype=torch.float32,
                device=codes.device,
            )
            warmup_codes: torch.Tensor = mimi.encode(warmup_wav)
        return codes_with_input_eos, warmup_codes

    def _decode_outputs(self, batch_codes: torch.Tensor, batch_text_tokens: torch.Tensor, mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor) -> list[tuple[torch.Tensor, str]]:
        """Helper from inference.py"""
        with torch.no_grad():
            output_wavs: torch.Tensor = mimi.decode(batch_codes).cpu()

        outputs: list[tuple[torch.Tensor, str]] = []
        for output_idx, wav in enumerate(output_wavs):
            text_tokens: list[int] = batch_text_tokens[output_idx].tolist()
            if text_tokenizer.eos_id() in text_tokens:
                eos_idx: int = text_tokens.index(text_tokenizer.eos_id())
            else:
                eos_idx: int = len(text_tokens) - 1
                while eos_idx > 0 and text_tokens[eos_idx] == text_tokenizer.pad_id():
                    eos_idx -= 1
            text_tokens = [t for t in text_tokens[:eos_idx] if t > text_tokenizer.pad_id()]
            text: str = text_tokenizer.decode(text_tokens)
            # Ensure wav is not empty before slicing
            if wav.shape[-1] > 0:
                wav = wav[:, : int(min(eos_idx, wav.shape[-1] * mimi.frame_rate / mimi.sample_rate) * mimi.sample_rate / mimi.frame_rate)]
            outputs.append((wav, text))
        return outputs

