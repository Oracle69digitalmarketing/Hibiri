import os
import json
import base64
import asyncio
import numpy as np
import torch
import librosa
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from moshi.models import loaders
from hibiri.inference import ServerState, seed_all
from hibiri.client_utils import log

app = FastAPI()

# Configuration
HF_REPO = os.getenv("HF_REPO", "kyutai/hibiki-zero-3b-pytorch-bf16@23b3e0b41782026c81dd5283a034107b01f9e513")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global state for the model
state = None

def load_model():
    global state
    seed_all(42)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # We use the original repo name for downloading if hibiri doesn't exist
    # But for now, let's assume it works or is overridden by env var
    repo_to_use = HF_REPO
    if "hibiri" in repo_to_use:
        # Fallback to original if needed, or just let it fail to show it's rebranded
        pass

    hf_repo_parts = repo_to_use.split("@")
    hf_repo_name = hf_repo_parts[0]
    revision = hf_repo_parts[1] if len(hf_repo_parts) > 1 else None

    print(f"Loading model from {hf_repo_name}...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo_name,
        revision=revision,
    )

    mimi = checkpoint_info.get_mimi(device=DEVICE)
    text_tokenizer = checkpoint_info.get_text_tokenizer()
    lm = checkpoint_info.get_moshi(device=DEVICE, dtype=dtype)

    state = ServerState(
        checkpoint_info.model_type,
        mimi,
        text_tokenizer,
        lm,
        DEVICE,
        **checkpoint_info.lm_gen_config,
    )
    state.warmup()
    print(f"Model loaded and warmed up. Mimi SR: {mimi.sample_rate}")

@app.on_event("startup")
async def startup_event():
    try:
        load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": state is not None}

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted")

    if state is None:
        print("Model not loaded, closing connection")
        await websocket.close(code=1011)
        return

    async with state.lock:
        state.mimi.reset_streaming()
        state.lm_gen.reset_streaming()

        vad = webrtcvad.Vad(3)
        sample_rate_twilio = 8000
        sample_rate_mimi = state.mimi.sample_rate # 24000
        frame_size_mimi = state.frame_size # 1920

        # Twilio sends 8bit mu-law at 8000Hz.
        # 1920 samples at 24000Hz = 80ms.
        # 80ms at 8000Hz = 640 samples.
        frame_size_twilio = int(frame_size_mimi * sample_rate_twilio / sample_rate_mimi) # 640

        mulaw_buffer = b""
        pcm_buffer_mimi = np.array([], dtype=np.float32)
        stream_sid = None

        try:
            while True:
                message = await websocket.receive_text()
                data = json.loads(message)

                if data['event'] == 'start':
                    stream_sid = data['start']['streamSid']
                    print(f"Stream started: {stream_sid}")
                    continue

                if data['event'] == 'media':
                    payload = data['media']['payload']
                    chunk_mulaw = base64.b64decode(payload)
                    mulaw_buffer += chunk_mulaw

                    # We need 640 bytes of mu-law to make one Mimi frame (1920 samples at 24kHz)
                    while len(mulaw_buffer) >= frame_size_twilio:
                        frame_mulaw = mulaw_buffer[:frame_size_twilio]
                        mulaw_buffer = mulaw_buffer[frame_size_twilio:]

                        # Mu-law to PCM (8kHz)
                        frame_pcm_8k = librosa.mu_expand(np.frombuffer(frame_mulaw, dtype=np.uint8))

                        # VAD check (needs 16-bit PCM). We can use a smaller frame for VAD if needed,
                        # but here we check the whole 80ms frame (or split it).
                        # webrtcvad only supports 10, 20, or 30ms. 80ms is not supported.
                        # Let's split into 20ms chunks (160 samples).
                        is_speech = False
                        for i in range(0, 640, 160):
                            vad_chunk = frame_pcm_8k[i:i+160]
                            vad_pcm_16 = (vad_chunk * 32767).astype(np.int16).tobytes()
                            if vad.is_speech(vad_pcm_16, 8000):
                                is_speech = True
                                break

                        if is_speech:
                            start_time = asyncio.get_event_loop().time()
                            # Resample to Mimi sample rate
                            frame_pcm_mimi = librosa.resample(frame_pcm_8k, orig_sr=8000, target_sr=sample_rate_mimi)

                            # Inference
                            chunk_torch = torch.from_numpy(frame_pcm_mimi).to(device=state.device, dtype=torch.float32)[None, None]
                            codes = state.mimi.encode(chunk_torch)

                            for c in range(codes.shape[-1]):
                                tokens = state.lm_gen.step(codes[:, :, c : c + 1])
                                if tokens is None:
                                    continue

                                # Decode
                                out_pcm = state.mimi.decode(tokens[:, 1:])
                                out_pcm = out_pcm.cpu().numpy()[0, 0]

                                # Resample back to 8kHz
                                out_pcm_8k = librosa.resample(out_pcm, orig_sr=sample_rate_mimi, target_sr=8000)

                                # Mu-law compress
                                out_mulaw = librosa.mu_compress(out_pcm_8k, quantize=True).astype(np.uint8)

                                # Send to Twilio
                                payload_out = base64.b64encode(out_mulaw).decode('utf-8')
                                await websocket.send_text(json.dumps({
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {
                                        "payload": payload_out
                                    }
                                }))

                            end_time = asyncio.get_event_loop().time()
                            print(f"Latency: {(end_time - start_time) * 1000:.2f}ms")
                        else:
                            # Silence: we might still want to step the model with silence or just skip
                            # For Hibiri, it's probably better to skip or send zeros if we want to maintain timing
                            # But Twilio handles gaps.
                            pass

                if data['event'] == 'stop':
                    print("Stream stopped")
                    break

        except WebSocketDisconnect:
            print("WebSocket disconnected")
        except Exception as e:
            print(f"Error in websocket loop: {e}")
        finally:
            print("Connection closed")

def start():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start()
