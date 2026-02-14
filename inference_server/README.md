# Inference Server

The inference server hosts the Hibiri (formerly Hibiki-Zero) model as a FastAPI WebSocket service. It is designed to integrate with telephony gateways like Twilio.

## Features
- FastAPI WebSocket endpoint at `/stream`.
- Real-time audio resampling (8kHz mu-law ↔ 24kHz PCM).
- Integrated Voice Activity Detection (VAD).
- Streaming inference using the Hibiri model.

## Running the server
To run the inference server:
```bash
uv run python -m inference_server.main
```
or
```bash
uvicorn inference_server.main:app --host 0.0.0.0 --port 8000
```