# hibiri/server.py
import asyncio
import uuid
import logging
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse
import uvicorn

from .session import CallSession
from .model import get_global_translator # Ensure global translator is initialized
from .client_utils import log # Using the existing logger from client_utils

# Initialize global translator upon import
get_global_translator()

app = FastAPI()

# Configure logging
logger = logging.getLogger(__name__)
# Basic config for now, will be enhanced with proper config in run.py or main.py
logging.basicConfig(level=logging.INFO)

# HTML response for root endpoint, for basic testing or documentation
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Hibiri WebSocket Stream</title>
        <style>
            body { font-family: sans-serif; margin: 2em; background-color: #f4f4f4; color: #333; }
            h1 { color: #0056b3; }
            #messages { list-style-type: none; padding: 0; }
            #messages li { background-color: #e9e9e9; margin-bottom: 0.5em; padding: 0.5em; border-radius: 5px; }
            #messages li.error { background-color: #ffcccc; color: #cc0000; }
            #messages li.info { background-color: #ccf; }
            #messages li.translated { background-color: #d4edda; color: #155724; }
        </style>
    </head>
    <body>
        <h1>Hibiri WebSocket Stream</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off" placeholder="Type a message or audio path"/>
            <button>Send Audio</button>
        </form>
        <p>WebSocket Status: <span id="wsStatus">Disconnected</span></p>
        <ul id="messages"></ul>
        <script>
            var ws = null;
            var reconnectInterval = 1000; // milliseconds
            var maxReconnectInterval = 10000; // milliseconds
            var reconnectAttempts = 0;

            function connectWebSocket() {
                var protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
                var host = window.location.host;
                ws = new WebSocket(protocol + "//" + host + "/stream");

                ws.onopen = function(event) {
                    console.log("WebSocket opened:", event);
                    document.getElementById("wsStatus").textContent = "Connected";
                    reconnectAttempts = 0;
                };

                ws.onmessage = function(event) {
                    var messages = document.getElementById("messages");
                    var message = document.createElement('li');
                    var data = event.data;

                    if (data instanceof Blob) {
                        // Assuming Blob is audio, let's play it
                        var audio = new Audio(URL.createObjectURL(data));
                        audio.play();
                        message.textContent = "Received audio (playing...)";
                        message.classList.add("translated");
                    } else {
                        // Assuming text message (e.g., status updates)
                        message.textContent = "Received: " + data;
                        message.classList.add("info");
                    }
                    messages.appendChild(message);
                };

                ws.onclose = function(event) {
                    console.log("WebSocket closed:", event);
                    document.getElementById("wsStatus").textContent = "Disconnected";
                    ws = null;
                    // Attempt to reconnect
                    if (reconnectAttempts < 5) { // Limit reconnect attempts
                        var delay = Math.min(reconnectInterval * Math.pow(2, reconnectAttempts), maxReconnectInterval);
                        console.log(`Attempting to reconnect in ${delay / 1000} seconds...`);
                        setTimeout(connectWebSocket, delay);
                        reconnectAttempts++;
                    } else {
                        console.log("Max reconnect attempts reached.");
                    }
                };

                ws.onerror = function(event) {
                    console.error("WebSocket error:", event);
                    document.getElementById("wsStatus").textContent = "Error";
                };
            }

            // Initial connection
            connectWebSocket();

            function sendMessage(event) {
                event.preventDefault();
                var input = document.getElementById("messageText");
                var message = input.value;
                if (ws && ws.readyState === WebSocket.OPEN) {
                    // Here you would typically send actual audio data (e.g., from microphone)
                    // For demonstration, we'll just send the text input as a placeholder.
                    // This is NOT how audio bytes would be sent.
                    var messages = document.getElementById("messages");
                    var li = document.createElement('li');
                    li.textContent = "Sent: " + message;
                    messages.appendChild(li);
                    
                    // Simulate sending binary audio data
                    // In a real scenario, this would be actual audio data from a microphone
                    // or a pre-recorded file converted to raw PCM bytes.
                    // For now, sending a dummy byte array or base64 encoded audio as a test.
                    // The actual /stream endpoint expects raw PCM bytes.
                    // This client-side code needs a proper audio capture mechanism.
                    var dummyAudioData = new Uint8Array([0x01, 0x02, 0x03, 0x04]); // Placeholder for raw audio bytes
                    ws.send(dummyAudioData.buffer); // Send as ArrayBuffer

                } else {
                    alert("WebSocket is not connected. Please refresh or check console for errors.");
                }
                input.value = '';
            }
        </script>
    </body>
</html>
"""


@app.get("/")
async def get():
    """Provides a simple HTML page to test the WebSocket connection."""
    return HTMLResponse(html)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": (get_global_translator() is not None)}

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """Handles real-time audio streaming via WebSocket."""
    await websocket.accept()
    call_sid = str(uuid.uuid4())
    logger.info(f"WebSocket connection accepted for call_sid: {call_sid}")

    session: CallSession = CallSession(call_sid=call_sid)

    try:
        # 1. Enrollment Phase
        await websocket.send_text("Please say a few words to enroll your voice.")
        enrollment_start_time = time.time()
        while not session.is_enrolled and (time.time() - enrollment_start_time) < (session.enrollment_duration_s + 2): # Add a small buffer for enrollment
            try:
                audio_bytes_8k = await websocket.receive_bytes()
                enrolled = await session.enroll_speaker(audio_bytes_8k)
                if enrolled:
                    await websocket.send_text(f"Enrollment complete. Language: {session.source_lang}. Connecting you now.")
                    break
            except asyncio.TimeoutError:
                logger.warning(f"Enrollment timeout for call {call_sid}.")
                break
            except WebSocketDisconnect:
                logger.info(f"Enrollment disconnected for call {call_sid}.")
                raise # Re-raise to be caught by outer handler

        if not session.is_enrolled:
            logger.error(f"Enrollment failed or timed out for call {call_sid}. Closing connection.")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        # 2. Bidirectional Streaming Phase
        logger.info(f"Starting bidirectional streaming for call_sid: {call_sid}")
        while True:
            audio_bytes_8k = await websocket.receive_bytes()
            translated_audio_8k = await session.process_chunk(audio_bytes_8k)
            if translated_audio_8k:
                await websocket.send_bytes(translated_audio_8k)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for call_sid: {call_sid}")
    except Exception as e:
        logger.error(f"Error in WebSocket processing for call_sid {call_sid}: {e}", exc_info=True)
    finally:
        await session.cleanup()
        logger.info(f"WebSocket connection closed for call_sid: {call_sid}")

if __name__ == "__main__":
    # This block is for direct execution of the server for testing.
    # In a real setup, uvicorn would be run via cli (e.g., from run.py)
    uvicorn.run(app, host="0.0.0.0", port=8000)
