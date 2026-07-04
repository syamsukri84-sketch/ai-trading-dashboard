from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json

router = APIRouter()

@router.websocket("/stream/{ticker}")
async def websocket_endpoint(websocket: WebSocket, ticker: str):
    await websocket.accept()
    try:
        while True:
            # Simulasi pengiriman data streaming setiap 5 detik
            await asyncio.sleep(5)
            data = {
                "ticker": ticker.upper(),
                "message": f"Streaming koneksi aktif untuk {ticker.upper()}",
            }
            await websocket.send_text(json.dumps(data))
    except WebSocketDisconnect:
        print(f"Client disconnected from {ticker} stream")