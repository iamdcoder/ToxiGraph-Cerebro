from __future__ import annotations

import asyncio
import time
from threading import Lock

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.security import GLOBAL_WS_CONNECTION_LIMITER, api_key_matches, client_ip, websocket_policy_ok

from app.audio.processor import AudioProcessingError
from app.config import settings
from app.deep_emotion.model import DeepEmotionModelError
from app.deep_emotion.service import get_model as get_deep_model
from app.dimensional_emotion.model import DimensionalEmotionError
from app.dimensional_emotion.service import get_model as get_affect_model
from app.emotion_baseline.model import BaselineModelError, ClassicalEmotionModel
from app.features.acoustic import extract_acoustic_features
from app.fusion.service import get_engine
from app.fusion.training import FusionTrainingError
from app.live.engine import LiveAnalysisError, LiveConfig, LiveInferenceEngine, LiveStreamSession

router = APIRouter()

_runtime = None
_runtime_lock = Lock()
_inference_lock = Lock()


def get_live_runtime():
    global _runtime
    if _runtime is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is not None:
            return _runtime

        try:
            fusion_engine = get_engine()
            classical_model = ClassicalEmotionModel.load(settings.CLASSICAL_MODEL_PATH)
            deep_model = get_deep_model()
        except (FusionTrainingError, BaselineModelError, DeepEmotionModelError) as exc:
            raise LiveAnalysisError(str(exc)) from exc

        affect_model = None
        if settings.DIMENSIONAL_EMOTION_ENABLED:
            try:
                affect_model = get_affect_model()
            except DimensionalEmotionError:
                affect_model = None

        def predictor(samples: np.ndarray, sample_rate: int) -> dict:
            # One global inference lock keeps concurrent live sessions from loading
            # or executing large model graphs against the same process simultaneously.
            with _inference_lock:
                features = extract_acoustic_features(samples, sample_rate)
                classical = classical_model.predict(features.to_vector())
                deep = deep_model.predict(samples, sample_rate)
                fused = fusion_engine.combine(
                    classical["probabilities"],
                    deep.probabilities,
                )
                output = {
                    "emotion": fused.emotion,
                    "confidence": fused.confidence,
                    "probabilities": dict(fused.probabilities),
                    "model_metadata": {
                        "fusion_calibration": fusion_engine.artifact.version,
                        "deep_model": getattr(deep.metadata, "model_id", None),
                    },
                }
                if affect_model is not None:
                    try:
                        affect = affect_model.predict(samples, sample_rate)
                        output.update(affect.values)
                        output["model_metadata"]["affect_model"] = affect_model.metadata()["model_id"]
                    except DimensionalEmotionError:
                        pass
                return output

        _runtime = LiveInferenceEngine(
            predictor=predictor,
            config=LiveConfig(
                target_sample_rate=settings.LIVE_STREAM_TARGET_SAMPLE_RATE,
                window_seconds=settings.LIVE_STREAM_WINDOW_SECONDS,
                hop_seconds=settings.LIVE_STREAM_HOP_SECONDS,
                min_window_seconds=settings.LIVE_STREAM_MIN_WINDOW_SECONDS,
                max_duration_seconds=settings.LIVE_STREAM_MAX_DURATION_SECONDS,
                max_chunk_bytes=settings.LIVE_STREAM_MAX_CHUNK_BYTES,
                min_speech_ratio=settings.LIVE_STREAM_MIN_SPEECH_RATIO,
                smoothing_alpha=settings.LIVE_STREAM_SMOOTHING_ALPHA,
            ),
        )
        return _runtime


def clear_live_runtime_cache() -> None:
    global _runtime
    with _runtime_lock:
        _runtime = None


@router.get("/status")
def live_status() -> dict:
    return {
        "available": bool(settings.LIVE_STREAM_ENABLED),
        "enabled": bool(settings.LIVE_STREAM_ENABLED),
        "mode": "single_speaker_realtime",
        "transport": "websocket",
        "window_seconds": settings.LIVE_STREAM_WINDOW_SECONDS,
        "hop_seconds": settings.LIVE_STREAM_HOP_SECONDS,
        "target_sample_rate": settings.LIVE_STREAM_TARGET_SAMPLE_RATE,
        "max_duration_seconds": settings.LIVE_STREAM_MAX_DURATION_SECONDS,
        "min_speech_ratio": settings.LIVE_STREAM_MIN_SPEECH_RATIO,
        "message": (
            "Live analysis streams microphone PCM16 chunks over WebSocket and runs rolling, "
            "single-speaker emotion and affect inference. It does not perform live speaker diarization."
        ),
    }


async def _run_window(runtime, session, snapshot, start_seconds, end_seconds):
    started = time.perf_counter()
    result = await asyncio.to_thread(
        runtime.analyze_window,
        session,
        snapshot,
        start_seconds,
        end_seconds,
        latency_ms=0.0,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    result = result.__class__(
        sequence=result.sequence,
        window_start_seconds=result.window_start_seconds,
        window_end_seconds=result.window_end_seconds,
        emotion=result.emotion,
        confidence=result.confidence,
        probabilities=result.probabilities,
        valence=result.valence,
        arousal=result.arousal,
        dominance=result.dominance,
        state=result.state,
        state_transition=result.state_transition,
        speech_ratio=result.speech_ratio,
        quality_score=result.quality_score,
        quality_verdict=result.quality_verdict,
        latency_ms=latency_ms,
        model_metadata=result.model_metadata,
    )
    return result


@router.websocket("/ws")
async def live_websocket(websocket: WebSocket) -> None:
    if not settings.LIVE_STREAM_ENABLED:
        await websocket.close(code=1008, reason="Live streaming is disabled by configuration.")
        return

    if not websocket_policy_ok(websocket, settings.ALLOWED_ORIGINS):
        await websocket.close(code=1008, reason="WebSocket origin is not allowed by CEREBRO configuration.")
        return

    connection_key = client_ip(websocket.scope)
    if not GLOBAL_WS_CONNECTION_LIMITER.acquire(connection_key):
        await websocket.close(code=1013, reason="Too many live connections from this client.")
        return

    await websocket.accept()
    session: LiveStreamSession | None = None
    runtime = None
    analysis_task: asyncio.Task | None = None
    receive_task: asyncio.Task | None = None
    stopping = False

    try:
        receive_task = asyncio.create_task(websocket.receive())
        while True:
            wait_tasks = {receive_task}
            if analysis_task is not None:
                wait_tasks.add(analysis_task)
            done, _ = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)

            if analysis_task is not None and analysis_task in done:
                try:
                    result = analysis_task.result()
                except LiveAnalysisError as exc:
                    await websocket.send_json({"type": "error", "code": "analysis_failed", "message": str(exc)})
                except Exception as exc:
                    await websocket.send_json({"type": "error", "code": "analysis_failed", "message": f"Live analysis failed: {exc}"})
                else:
                    await websocket.send_json(result.to_dict())
                analysis_task = None

                if stopping and session is not None:
                    await websocket.send_json({
                        "type": "complete",
                        "duration_seconds": round(session.duration_seconds, 4),
                        "updates": session.sequence,
                        "final_state": session.state_machine.current_state,
                    })
                    break

            if receive_task in done:
                message = receive_task.result()
                receive_task = asyncio.create_task(websocket.receive())

                if message.get("type") == "websocket.disconnect":
                    break

                if message.get("text") is not None:
                    import json

                    try:
                        payload = json.loads(message["text"])
                    except json.JSONDecodeError:
                        await websocket.send_json({"type": "error", "code": "invalid_json", "message": "Control message must be valid JSON."})
                        continue

                    message_type = payload.get("type")
                    if message_type == "start":
                        if settings.WEBSOCKET_AUTH_ENABLED and not api_key_matches(
                            str(payload.get("token", "")),
                            settings.WEBSOCKET_AUTH_TOKEN,
                        ):
                            await websocket.send_json({
                                "type": "error",
                                "code": "unauthorized",
                                "message": "A valid WebSocket authentication token is required.",
                            })
                            continue
                        if session is not None:
                            await websocket.send_json({"type": "error", "code": "already_started", "message": "The live session has already started."})
                            continue
                        try:
                            source_rate = int(payload.get("sample_rate", 0))
                            session = LiveStreamSession(
                                source_sample_rate=source_rate,
                                config=LiveConfig(
                                    target_sample_rate=settings.LIVE_STREAM_TARGET_SAMPLE_RATE,
                                    window_seconds=settings.LIVE_STREAM_WINDOW_SECONDS,
                                    hop_seconds=settings.LIVE_STREAM_HOP_SECONDS,
                                    min_window_seconds=settings.LIVE_STREAM_MIN_WINDOW_SECONDS,
                                    max_duration_seconds=settings.LIVE_STREAM_MAX_DURATION_SECONDS,
                                    max_chunk_bytes=settings.LIVE_STREAM_MAX_CHUNK_BYTES,
                                    min_speech_ratio=settings.LIVE_STREAM_MIN_SPEECH_RATIO,
                                    smoothing_alpha=settings.LIVE_STREAM_SMOOTHING_ALPHA,
                                ),
                            )
                            runtime = await asyncio.to_thread(get_live_runtime)
                        except (ValueError, LiveAnalysisError) as exc:
                            await websocket.send_json({"type": "error", "code": "start_failed", "message": str(exc)})
                            await websocket.close(code=1008, reason="Invalid live-session configuration.")
                            break
                        await websocket.send_json({
                            "type": "ready",
                            "mode": "single_speaker_realtime",
                            "source_sample_rate": source_rate,
                            "target_sample_rate": session.config.target_sample_rate,
                            "window_seconds": session.config.window_seconds,
                            "hop_seconds": session.config.hop_seconds,
                            "max_duration_seconds": session.config.max_duration_seconds,
                        })
                    elif message_type == "stop":
                        if session is None:
                            await websocket.send_json({"type": "error", "code": "not_started", "message": "Start the live session before stopping it."})
                            continue
                        stopping = True
                        if analysis_task is None:
                            pending = session.pending_window(final=True)
                            if pending is not None:
                                snapshot, start_seconds, end_seconds = pending
                                analysis_task = asyncio.create_task(
                                    _run_window(runtime, session, snapshot, start_seconds, end_seconds)
                                )
                            else:
                                await websocket.send_json({
                                    "type": "complete",
                                    "duration_seconds": round(session.duration_seconds, 4),
                                    "updates": session.sequence,
                                    "final_state": session.state_machine.current_state,
                                })
                                break
                    elif message_type == "ping":
                        await websocket.send_json({"type": "pong"})
                    else:
                        await websocket.send_json({"type": "error", "code": "unknown_control", "message": "Use start, stop, or ping control messages."})

                elif message.get("bytes") is not None:
                    if session is None:
                        await websocket.send_json({"type": "error", "code": "not_started", "message": "Send a start control message before audio."})
                        continue
                    try:
                        session.append_pcm16(message["bytes"])
                    except LiveAnalysisError as exc:
                        await websocket.send_json({"type": "error", "code": "audio_rejected", "message": str(exc)})
                        await websocket.close(code=1008, reason="Invalid live audio chunk.")
                        break

                    if analysis_task is None and not stopping:
                        pending = session.pending_window()
                        if pending is not None:
                            snapshot, start_seconds, end_seconds = pending
                            analysis_task = asyncio.create_task(
                                _run_window(runtime, session, snapshot, start_seconds, end_seconds)
                            )

    except WebSocketDisconnect:
        return
    finally:
        GLOBAL_WS_CONNECTION_LIMITER.release(connection_key)
        if receive_task is not None and not receive_task.done():
            receive_task.cancel()
        if analysis_task is not None and not analysis_task.done():
            analysis_task.cancel()
