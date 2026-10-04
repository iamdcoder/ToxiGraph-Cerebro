import io
import wave

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.transcription.service import ASRTranscriptionError, TranscriptionResult, WhisperTranscriber
from app.transcription.verification import normalize_transcript, verify_transcripts, word_error_rate

client = TestClient(app)


def make_wav(*, duration: float = 1.2, frequency: float = 220.0, amplitude: float = 0.15) -> bytes:
    count = int(duration * 16_000)
    t = np.arange(count, dtype=np.float32) / 16_000
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    pcm = np.clip(signal * 32767.0, -32768, 32767).astype('<i2')
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def test_normalize_transcript_is_stable():
    assert normalize_transcript("  Hello, WORLD! ") == "hello world"
    assert normalize_transcript("Same\n sentence.") == "same sentence"


def test_word_error_rate_exact_and_single_substitution():
    assert word_error_rate(["hello", "world"], ["hello", "world"]) == 0.0
    assert word_error_rate(["hello", "world"], ["hello", "there"]) == 0.5


def test_verify_transcripts_marks_same():
    result = verify_transcripts("I am completely fine.", "i am completely fine")
    assert result.status == "same"
    assert result.similarity == 1.0
    assert result.word_error_rate == 0.0


def test_verify_transcripts_allows_small_asr_variation():
    result = verify_transcripts("I am completely fine.", "I am completely find")
    assert result.status == "same"
    assert result.similarity >= 0.9


def test_verify_transcripts_marks_uncertain_for_partial_overlap():
    result = verify_transcripts("I am completely fine and calm", "I am fine")
    assert result.status == "uncertain"
    assert 0.0 < result.similarity < 0.9


def test_verify_transcripts_marks_different():
    result = verify_transcripts("The meeting starts at five", "The weather is beautiful today")
    assert result.status == "different"
    assert result.similarity < 0.70


def test_verify_transcripts_marks_empty_as_uncertain():
    result = verify_transcripts("", "hello")
    assert result.status == "uncertain"
    assert result.words_a == 0


def test_transcription_model_rejects_wrong_sample_rate_without_loading():
    transcriber = WhisperTranscriber(
        model_id="fixture",
        processor=None,
        model=None,
        device="cpu",
    )
    try:
        transcriber.transcribe(np.zeros(16_000, dtype=np.float32), 8_000)
    except ASRTranscriptionError as exc:
        assert "expects 16000 Hz" in str(exc)
    else:
        raise AssertionError("Expected sample-rate validation failure")


def test_transcription_status_describes_local_whisper_configuration():
    response = client.get('/api/v1/audio/transcribe/status')
    assert response.status_code == 200
    payload = response.json()
    assert payload['model_id'] == 'openai/whisper-small'
    assert payload['sample_rate'] == 16000


def test_transcription_endpoint_uses_injected_transcriber(monkeypatch):
    import app.routers.transcription as router

    class FakeTranscriber:
        def transcribe(self, samples, sample_rate):
            return TranscriptionResult(
                text='I am completely fine',
                model_id='fixture-whisper',
                language='en',
                duration_seconds=float(len(samples) / sample_rate),
            )

    monkeypatch.setattr(router, 'get_transcriber', lambda: FakeTranscriber())
    response = client.post(
        '/api/v1/audio/transcribe',
        files={'file': ('sample.wav', make_wav(), 'audio/wav')},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['transcription']['text'] == 'I am completely fine'
    assert payload['transcription']['model_id'] == 'fixture-whisper'


def test_transcriber_runs_generation_and_decoding_with_injected_components(monkeypatch):
    import app.transcription.service as service

    class FakeTensor:
        def to(self, device):
            return self

    class FakeTorch:
        class _Context:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
        @staticmethod
        def inference_mode():
            return FakeTorch._Context()

    class FakeProcessor:
        def __call__(self, array, sampling_rate, return_tensors):
            assert sampling_rate == 16000
            return {"input_features": FakeTensor()}
        def batch_decode(self, generated, skip_special_tokens=True):
            assert generated == [[1, 2, 3]]
            return ["  I am completely fine.  "]

    class FakeModel:
        def generate(self, **kwargs):
            assert "input_features" in kwargs
            return [[1, 2, 3]]

    original_import = service.importlib.import_module
    def fake_import(name):
        if name == "torch":
            return FakeTorch
        return original_import(name)

    monkeypatch.setattr(service.importlib, "import_module", fake_import)
    transcriber = WhisperTranscriber(
        model_id="fixture-whisper",
        processor=FakeProcessor(),
        model=FakeModel(),
        device="cpu",
    )
    result = transcriber.transcribe(np.ones(16_000, dtype=np.float32) * 0.05, 16000)
    assert result.text == "I am completely fine."
