from __future__ import annotations

import io
import wave
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.speaker_diarization.engine import SpeakerAwareAnalyzer, SpeakerAnalysisError
from app.speaker_diarization.model import SpeakerDiarizationError, SpeakerDiarizationModel, SpeakerTurn, parse_diarization_output

client = TestClient(app)


def make_wav(duration: float = 6.0, frequency: float = 220.0) -> bytes:
    count = int(duration * 16_000)
    t = np.arange(count, dtype=np.float32) / 16_000
    signal = 0.08 * np.sin(2 * np.pi * frequency * t)
    pcm = np.clip(signal * 32767.0, -32768, 32767).astype('<i2')
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def test_parse_pyannote_annotation_tracks():
    class Annotation:
        def itertracks(self, yield_label=True):
            assert yield_label is True
            yield SimpleNamespace(start=0.0, end=1.2), "track0", "SPEAKER_00"
            yield SimpleNamespace(start=1.4, end=2.0), "track1", "SPEAKER_01"

    turns = parse_diarization_output(Annotation())
    assert [item.speaker for item in turns] == ["SPEAKER_00", "SPEAKER_01"]
    assert turns[0].duration_seconds == 1.2


def test_parse_diarize_output_wrapper():
    class Annotation:
        def itertracks(self, yield_label=True):
            yield SimpleNamespace(start=0.0, end=1.0), "t", "A"

    wrapped = SimpleNamespace(speaker_diarization=Annotation())
    turns = parse_diarization_output(wrapped)
    assert turns[0].speaker == "A"


def test_diarization_requires_token_before_import(monkeypatch):
    try:
        SpeakerDiarizationModel.load("fixture", token=None)
    except SpeakerDiarizationError as exc:
        assert "Hugging Face access token" in str(exc)
    else:
        raise AssertionError("Expected token validation failure")


def test_diarization_rejects_wrong_sample_rate():
    class Pipeline:
        def __call__(self, *args, **kwargs):
            raise AssertionError("pipeline must not be called")

    model = SpeakerDiarizationModel(
        pipeline=Pipeline(),
        model_id="fixture",
        device="cpu",
    )
    try:
        model.diarize(np.zeros(16000, dtype=np.float32), 8000)
    except SpeakerDiarizationError as exc:
        assert "expects 16000 Hz" in str(exc)
    else:
        raise AssertionError("Expected sample-rate validation failure")


def test_diarization_passes_waveform_dict_and_speaker_bounds(monkeypatch):
    calls = {}

    class FakeTorchTensor:
        shape = (1, 16000)
        def unsqueeze(self, dim):
            assert dim == 0
            return self

    class FakeTorch:
        @staticmethod
        def from_numpy(array):
            assert array.dtype == np.float32
            return FakeTorchTensor()

    class Pipeline:
        def __call__(self, request, **kwargs):
            calls["request"] = request
            calls["kwargs"] = kwargs
            return SimpleNamespace(speaker_diarization=_annotation())

    def _annotation():
        class Annotation:
            def itertracks(self, yield_label=True):
                yield SimpleNamespace(start=0.0, end=1.5), "0", "A"
        return Annotation()

    import app.speaker_diarization.model as model_module
    monkeypatch.setattr(model_module.importlib, "import_module", lambda name: FakeTorch if name == "torch" else (_ for _ in ()).throw(ImportError()))
    model = SpeakerDiarizationModel(
        pipeline=Pipeline(),
        model_id="fixture",
        device="cpu",
        min_speakers=1,
        max_speakers=3,
    )
    turns = model.diarize(np.zeros(16000, dtype=np.float32), 16000)
    assert len(turns) == 1
    assert calls["kwargs"] == {"min_speakers": 1, "max_speakers": 3}
    assert calls["request"]["sample_rate"] == 16000


def test_speaker_engine_merges_adjacent_same_speaker_and_summarizes():
    calls = []

    def predictor(samples, sample_rate):
        calls.append(len(samples))
        return {
            "emotion": "angry",
            "confidence": 0.8,
            "probabilities": {"angry": 0.8, "neutral": 0.2},
            "valence": 0.2,
            "arousal": 0.9,
            "dominance": 0.7,
        }

    samples = np.ones(6 * 16_000, dtype=np.float32)
    turns = [
        SpeakerTurn("A", 0.0, 1.0, 1.0),
        SpeakerTurn("A", 1.1, 2.0, 0.9),
        SpeakerTurn("B", 2.4, 3.6, 1.2),
    ]
    result = SpeakerAwareAnalyzer().analyze(samples, 16000, turns, predictor=predictor)
    assert result.speaker_count == 2
    assert result.speakers[0].speaker == "A"
    assert result.speakers[0].turn_count == 1
    assert result.speaker_switches == 1
    assert result.dominant_speaker == "A"
    assert result.speech_coverage == 3.2 / 6.0
    assert len(calls) == 2


def test_speaker_engine_reports_interaction_shift():
    def predictor(samples, sample_rate):
        if len(samples) <= 16_000:
            return {
                "emotion": "angry",
                "confidence": 0.9,
                "probabilities": {"angry": 0.9, "neutral": 0.1},
                "valence": 0.1,
                "arousal": 0.9,
                "dominance": 0.8,
            }
        return {
            "emotion": "neutral",
            "confidence": 0.8,
            "probabilities": {"neutral": 0.8, "angry": 0.2},
            "valence": 0.8,
            "arousal": 0.5,
            "dominance": 0.5,
        }

    samples = np.ones(4 * 16000, dtype=np.float32)
    turns = [SpeakerTurn("A", 0.0, 1.0, 1.0), SpeakerTurn("B", 1.2, 2.4, 1.2)]
    result = SpeakerAwareAnalyzer().analyze(samples, 16000, turns, predictor=predictor)
    assert result.interactions[0].type in {"activated_shift", "emotion_transition"}
    assert result.interactions[0].from_speaker == "A"
    assert result.interactions[0].to_speaker == "B"


def test_speaker_engine_rejects_invalid_probabilities():
    def predictor(samples, sample_rate):
        return {"emotion": "neutral", "confidence": 0.5, "probabilities": {"neutral": -1}}

    with np.testing.assert_raises(SpeakerAnalysisError):
        SpeakerAwareAnalyzer().analyze(
            np.ones(2 * 16000, dtype=np.float32),
            16000,
            [SpeakerTurn("A", 0, 1.2, 1.2)],
            predictor=predictor,
        )


def test_speaker_status_has_explicit_unavailable_path():
    import app.routers.speaker_aware as router

    class FakeStatus:
        pass

    from app.config import settings
    original = settings.SPEAKER_DIARIZATION_ENABLED
    settings.SPEAKER_DIARIZATION_ENABLED = False
    try:
        response = client.get('/api/v1/audio/speakers/status')
        assert response.status_code == 200
        assert response.json()['available'] is False
        assert response.json()['enabled'] is False
    finally:
        settings.SPEAKER_DIARIZATION_ENABLED = original


def test_speaker_endpoint_uses_injected_models(monkeypatch):
    import app.routers.speaker_aware as router
    from app.emotion_baseline.model import ClassicalEmotionModel

    class FakeDiarizer:
        model_id = 'fixture-diarizer'
        def metadata(self):
            return {'model_id': self.model_id, 'sample_rate': 16000, 'device': 'cpu'}
        def diarize(self, samples, sample_rate):
            return [
                SpeakerTurn('A', 0.0, 1.5, 1.5),
                SpeakerTurn('B', 1.7, 3.2, 1.5),
            ]

    class FakeFusionArtifact:
        version = 'fixture-fusion'

    class FakeFusion:
        artifact = FakeFusionArtifact()
        def combine(self, classical, deep):
            return SimpleNamespace(emotion='neutral', confidence=0.8, probabilities={'neutral': 0.8, 'angry': 0.2})

    class FakeClassical:
        def predict(self, vector):
            return {'probabilities': {'neutral': 0.8, 'angry': 0.2}, 'confidence': 0.8}

    class FakeDeepPrediction:
        emotion = 'neutral'
        confidence = 0.8
        probabilities = {'neutral': 0.8, 'angry': 0.2}
        class Metadata:
            model_id = 'fixture-deep'
        metadata = Metadata()

    class FakeDeep:
        def predict(self, samples, sample_rate):
            return FakeDeepPrediction()

    class FakeAffect:
        def predict(self, samples, sample_rate):
            return SimpleNamespace(values={'valence': 0.7, 'arousal': 0.4, 'dominance': 0.5})
        def metadata(self):
            return {'model_id': 'fixture-affect'}

    monkeypatch.setattr(router, 'get_diarization_model', lambda: FakeDiarizer())
    monkeypatch.setattr(router, 'get_engine', lambda: FakeFusion())
    monkeypatch.setattr(ClassicalEmotionModel, 'load', lambda _: FakeClassical())
    monkeypatch.setattr(router, 'get_deep_model', lambda: FakeDeep())
    monkeypatch.setattr(router, 'get_affect_model', lambda: FakeAffect())
    monkeypatch.setattr(router.settings, 'SPEAKER_DIARIZATION_ENABLED', True)
    monkeypatch.setattr(router.settings, 'DIMENSIONAL_EMOTION_ENABLED', True)

    response = client.post(
        '/api/v1/audio/speakers',
        files={'file': ('conversation.wav', make_wav(), 'audio/wav')},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['speaker_count'] == 2
    assert {item['speaker'] for item in payload['speakers']} == {'A', 'B'}
    assert payload['turns'][0]['emotion'] == 'neutral'
    assert payload['affect_model'] == 'fixture-affect'
    assert payload['interaction_graph']['graph_type'] == 'directed_conversation_interaction'
    assert len(payload['interaction_graph']['nodes']) == 2
    assert len(payload['interaction_graph']['edges']) == 1
    assert payload['conversation_dynamics']['event_count'] == 1
    assert payload['conversation_dynamics']['dominant_pattern'] in {'escalating', 'deescalating', 'stable', 'mixed'}
    assert len(payload['conversation_dynamics']['pair_metrics']) == 1
    assert payload['conversation_state']['state_count'] == payload['conversation_dynamics']['event_count']
    assert payload['conversation_state']['end_state'] in {'calm', 'engaged', 'stable', 'tense', 'escalating', 'peak_tension', 'cooling', 'deescalating', 'insufficient_data'}


def test_speaker_endpoint_rejects_overlong_audio(monkeypatch):
    import app.routers.speaker_aware as router
    from app.config import settings

    original = settings.SPEAKER_DIARIZATION_MAX_DURATION_SECONDS
    monkeypatch.setattr(router.settings, 'SPEAKER_DIARIZATION_MAX_DURATION_SECONDS', 1.0)
    monkeypatch.setattr(router.settings, 'SPEAKER_DIARIZATION_ENABLED', True)
    try:
        response = client.post(
            '/api/v1/audio/speakers',
            files={'file': ('conversation.wav', make_wav(duration=1.2), 'audio/wav')},
        )
        assert response.status_code == 422
        assert 'too long' in response.json()['detail'].lower()
    finally:
        settings.SPEAKER_DIARIZATION_MAX_DURATION_SECONDS = original
