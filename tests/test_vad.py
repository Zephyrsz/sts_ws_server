import struct

import numpy as np

from sts_ws_server.vad import EnergyVAD, SileroONNXVAD


def pcm_frame(amplitude: int, frame_samples: int = 320) -> bytes:
    return struct.pack("<" + "h" * frame_samples, *([amplitude] * frame_samples))


def test_energy_vad_removes_silence_and_keeps_nearby_padding() -> None:
    vad = EnergyVAD(sample_rate=16_000, frame_ms=20, threshold=500, padding_ms=20)
    audio = b"".join(
        [pcm_frame(0) for _ in range(5)]
        + [pcm_frame(2_000) for _ in range(3)]
        + [pcm_frame(0) for _ in range(5)]
    )

    filtered = vad.filter(audio)

    assert len(filtered) == 5 * 640
    assert filtered == b"".join([pcm_frame(0), *[pcm_frame(2_000) for _ in range(3)], pcm_frame(0)])


def test_energy_vad_returns_empty_for_silence() -> None:
    vad = EnergyVAD(sample_rate=16_000, frame_ms=20, threshold=500, padding_ms=20)

    assert vad.filter(pcm_frame(0) * 4) == b""


def test_silero_onnx_vad_runs_stateful_16khz_inference_and_keeps_padding(monkeypatch) -> None:
    class FakeSession:
        def __init__(self, model_path, providers):
            assert model_path == "/models/silero_vad.onnx"
            assert providers == ["CPUExecutionProvider"]
            self.calls = []

        def get_inputs(self):
            return [
                type("Input", (), {"name": "input", "shape": [None, None]})(),
                type("Input", (), {"name": "state", "shape": [2, None, 128]})(),
                type("Input", (), {"name": "sr", "shape": []})(),
            ]

        def run(self, _outputs, inputs):
            self.calls.append(inputs)
            probabilities = [0.05, 0.9, 0.05]
            return [
                np.array([[probabilities[len(self.calls) - 1]]], dtype=np.float32),
                np.ones((2, 1, 128), dtype=np.float32) * len(self.calls),
            ]

    fake_session = FakeSession

    class Runtime:
        SessionOptions = type("SessionOptions", (), {})
        InferenceSession = staticmethod(lambda path, sess_options, providers: _create_session(path, providers))

    def _create_session(path, providers):
        session = fake_session(path, providers)
        sessions.append(session)
        return session

    sessions = []
    monkeypatch.setattr("sts_ws_server.vad.Path.is_file", lambda _self: True)
    monkeypatch.setattr("sts_ws_server.vad._load_onnxruntime", lambda: (Runtime, np))
    vad = SileroONNXVAD("/models/silero_vad.onnx", padding_ms=20, threshold=0.5)

    audio = b"".join([pcm_frame(0, 512), pcm_frame(1000, 512), pcm_frame(0, 512)])
    filtered = vad.filter(audio)

    assert filtered == audio
    assert [call["input"].shape for call in sessions[0].calls] == [(1, 512)] * 3
    assert all(int(call["sr"].item()) == 16_000 for call in sessions[0].calls)
    assert np.all(sessions[0].calls[1]["state"] == 1)
    assert vad.filter(pcm_frame(1000, frame_samples=320)) == b""
    assert len(sessions[0].calls) == 3
