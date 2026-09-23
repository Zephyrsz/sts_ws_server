from __future__ import annotations

import math
import struct
from pathlib import Path


class EnergyVAD:
    """Small PCM16 energy gate used inside the ASR service."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        frame_ms: int = 20,
        threshold: float = 450.0,
        padding_ms: int = 120,
    ) -> None:
        if sample_rate < 1 or frame_ms < 1 or padding_ms < 0 or threshold < 0:
            raise ValueError("invalid VAD configuration")
        self.frame_bytes = sample_rate * 2 * frame_ms // 1000
        self.padding_frames = math.ceil(padding_ms / frame_ms)
        self.threshold = threshold

    def filter(self, pcm_s16le: bytes) -> bytes:
        if not pcm_s16le:
            return b""
        if len(pcm_s16le) % 2:
            raise ValueError("PCM S16LE audio must contain complete 16-bit samples")

        frames = [
            pcm_s16le[offset : offset + self.frame_bytes]
            for offset in range(0, len(pcm_s16le), self.frame_bytes)
        ]
        voiced = [index for index, frame in enumerate(frames) if self._rms(frame) >= self.threshold]
        if not voiced:
            return b""

        keep = bytearray(len(frames))
        for index in voiced:
            start = max(0, index - self.padding_frames)
            end = min(len(frames), index + self.padding_frames + 1)
            keep[start:end] = b"\x01" * (end - start)
        return b"".join(frame for index, frame in enumerate(frames) if keep[index])

    @staticmethod
    def _rms(frame: bytes) -> float:
        values = (sample[0] for sample in struct.iter_unpack("<h", frame[: len(frame) - len(frame) % 2]))
        total = 0
        count = 0
        for sample in values:
            total += sample * sample
            count += 1
        return math.sqrt(total / count) if count else 0.0


def _load_onnxruntime():
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("VAD_BACKEND=silero requires the 'vad' extra") from exc
    return ort, np


class SileroONNXVAD:
    """Stateful Silero VAD v5 inference over 16 kHz PCM16 audio."""

    def __init__(
        self,
        model_path: str,
        *,
        threshold: float = 0.5,
        frame_ms: int = 32,
        padding_ms: int = 120,
        provider: str = "CPUExecutionProvider",
    ) -> None:
        if not 0.0 <= threshold <= 1.0 or frame_ms != 32 or padding_ms < 0:
            raise ValueError("Silero VAD requires threshold in [0, 1], 32 ms frames, and non-negative padding")
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Silero ONNX model not found: {model_path}")

        ort, self.np = _load_onnxruntime()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=[provider])
        session_inputs = self.session.get_inputs()
        self.input_names = {item.name for item in session_inputs}
        if not {"input", "state", "sr"}.issubset(self.input_names):
            raise ValueError("Silero VAD ONNX graph must expose input, state, and sr")
        state_input = next(item for item in session_inputs if item.name == "state")
        state_shape = getattr(state_input, "shape", None)
        state_width = state_shape[-1] if isinstance(state_shape, (list, tuple)) else None
        if not isinstance(state_width, int) or state_width < 1:
            state_width = 128
        self.state_shape = (2, 1, state_width)
        self.threshold = threshold
        self.frame_ms = frame_ms
        self.frame_samples = 512
        self.frame_bytes = self.frame_samples * 2
        self.padding_frames = math.ceil(padding_ms / frame_ms)
        self.reset_states()

    def reset_states(self) -> None:
        self.state = self.np.zeros(self.state_shape, dtype=self.np.float32)

    def filter(self, pcm_s16le: bytes) -> bytes:
        if not pcm_s16le:
            return b""
        if len(pcm_s16le) % 2:
            raise ValueError("PCM S16LE audio must contain complete 16-bit samples")
        np = self.np
        samples = np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float32) / 32768.0
        if len(samples) == 0:
            return b""
        if len(samples) < self.frame_samples:
            return b""

        frames = [samples[offset : offset + self.frame_samples] for offset in range(0, len(samples), self.frame_samples)]
        framed = [np.pad(frame, (0, self.frame_samples - len(frame))) if len(frame) < self.frame_samples else frame for frame in frames]
        if len(samples) * 1000 / 16_000 < 32:
            raise ValueError("Silero VAD requires audio chunks of at least 32 ms at 16000 Hz")

        self.reset_states()
        probabilities: list[float] = []
        sample_rate = np.array(16_000, dtype=np.int64)
        for frame in framed:
            inputs = {"input": frame.reshape(1, -1), "state": self.state, "sr": sample_rate}
            outputs = self.session.run(None, inputs)
            probabilities.append(float(np.asarray(outputs[0]).reshape(-1)[0]))
            self.state = np.asarray(outputs[1], dtype=np.float32)

        keep = np.zeros(len(frames), dtype=bool)
        for index, probability in enumerate(probabilities):
            if probability >= self.threshold:
                start = max(0, index - self.padding_frames)
                end = min(len(frames), index + self.padding_frames + 1)
                keep[start:end] = True
        if not keep.any():
            return b""
        raw_frames = [
            pcm_s16le[offset : offset + self.frame_bytes]
            for offset in range(0, len(pcm_s16le), self.frame_bytes)
        ]
        return b"".join(raw_frames[index] for index in np.flatnonzero(keep))
