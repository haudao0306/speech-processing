import os

from faster_whisper import WhisperModel


FFMPEG_BIN = r"D:\ffmpeg\bin"
MODEL_SIZE = "medium"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

_model = None


def configure_ffmpeg():
    if FFMPEG_BIN not in os.environ["PATH"]:
        os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ["PATH"]


def get_model():
    global _model
    if _model is None:
        configure_ffmpeg()
        print(f"Loading faster-whisper model: {MODEL_SIZE}")
        _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("Whisper model is ready.")
    return _model


def transcribe(wav_file, language="vi", beam_size=5, word_timestamps=False):
    print(f"Transcribing: {wav_file}")

    model = get_model()
    segments, info = model.transcribe(
        wav_file,
        language=language,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
    )

    print(f"Detected language: {info.language} ({info.language_probability:.2f})\n")

    results = []
    full_text = []

    for seg in segments:
        text = seg.text.strip()
        segment_data = {
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": text,
        }
        results.append(segment_data)
        full_text.append(text)

        print(f"[{seg.start:5.2f}s -> {seg.end:5.2f}s] {text}")

    return results, " ".join(full_text)
