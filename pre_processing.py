from pathlib import Path

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf


TARGET_SAMPLE_RATE = 16000


def load_audio(wav_file, sample_rate=TARGET_SAMPLE_RATE):
    """Load audio as mono float32 and resample for Whisper."""
    audio, sr = librosa.load(wav_file, sr=sample_rate, mono=True)
    return audio.astype(np.float32), sr


def reduce_noise(audio, sr, noise_duration=0.5, prop_decrease=0.8):
    noise_sample_count = min(len(audio), int(noise_duration * sr))
    if noise_sample_count <= 0:
        return audio

    noise_sample = audio[:noise_sample_count]
    return nr.reduce_noise(
        y=audio,
        sr=sr,
        y_noise=noise_sample,
        prop_decrease=prop_decrease,
        stationary=True,
    )


def normalize_audio(audio, target_db=-3.0):
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio

    target_amplitude = 10 ** (target_db / 20)
    audio_normalized = audio * (target_amplitude / peak)
    return np.clip(audio_normalized, -1.0, 1.0)


def trim_silence(audio, sr, top_db=30):
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    print(f"Trim silence: {len(audio) / sr:.2f}s -> {len(audio_trimmed) / sr:.2f}s")
    return audio_trimmed


def build_output_path(wav_path, output_path=None):
    if output_path:
        return Path(output_path)

    input_path = Path(wav_path)
    return input_path.with_name(f"{input_path.stem}_processed.wav")


def apply_preprocessing(wav_path, output_path=None, sample_rate=TARGET_SAMPLE_RATE):
    """Create a processed WAV without overwriting the original input file."""
    processed_path = build_output_path(wav_path, output_path)

    audio, sr = load_audio(wav_path, sample_rate=sample_rate)
    print(f"Loaded audio: mono, {sr} Hz")

    audio = reduce_noise(audio, sr)
    print("Noise reduction done.")

    audio = normalize_audio(audio)
    print("Volume normalization done.")

    audio = trim_silence(audio, sr)

    sf.write(processed_path, audio, sr)
    print(f"Pre-processing done: {processed_path}")
    return str(processed_path)
