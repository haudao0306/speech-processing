import os

from pre_processing import apply_preprocessing
from semantic_processor import correct_transcription
from stt import transcribe


def print_changes(changes):
    if not changes:
        print("No semantic corrections needed.")
        return

    print("Semantic corrections:")
    for change in changes:
        segment = change.get("segment_index")
        location = f"segment {segment + 1}" if segment is not None else "full text"
        print(f"  - [{location}] {change['from']} -> {change['to']}")


def main():
    wav_path = r"D:\NAM4\TTDN\test_file_4.wav"

    if not os.path.exists(wav_path):
        print(f"Input file not found: {wav_path}")
        print("Please convert your source audio to WAV first.")
        return

    print("--- START AUDIO PIPELINE ---")

    try:
        print("\n[STEP 1: PRE-PROCESSING]")
        processed_wav_path = apply_preprocessing(wav_path)

        print("\n[STEP 2: SPEECH-TO-TEXT]")
        results, full_text = transcribe(processed_wav_path, language="vi")

        if not full_text:
            print("Whisper did not detect any speech content.")
            return

        print("\n[STEP 3: SEMANTIC PROCESSING]")
        try:
            results, full_text, changes = correct_transcription(results, full_text)
            print_changes(changes)
        except Exception as exc:
            print(f"Semantic processing failed. Keeping Whisper output. Error: {exc}")

        print("\n" + "=" * 30)
        print(f"FINAL TEXT: {full_text}")
        print("=" * 30)

    except Exception as exc:
        print(f"Pipeline failed: {exc}")


if __name__ == "__main__":
    main()
