import os
from pydub import AudioSegment

# Fix FFmpeg path
os.environ["PATH"] += os.pathsep + r"D:\ffmpeg\bin"
AudioSegment.converter = r"D:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe   = r"D:\ffmpeg\bin\ffprobe.exe"

def convert_m4a_to_wav(src_file, dst_file):
    """Chuyển đổi m4a sang wav chuẩn 16kHz, mono"""
    try:
        audio = AudioSegment.from_file(src_file, format="m4a")
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        audio.export(dst_file, format="wav")
        print(f"✅ Convert thành công: {src_file} → {dst_file}")
        return True
    except Exception as e:
        print(f"❌ Convert failed: {e}")
        return False

if __name__ == "__main__":
    src = r"D:\NAM4\TTDN\test_file_6.m4a"
    dst = r"D:\NAM4\TTDN\test_file_6.wav"
    convert_m4a_to_wav(src, dst)