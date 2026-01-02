import os
import wave
import shutil
import time
from groq import Groq
from utils import (
    AUDIO_DIR,
    TRANSCRIPT_DIR,
    GROQ_API_KEY,
    GROQ_STT_MODEL,
    validate_env
)

validate_env()
client = Groq(api_key=GROQ_API_KEY)

# Split WAV into chunks (seconds) and return list of chunk file paths
def split_wav(input_path, seconds=60):
    chunks = []
    wav = wave.open(input_path, 'rb')
    params = wav.getparams()
    n_channels = wav.getnchannels()
    sampwidth = wav.getsampwidth()
    framerate = wav.getframerate()
    n_frames = wav.getnframes()
    duration = n_frames / float(framerate)

    frames_per_chunk = int(framerate * seconds)
    base = os.path.splitext(os.path.basename(input_path))[0]
    tmp_dir = os.path.join(os.path.dirname(input_path), f"{base}_chunks")
    os.makedirs(tmp_dir, exist_ok=True)

    wav.rewind()
    chunk_index = 0
    while True:
        frames = wav.readframes(frames_per_chunk)
        if not frames:
            break
        chunk_path = os.path.join(tmp_dir, f"{base}_{chunk_index}.wav")
        out = wave.open(chunk_path, 'wb')
        out.setnchannels(n_channels)
        out.setsampwidth(sampwidth)
        out.setframerate(framerate)
        out.writeframes(frames)
        out.close()
        chunks.append(chunk_path)
        chunk_index += 1

    wav.close()
    return chunks


def transcribe(topic, chunk_seconds=60, max_retries=3, retry_delay=2):
    audio_path = os.path.join(AUDIO_DIR, f"{topic}.wav")
    out_path = os.path.join(TRANSCRIPT_DIR, f"{topic}.txt")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Missing audio file: {audio_path}")

    print(f"📝 Transcribing {topic} using Groq Whisper ({GROQ_STT_MODEL})")

    # Determine duration and decide whether to chunk
    try:
        with wave.open(audio_path, 'rb') as w:
            framerate = w.getframerate()
            n_frames = w.getnframes()
            duration = n_frames / float(framerate)
    except wave.Error:
        duration = None

    parts = []
    chunk_paths = None
    try:
        if duration and duration > chunk_seconds:
            print(f"🔪 Splitting audio ({duration:.1f}s) into {chunk_seconds}s chunks")
            chunk_paths = split_wav(audio_path, seconds=chunk_seconds)
            targets = chunk_paths
        else:
            targets = [audio_path]

        for idx, path in enumerate(targets):
            for attempt in range(1, max_retries + 1):
                try:
                    with open(path, 'rb') as audio:
                        # Exact Groq API implementation as specified
                        result = client.audio.transcriptions.create(
                            file=audio,                           # Audio blob
                            model=GROQ_STT_MODEL,                # STT model (whisper-large-v3-turbo)
                            response_format="verbose_json",      # verbose_json format
                            language="en",                       # Always English
                            prompt="Medical transcription",      # Optional prompt for better accuracy
                            temperature=0.0                      # Optional temperature for consistency
                        )
                    parts.append(result.text)
                    print(f"✅ Chunk {idx+1}/{len(targets)} transcribed")
                    break
                except Exception as e:
                    print(f"⚠️ Transcription attempt {attempt} failed for chunk {idx+1}: {e}")
                    if attempt < max_retries:
                        time.sleep(retry_delay * attempt)
                    else:
                        raise

        # Combine parts and save
        combined = "\n\n".join(parts)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(combined)

        print(f"✅ Transcript saved: {out_path}")

    finally:
        # cleanup chunk files
        if chunk_paths:
            shutil.rmtree(os.path.dirname(chunk_paths[0]), ignore_errors=True)


if __name__ == "__main__":
    topics = [
        "diagnostic_exam",
        "preventive_hygiene",
        "restorative_fillings",
        "endodontic_root_canal",
        "periodontal_oral_surgery",
        "prosthodontic_implant",
        "other_specialty"
    ]

    for t in topics:
        transcribe(t)
