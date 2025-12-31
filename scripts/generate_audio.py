import asyncio
import edge_tts
from pathlib import Path

CONVERSATION_DIR = Path("conversations")
AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

VOICE = "en-US-JennyNeural"   # fast & clear

async def generate():
    for txt_file in CONVERSATION_DIR.glob("*_reference.txt"):
        topic = txt_file.stem.replace("_reference", "")
        audio_path = AUDIO_DIR / f"{topic}.wav"

        print(f"⚡ Generating FAST audio for: {topic}")

        text = txt_file.read_text(encoding="utf-8")

        communicate = edge_tts.Communicate(
            text=text,
            voice=VOICE,
            rate="+25%"      # 🔥 faster speaking
        )

        await communicate.save(str(audio_path))

    print("\n✅ FAST audio generation completed")

asyncio.run(generate())
