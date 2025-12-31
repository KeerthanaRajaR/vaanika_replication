import os
from groq import Groq
from pathlib import Path

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 📂 folders
TRANSCRIPTS_DIR = Path("transcripts")
CLEAN_DIR = Path("transcripts_clean")

CLEAN_DIR.mkdir(exist_ok=True)

def build_cleanup_prompt(text: str) -> str:
    return f"""
You are a medical transcription specialist.

Clean and correct the following transcription:
- Fix grammar and punctuation
- Correct obvious ASR errors
- Normalize medical terminology
- Remove filler words and repetitions
- DO NOT add new medical facts
- DO NOT change clinical meaning
- Return ONLY the cleaned text as a single paragraph

Original transcription:
{text}
"""

for txt_file in TRANSCRIPTS_DIR.glob("*.txt"):
    print(f"🧹 Cleaning: {txt_file.name}")

    raw_text = txt_file.read_text(encoding="utf-8")

    response = client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct-0905",
        messages=[
            {"role": "user", "content": build_cleanup_prompt(raw_text)}
        ],
        temperature=0.1
    )

    cleaned_text = response.choices[0].message.content.strip()

    output_path = CLEAN_DIR / txt_file.name
    output_path.write_text(cleaned_text, encoding="utf-8")

print("\n✅ Transcription cleanup completed")
