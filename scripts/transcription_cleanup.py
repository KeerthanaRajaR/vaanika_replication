import os
import time
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = Path(__file__).parent.parent
ENV_PATH = BASE_DIR / "configs" / ".env"
load_dotenv(ENV_PATH)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 📂 folders
TRANSCRIPTS_DIR = Path("transcripts")
CLEAN_DIR = Path("transcripts_clean")

CLEAN_DIR.mkdir(exist_ok=True)

def build_cleanup_prompt(text: str) -> str:
    return f"""You are a medical transcription specialist. Please clean and correct the following medical transcription while preserving the speaker's exact point of view and ensuring all content is in English.

CRITICAL GUIDELINES:

1. Speaker Label Removal
   • REMOVE ALL speaker prefixes like "Dr. Ananya,", "Patient,", "Doctor,", etc.
   • Convert to natural dialogue format without any speaker labels
   • Maintain the conversation flow as continuous dialogue

2. Language Filtering
   • Remove ALL non-English words, phrases, or sentences
   • If the transcription contains other languages mixed with English, keep ONLY the English portions
   • Do NOT translate non-English content - simply remove it
   • Ensure the output is 100% English

3. Transcription Correction
   • Correct spelling/capitalization of medical terminology
   • Fix obvious transcription mistakes
   • Preserve the speaker's wording, tone, and grammatical person (no conversion to third person)

4. Content Preservation
   • Maintain the original meaning and medical findings
   • Use proper medical punctuation where needed without rephrasing
   • Keep the same structure and content as the original
   • Do not add information that wasn't in the original text

5. Output Format
   • Return ONLY the cleaned English text
   • No formatting symbols, bullet points, or section headers
   • Return as a single paragraph without line breaks
   • Do not add introductory text like "Here is the cleaned version"

Original transcription:
{text}

Return the cleaned English-only version as a single paragraph:"""

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
    
    print(f"✅ Completed: {txt_file.name}")
    
    # Rate limiting: wait 5 seconds between API calls
    time.sleep(5)

print("\n✅ Transcription cleanup completed")
