import os
from dotenv import load_dotenv

# Load .env from configs folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, "configs", ".env")
load_dotenv(ENV_PATH)

# ===== ENV VARIABLES =====
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")

# ===== PROJECT PATHS =====
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "transcripts")
CONVERSATION_DIR = os.path.join(BASE_DIR, "conversations")

# ===== ENSURE DIRS EXIST =====
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

def validate_env():
    if not GROQ_API_KEY:
        raise EnvironmentError("❌ GROQ_API_KEY not found in config/.env")

    print("✅ Environment loaded successfully")
