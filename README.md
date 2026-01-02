# Vaanika's project Replication

## Overview
Created **~10 minute** doctor–patient dental conversations for speech-to-text (ASR) testing. Text is authored to feel clinically realistic, converted to audio with TTS, transcribed with **Whisper (Groq STT API)**, cleaned, and optionally turned into SOAP-style clinical reports.

## Scenarios (≈10 minutes each)
- Diagnostic Examination
- Preventive Hygiene
- Restorative Fillings
- Endodontic / Root Canal
- Periodontal / Oral Surgery
- Prosthodontic / Implant
- Other Dental Specialties

## Workflow
1. Author conversation text in `conversations/*_reference.txt`.
2. Generate long-form audio from text (`scripts/generate_audio.py`, Edge TTS en-US-JennyNeural, speed +25%).
3. Transcribe audio with Groq Whisper (`scripts/transcribe_groq.py`, `whisper-large-v3-turbo`).
4. Clean transcripts (`scripts/transcription_cleanup.py`) to remove speaker prefixes and tidy formatting.
5. Generate SOAP reports (`scripts/report_generation.py`) from cleaned transcripts (optional downstream step).

## Project Layout
- `conversations/` – reference texts.
- `transcripts/` – raw Groq Whisper outputs.
- `transcripts_clean/` – cleaned transcripts.
- `configs/.env` – secrets and model config (not committed).
- `requirements.txt` – Python deps.
- `scripts/`:
	- `generate_audio.py` – text → audio (Edge TTS).
	- `transcribe_groq.py` – audio → text (Groq Whisper, verbose_json, language="en", prompt="Medical transcription", temperature=0.0).
	- `transcription_cleanup.py` – Groq chat cleanup pass.
	- `report_generation.py` – SOAP report generator from cleaned transcripts.
	- `utils.py` – helpers (paths, logging, chunking).

## Setup
1. Python 3.11 recommended.
2. Install deps:
	 ```bash
	 pip install -r requirements.txt
	 ```
3. Configure env (Windows cmd example):
	 ```bat
	 set GROQ_API_KEY=your_api_key_here
	 set GROQ_MODEL=whisper-large-v3-turbo
	 ```
	 Or create `configs/.env` with the same keys.

## Usage
- Generate audio (from conversations to wav; creates `audio/` if missing):
	```bash
	python scripts/generate_audio.py
	```
- Transcribe audio with Groq Whisper:
	```bash
	python scripts/transcribe_groq.py
	```
- Clean transcripts (speaker tags, formatting):
	```bash
	python scripts/transcription_cleanup.py
	```
- Generate SOAP clinical reports (optional):
	```bash
	python scripts/report_generation.py
	```

## Notes
- Audio files are long-form (~10 minutes) to mimic realistic clinical sessions.
- Transcription uses Groq-hosted Whisper; set `GROQ_API_KEY` before running.
- Cleaning and report steps are deterministic prompts tuned for dental context.

