import os
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv
import re

# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
ENV_PATH = BASE_DIR / "configs" / ".env"
load_dotenv(ENV_PATH)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv(
    "GROQ_TEXT_POSTPROCESSING_MODEL",
    "moonshotai/kimi-k2-instruct-0905"
)

# --------------------------------------------------
# DIRECTORIES
# --------------------------------------------------
CLEAN_TRANSCRIPTS_DIR = BASE_DIR / "transcripts_clean"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# CANONICAL SOAP TEMPLATE (ALWAYS USED)
# --------------------------------------------------
CANONICAL_SOAP_TEMPLATE = """
S:
- Chief Complaint:
- History of Present Illness:
- Dental History:
- Medical History:
- Patient Goals:

O:
- Clinical Examination:
- Diagnostic Tests:
- Radiographic Findings:

A:
- Diagnosis:

P:
- Treatment Plan:
"""

# --------------------------------------------------
# NORMALIZATION HELPERS
# --------------------------------------------------
def normalize_tooth_numbers(text: str) -> str:
    # Enforce Tooth #14 format
    return re.sub(r"\btooth\s*(\d+)\b", r"Tooth #\1", text, flags=re.IGNORECASE)

def sanitize_text(text: str) -> str:
    text = normalize_tooth_numbers(text)
    return text.strip()

# --------------------------------------------------
# PROMPT BUILDER (STRICT)
# --------------------------------------------------
def build_prompt(
    cleaned_transcription: str,
    session_notes: str = ""
) -> str:

    notes_block = ""
    if session_notes.strip():
        notes_block = f"\n\nSession Notes:\n{session_notes}"

    return f"""
Generate a COMPLETE, READY-TO-USE clinical document suitable for a real dental chart and legal medical records.

CRITICAL REQUIREMENTS:

1. CONVERSATIONAL NOISE REMOVAL (HIGHEST PRIORITY)
   • REMOVE all non-clinical conversational content:
     - Food and drink mentions (coffee, steak, ice cream, soup, grilled meat, dinner)
     - Weather, traffic, family, work, lifestyle chatter
     - Humor, emotions, metaphors ("heartbeat", "regret my life choices", "insane")
     - Narrative storytelling phrases ("Friday night dinner", "we ordered", "kids were there")
     - Casual greetings and small talk ("you look tired", "work's been crazy")
   • ONLY include dietary habits if clinically relevant (e.g., under Dietary Counseling for bruxism-related habits like ice chewing, hard foods)
   • RULE: If it was said casually in conversation and does not change diagnosis or treatment → REMOVE IT
   • Transform narrative context to clinical language:
     - WRONG: "Pain started while eating grilled meat"
     - CORRECT: "Symptoms initiated during mastication"

2. STANDARDIZED TOOTH NOTATION (MANDATORY)
   • ALWAYS use format: Tooth #X (capital T, number sign, numeral)
   • Examples: Tooth #4, Tooth #30, Tooth #14
   • NEVER use:
     - "tooth number 4" (spell out)
     - "tooth four" (word form)
     - "# four" (mixed format)
     - "tooth 4" (missing #)
   • Apply consistently throughout the ENTIRE document

3. MANDATORY SOAP FORMAT ENFORCEMENT (CRITICAL - OVERRIDES TEMPLATE)
   • ALWAYS output clinical reports in SOAP format with explicit section headers
   • This is MANDATORY regardless of how the template is structured
   
   • REQUIRED SECTION HEADERS (in this exact order):
     1. **Subjective** - Patient-reported information
     2. **Objective** - Clinical examination findings
     3. **Assessment** - Diagnosis and clinical judgment
     4. **Plan** - Treatment plan and follow-up
   
   • TEMPLATE OVERRIDE RULES:
     - If template does NOT have SOAP headers → ADD them and organize content under them
     - If template has SOAP headers in WRONG order → REORDER to correct S-O-A-P sequence
     - If template fields don't map cleanly → use clinical judgment to place under correct SOAP section
   
   • FIELD MAPPING TO SOAP SECTIONS (IN THIS ORDER):
     - SUBJECTIVE (in order):
       1. Chief Complaint
       2. History of Present Illness (HPI)
       3. Dental History
       4. Medical History
       5. Patient Goals
     
     - OBJECTIVE (in order - follows clinical workflow: examine → test → image):
       1. Clinical Examination (FIRST - you examine before testing)
       2. Diagnostic Tests (SECOND - tests based on clinical findings)
       3. Radiographic Findings (THIRD - imaging supports clinical assessment)
       4. Other Observations (LAST)
       ❗ WRONG ORDER: Diagnostic Tests before Clinical Examination
       ❗ Clinical examination must ALWAYS precede diagnostic tests
     
     - ASSESSMENT (in order):
       1. Diagnosis
       2. Risk Assessment
       3. Prognosis
     
     - PLAN (in order):
       1. Treatment Plan
       2. Patient Education
       3. Follow-Up
       4. Additional Notes (if session notes provided)
   
   • WHY THIS RULE EXISTS:
     - SOAP is the legal/clinical standard for documentation
     - Sub-section order follows clinical workflow logic
     - Templates may be malformed or missing structure
     - Output must be consistent regardless of template quality

4. SUBJECTIVE SECTION HYGIENE
   • ALLOWED content:
     - Chief complaint (patient-reported symptoms)
     - Symptom timing and duration
     - Sensitivity description (cold, heat, biting, pressure)
     - Pain duration characteristics (lingering vs transient)
     - Patient goals for treatment (CLINICAL goals only)
   • NOT ALLOWED:
     - Food brands, specific meals, or eating occasions
     - Emotional reactions or mood descriptions
     - Story context ("Friday night", "at dinner", "with kids")
     - Work or lifestyle commentary
     - Personal events or deadlines (weddings, vacations, trips, etc.)
   • PATIENT GOALS GUIDELINES:
     - Focus on CLINICAL/TREATMENT goals only
     - WRONG: "Patient desires resolution of symptoms prior to upcoming wedding event next month" (lifestyle context)
     - CORRECT: "Patient desires timely resolution of symptoms"
     - WRONG: "Patient wants treatment before vacation" (personal deadline)
     - CORRECT: "Patient desires prompt treatment"
     - Personal events are NOT clinically relevant and should be excluded
   • SECTION LEAKAGE PREVENTION:
     - Subjective section must use PATIENT-REPORTED language, not clinical testing language
     - WRONG: "Thermal testing: Cold sensitivity with non-lingering response" (sounds like Objective)
     - CORRECT: "Patient reports cold sensitivity with non-lingering response"
     - WRONG: "Cold test positive" in Subjective (this is Objective data)
     - CORRECT: "Patient reports sensitivity to cold" in Subjective
   • Medical/Dental History: List ONLY clinically relevant conditions
     - WRONG: "Patient trying to lose weight" → REMOVE (unless medically relevant)
     - WRONG: "Patient drinks a ton of coffee" → REMOVE (unless relevant to treatment)

5. OBJECTIVE SECTION PRECISION
   • Include ONLY pure clinical observations
   • NO interpretation language ("likely", "probably", "seems", "appears to be")
   • NO patient quotes or conversational references
   • NO subjective descriptors from patient
   • Transform observations:
     - WRONG: "Patient felt sharp pain when biting steak"
     - CORRECT: "Pain elicited on mastication in right posterior dentition"

6. DIAGNOSTIC TEST REPORTING STANDARDS
   • Every test MUST include:
     - Test name
     - Tooth reference (using Tooth #X format)
     - Response type (positive/negative/WNL)
     - Duration if applicable (lingering/non-lingering)
   • Standard format examples:
     - "Cold test: Positive, non-lingering response on Tooth #30"
     - "Percussion: WNL on Tooth #4"
     - "Palpation: Negative"
   • REMOVE subjective adjectives:
     - WRONG: "Very painful", "Extremely sensitive", "Really bad"
     - CORRECT: Use clinical descriptors: "Positive response", "Hypersensitivity noted"

7. DIAGNOSIS LANGUAGE CONTROL
   • Use ONLY accepted dental terminology
   • Diagnosis MUST match documented findings
   • NO conversational phrasing:
     - WRONG: "Cracked filling causing weird sensation"
     - CORRECT: "Cracked amalgam restoration with reversible pulpitis"
   • Include:
     - Primary diagnosis
     - Affected tooth (Tooth #X format)
     - Clinical classification when applicable
   • DIAGNOSTIC UNCERTAINTY PRESERVATION (CRITICAL - STRUCTURAL, NOT LEXICAL):
     - Uncertainty must be STRUCTURALLY preserved, not just copied as words
     - Even copying "likely irreversible" as a definitive "Diagnosis:" reads as resolved
     - The diagnosis line itself must signal non-finality
     
     - Uncertainty indicators to watch for:
       * "likely" (e.g., "likely irreversible") - this is NOT certainty
       * "reversible vs irreversible?"
       * "possible crack"
       * "may need RCT"
       * "monitor symptoms"
       * Question marks in diagnosis
       * "concern for", "suggest", "suspect"
     
     - WRONG (reads as resolved even with "likely"):
       * "Diagnosis: Symptomatic pulpitis, likely irreversible"
       * This structurally presents as a definitive conclusion
     
     - WRONG (combines conflicting framings - escalates certainty):
       * "Diagnosis: Symptomatic pulpitis (reversible vs irreversible - to be reassessed), likely irreversible"
       * Mixing "reversible vs irreversible" (neutral) with "likely irreversible" (leaning) is contradictory
       * Choose ONE approach, not both
     
     - CORRECT (use ONE uncertainty framing only):
       * "Diagnosis: Symptomatic pulpitis (reversible vs irreversible - to be reassessed)"
       * OR "Diagnosis: Symptomatic pulpitis with clinical concern for irreversible involvement"
       * OR "Diagnosis: Symptomatic pulpitis - irreversible status not yet confirmed"
       * Pick the framing that best matches the session notes tone
     
     - CERTAINTY MATCHING RULE (CRITICAL - NEVER ESCALATE):
       * Assessment certainty must be ≤ session notes certainty
       * NEVER upgrade certainty language beyond what clinician documented
       * Match the EXACT certainty level from session notes
       
       * CERTAINTY HIERARCHY (low to high):
         possible < suspected < likely < probable < clinical concern < confirmed
       
       * ESCALATION EXAMPLES (ALL WRONG):
         - Session notes: "likely irreversible" → Assessment: "irreversible" (ESCALATION)
         - Session notes: "likely irreversible" → Assessment: "clinical concern for irreversible" (ESCALATION)
         - Session notes: "possible crack" → Assessment: "cracked" (ESCALATION)
         - Session notes: "impression" → Assessment: "diagnosis" (ESCALATION)
       
       * CORRECT CERTAINTY PRESERVATION:
         - Session notes: "likely irreversible" → Assessment: "Symptomatic pulpitis, likely irreversible"
         - Session notes: "reversible vs irreversible" → Assessment: "Symptomatic pulpitis (reversible vs irreversible - to be reassessed)"
         - Session notes: "possible crack" → Assessment: "possible crack in restoration"
       
       * "Clinical concern for X" is STRONGER than "likely X" - do NOT use it as a replacement
       * When session notes say "likely", use "likely" in assessment - not "concern for"
     
     - NO DIAGNOSIS ADDITION (CRITICAL - ZERO TOLERANCE):
       * ONLY include diagnoses that were EXPLICITLY stated by the clinician
       * Do NOT add diagnoses by inference from clinical findings
       * Presence of clinical finding ≠ diagnosed condition
       
       * KEY DISTINCTION:
         - CLINICAL FINDINGS (document what you observed)
         - DIAGNOSIS (document what clinician explicitly concluded)
         - Do NOT convert findings into diagnoses
       
       * EXPLICIT WRONG EXAMPLES:
         - Purulent exudate observed → Adding "acute apical abscess" (NOT diagnosed)
         - Pus from canal → Adding "periapical abscess" (finding ≠ diagnosis)
         - Necrotic pulp → Adding "pulp necrosis with abscess" (inference)
         - Pain at night → Adding "acute apical periodontitis" (inference)
         - Swelling present → Adding "cellulitis" (inference)
       
       * WHAT TO DO INSTEAD:
         - Document the FINDING: "Purulent exudate noted from palatal canal"
         - Document ONLY the stated DIAGNOSIS: "Irreversible pulpitis"
         - Do NOT combine finding + inferred diagnosis
       
       * CORRECT FORMAT:
         - Diagnosis: Irreversible pulpitis with necrotic pulp, Tooth #3
         - Clinical Findings: Purulent exudate from palatal canal (documented separately)
       
       * WRONG FORMAT:
         - Diagnosis: Irreversible pulpitis with necrotic pulp and acute apical abscess
         (abscess was never explicitly diagnosed - only pus was observed)
       
       * RULE: If clinician only said "irreversible pulpitis" → write "Irreversible pulpitis"
         Do NOT add "with abscess" or "with apical periodontitis" based on findings
     
     - This is critical for medico-legal accuracy - over-asserting diagnosis is clinical error

   • RADIOGRAPHIC FINDINGS SCOPE (IMPORTANT):
     - Radiographic Findings should ONLY contain imaging interpretation:
       * Periapical status (radiolucency, normal, etc.)
       * Bone levels
       * Root morphology
       * Apex visibility
       * Pre-existing conditions visible on radiograph
     - Do NOT include procedural data in Radiographic Findings:
       * WRONG: "Working length confirmed at 21 mm" (procedural measurement)
       * WRONG: "Obturation fill appears adequate with no voids" (post-procedural assessment)
       * These belong in: Procedure Performed or Objective → Clinical Findings
     - Radiographic Findings = what you SEE on the image, not what you DID during procedure

8. TREATMENT PLAN SPECIFICITY
   • Every treatment plan MUST include:
     - Clear procedure name
     - Tooth reference (Tooth #X format)
     - NO speculative language ("probably", "might", "we'll see")
   • Transform:
     - WRONG: "We'll probably just replace the filling"
     - CORRECT: "Replacement of cracked amalgam restoration on Tooth #30"

9. FOLLOW-UP & INSTRUCTIONS CONSISTENCY
   • Follow-up must be:
     - Logical and clinically appropriate
     - Specific (timeframe or condition-based)
     - NO casual reassurance language
   • Transform:
     - WRONG: "No worries, you're good"
     - CORRECT: "No follow-up required unless symptoms persist"
     - WRONG: "Let us know if anything happens"
     - CORRECT: "Return if spontaneous pain, lingering sensitivity, or swelling develops"

10. FORMATTING & LEGAL READINESS
    • Use **bold** markdown for date labels (e.g., **Date of Service:**)
    • Use ### markdown for section headers (e.g., ### Subjective, ### Objective)
    • Use hyphens (-) for bullet points within sections
    • Use **bold** for subsection labels (e.g., **Chief Complaint:**, **Clinical Findings:**)
    • Ensure:
      - All SOAP sections present and complete
      - Consistent bullet formatting throughout
      - No spelling errors
      - No colloquial words or phrases
      - Provider name with credentials at end
      - Consistent date format (Month Day, Year)
    • Write complete, grammatically correct sentences

11. CLINICAL LANGUAGE CONVERSION (CRITICAL)
    • Write EXCLUSIVELY in professional clinical third-person language
    • Convert ALL patient-style language to clinical terminology:
      * "I have pain" → "Patient reports pain"
      * "My tooth hurts when I bite" → "Patient reports pain upon mastication"
      * "It's kind of swollen" → "Mild gingival edema noted"
      * "This is killing me" → "Patient reports severe pain"
      * "It feels off" → "Patient reports altered sensation"
      * "It bugs me" → "Patient reports discomfort"
    • Use precise dental terminology (MOD amalgam, PDL widening, caries, periodontal, pulpitis)
    • REMOVE all colloquial language: "like", "kind of", "basically", "you know", "well", "actually", "stuff", "thing"
    • Do NOT use phrases like "as mentioned", "according to transcript"

12. CONTENT EXTRACTION
    • Extract clinical information directly from the transcript
    • Use session context to fill in dates, patient name, practitioner name
    • Replace ALL placeholders with actual values
    • You MAY logically infer clinical details when strongly implied
    • Do NOT invent facts that contradict the transcript

13. MANDATORY FIELD INCLUSION (CRITICAL - NEVER OMIT TEMPLATE FIELDS)
    • Every field defined in the template MUST appear in the final report
    • This is non-negotiable for legal, clinical, and software audit purposes
    
    • IF data exists in transcript or session notes:
      - Populate the field normally with extracted information
      - Example: "Pain scale: 3/10"
      - Example: "Impact on daily activities: Patient avoids chewing on affected side"
    
    • IF data does NOT exist in transcript or session notes:
      - STILL INCLUDE the field in the output
      - Explicitly state: "Not available in transcription"
      - Example: "Pain scale: Not available in transcription"
      - Example: "Impact on daily activities: Not available in transcription"
    
    • NEVER silently omit fields:
      - WRONG: (HPI contains no mention of pain scale at all) - field missing entirely
      - CORRECT: "Pain scale: Not available in transcription" - field present, status clear
    
    • NEVER invent or assume values:
      - WRONG: "Pain scale: Mild" (when not stated in transcript)
      - WRONG: "Impact on daily activities: Minimal" (hallucinated)
      - CORRECT: "Not available in transcription"
    
    • WHY THIS RULE EXISTS:
      - Legal/Audit: Auditors can distinguish "not asked" vs "asked but not documented"
      - Software: Guarantees stable document structure for parsing and validation
      - Quality: Prevents both silent omission AND hallucination
    
    • PLACEHOLDER RULES:
      - NEVER use: [DATE], [DOCTOR_NAME], [PATIENT_NAME], "TBD", "XXXX"
      - ALWAYS use: "Not available in transcription" for missing clinical data

14. MEDICAL/DENTAL HISTORY ACCURACY (AVOID FALSE NEGATIONS - STRICT)
    • Distinguish between "no changes reported" vs "no conditions exist":
      - If patient says "no changes" → write ONLY "No changes to medical history reported"
      - If patient says "no medications" → write "No medications reported"
      - If not discussed at all → write "Not available in transcription"
    • STRICT MEDICATION RULE:
      - If patient says "no changes" to medications → DO NOT add "No medications reported"
      - "No changes" means existing medications unchanged, NOT absence of medications
      - WRONG: "No changes to medical history reported. No medications reported."
      - CORRECT: "No changes to medical history reported."
    • NEVER assume absence of medications or conditions:
      - WRONG: "No medications reported" (when patient only said "no changes")
      - CORRECT: "No changes to medications reported"
    • For fields in template with undocumented information:
      - WRONG: "Last prophylaxis not documented" (introduces absence of data)
      - CORRECT: "Non-contributory" or "Not available in transcription"
    • RULE: Only state what was EXPLICITLY confirmed or denied by the patient
    • If in doubt about what patient said, use "Not available in transcription" rather than assume

15. RADIOGRAPHIC VS CLINICAL FINDINGS DISTINCTION
    • Distinguish between clinically observed vs radiographically evident findings:
      - Cracks in restorations are typically CLINICAL observations, not radiographic
      - Periapical pathology, bone loss, and internal structures are RADIOGRAPHIC findings
    • Correct attribution:
      - WRONG: "Radiographic evaluation reveals cracked amalgam restoration" (cracks are clinical)
      - CORRECT: "Radiographs reviewed. Existing amalgam restoration present on Tooth #X. Pulp chamber appears normal with no periapical pathology."
    • Clinical findings section should describe:
      - Visual observations (cracks, decay, discoloration)
      - Tactile findings (mobility, percussion response)
    • Radiographic findings should describe:
      - Bone levels, periapical status, pulp chamber size
      - Internal tooth structure, root morphology
    • This distinction is critical for medico-legal accuracy

16. CONSERVATIVE RISK ASSESSMENT LANGUAGE
    • Use evidence-based, conservative risk stratification:
      - Non-lingering cold response + no spontaneous pain + WNL percussion = LOW to MODERATE risk
      - Lingering response or spontaneous pain = MODERATE to HIGH risk
    • Avoid overstating risk when clinical findings are favorable:
      - WRONG: "Moderate risk for progression to irreversible pulpitis" (when findings are mild)
      - CORRECT: "Low to moderate risk for pulpal progression if untreated"
    • Risk language should match documented clinical findings
    • Prognosis should be realistic and based on standard dental outcomes

17. EXPLICIT FOLLOW-UP STRUCTURE
    • Follow-up instructions should include:
      - Follow-up timing (as needed, specific timeframe, or routine recall)
      - Specific return conditions
    • Standard format:
      - CORRECT: "Follow-up as needed. Patient instructed to return if spontaneous pain, lingering sensitivity, or swelling develops."
    • Avoid ambiguous or incomplete follow-up:
      - WRONG: "Return if symptoms worsen" (vague)
      - CORRECT: "Return if spontaneous pain, prolonged sensitivity to thermal stimuli, or facial swelling develops"

18. NO WNL/NEGATIVE INVENTION FOR UNPERFORMED TESTS (CRITICAL - LEGAL RISK)
    • NEVER invent "WNL" or "negative" results for tests not explicitly performed
    • Do NOT assume or auto-populate positive findings:
      - WRONG: "Extraoral examination: WNL" (unless explicitly examined)
      - WRONG: "Periodontal tissues: WNL" (unless probing/gingival exam stated)
      - WRONG: "Palpation: WNL" (unless palpation was performed)
      - WRONG: "Mobility: WNL" (unless mobility was tested)
    
    • For template-required diagnostic test fields:
      - IF test was performed → document the actual result
      - IF test was NOT performed → write "Not performed" or "Not available in transcription"
      - NEVER write "WNL" for a test that wasn't done (this is false documentation)
    
    • For clinical examination fields in template:
      - IF examination was done → document findings
      - IF examination was NOT mentioned → write "Not performed" or "Not available in transcription"
    
    • This is critical for medico-legal accuracy:
      - Documenting "WNL" for unperformed exams = false documentation
      - Documenting "Not performed" = accurate record of what was done

19. NO QUOTED CONVERSATIONAL LANGUAGE
    • NEVER quote patient speech in clinical documentation
    • Translate all patient expressions to clinical terminology:
      - WRONG: Character: Discomfort described as "feels off" during chewing
      - CORRECT: Character: Discomfort during mastication
      - WRONG: Patient states "it's killing me"
      - CORRECT: Patient reports severe pain
    • Remove quotation marks around patient descriptions
    • Clinical notes document findings, not patient quotes

20. SECTION PLACEMENT RULES
    • SUBJECTIVE section contains:
      - Patient-reported symptoms and history
      - Patient behavior (e.g., avoiding chewing on affected side)
      - Patient goals and concerns
    • OBJECTIVE section contains ONLY:
      - Clinical examination findings by the practitioner
      - Diagnostic test results
      - Radiographic findings
    • WRONG placements to avoid:
      - WRONG: "Other Observations: Patient demonstrates avoidance of right side" (move to Subjective/HPI)
      - WRONG: "Dental History: Patient reports avoiding mastication" (this is HPI behavior, not dental history)
    • Dental History should contain:
      - Prior treatments and restorations
      - Previous dental conditions
      - Last prophylaxis if mentioned
      - NOT current symptoms or behaviors
    • Dental History standard phrasing:
      - If no relevant dental history mentioned, use: "Non-contributory."
      - Do NOT repeat the section label in the content:
        - WRONG: "Dental History: Dental history: Non-contributory." (redundant prefix)
        - CORRECT: "Dental History: Non-contributory."
      - WRONG: "No changes reported." (vague)
      - CORRECT: "Non-contributory." (standard clinical phrasing)

21. SYMPTOM CONSOLIDATION (AVOID REDUNDANCY)
    • Do not repeat the same clinical finding in multiple sections
    • Specifically avoid duplicating behavior:
      - WRONG: "Patient reports avoiding mastication on affected side" in BOTH HPI AND Dental History
      - CORRECT: Include avoidance behavior ONLY in HPI, remove from Dental History
    • Consolidate related information:
      - WRONG: "Cold sensitivity, non-lingering" AND "No thermal lingering" (redundant)
      - CORRECT: "Cold sensitivity present with non-lingering response"
    • Avoid misusing HPI labels:
      - WRONG: "Timing: Cold sensitivity present with non-lingering response" (this is not timing)
      - CORRECT: "Cold sensitivity: Present with non-lingering response" OR just include as bullet without label
      - "Timing" should only refer to temporal course (when symptoms occur, duration, frequency)
    • Scope clinical findings to specific tooth when applicable:
      - WRONG: "No caries evident on clinical examination" (too broad, implies full mouth exam)
      - CORRECT: "No additional caries noted on Tooth #4" (tooth-specific, defensible)
    • Each clinical finding should appear only once in the most appropriate location

22. SESSION NOTES INCLUSION (IF PROVIDED)
    • If clinician-provided session notes are included in the input, add them to the report
    • Create a section titled "**Additional Notes**" or "**Clinician Notes**"
    • Place this section AFTER Plan and BEFORE Provider signature
    • Present the session notes content:
      - Include the notes verbatim or lightly cleaned for formatting
      - Do NOT interpret or expand the notes
      - Do NOT remove information from the notes
      - Preserve the clinician's original observations
    • If no session notes are provided, OMIT this section entirely
    • Format example:
      **Additional Notes**
      [Clinician-provided session notes content here]

      **Provider:** Name, DDS

23. OUTPUT FORMAT
    • Output ONLY the completed clinical document
    • No explanations before or after
    • No markdown code fences (\`\`\`)
    • No meta commentary
    • Start directly with **Date of Service:** line
    • End with provider information and credentials

Cleaned Transcription:
{cleaned_transcription}
{notes_block}

Required SOAP Template:
{CANONICAL_SOAP_TEMPLATE}
"""

# --------------------------------------------------
# REPORT GENERATION
# --------------------------------------------------
def generate_report(topic: str, session_notes: str = ""):
    transcript_path = CLEAN_TRANSCRIPTS_DIR / f"{topic}.txt"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Missing cleaned transcript: {transcript_path}")

    cleaned_transcription = sanitize_text(
        transcript_path.read_text(encoding="utf-8")
    )

    prompt = build_prompt(cleaned_transcription, session_notes)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    report_text = response.choices[0].message.content.strip()

    output_path = REPORTS_DIR / f"{topic}_soap_report.txt"
    output_path.write_text(report_text, encoding="utf-8")

    print(f"✅ Report generated: {output_path}")
    return report_text

# --------------------------------------------------
# BATCH RUNNER
# --------------------------------------------------
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

    for topic in topics:
        try:
            generate_report(topic)
        except Exception as e:
            print(f"❌ {topic} failed: {e}")

    print("\n✅ All SOAP reports generated successfully")
