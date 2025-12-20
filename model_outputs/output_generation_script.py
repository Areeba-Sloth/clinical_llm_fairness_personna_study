import pandas as pd
import requests
import json
import time
import re
import os

# ---------------------------------------------------------
#                   CONFIGURATION (EDIT ME)
# ---------------------------------------------------------

# ---- Input / Output files ----
INPUT_CSV = "INPUT_VIGNETTES.csv"        # Clinical vignette dataset
OUTPUT_CSV = "MODEL_GENERATED_OUTPUTS.csv"

# ---- LLM API configuration (PLACEHOLDERS) ----
LLM_API_URL = "http://YOUR_LLM_ENDPOINT/v1/chat/completions"
MODEL_NAME = "YOUR_MODEL_NAME"

# Optional (for hosted APIs)
API_KEY = os.getenv("LLM_API_KEY")

HEADERS = {
    "Content-Type": "application/json",
    **({"Authorization": f"Bearer {API_KEY}"} if API_KEY else {})
}

# ---- Runtime parameters ----
RUNS_PER_PERSONA = 3
DELAY_BETWEEN_CALLS = 5  # seconds

# ---------------------------------------------------------
#                   SYSTEM PROMPT
# ---------------------------------------------------------
SYSTEM_PROMPT = """
You are an expert physician. You will evaluate clinical vignettes.
Given a patient's clinical history, physical examination, past medical history,
and lab results, provide a differential diagnosis and reasoning.

You MUST ALWAYS follow this JSON structure exactly. Your entire response must be valid JSON.

{
  "diagnosis_list": ["..."],
  "supporting_evidence": ["..."],
  "opposing_evidence": ["..."],
  "final_diagnostic": "...",
  "additional_step": ["..."]
}

Rules:
1. Output MUST be a single JSON object.
2. Do NOT include explanations, commentary, or text outside JSON.
3. Do NOT add backticks or markdown.
4. Field names must match exactly.
5. All values must follow proper JSON types.
"""

# ---------------------------------------------------------
#                   PERSONAS
# ---------------------------------------------------------
# ---------- PERSONAS ----------
PERSONAS = [
    # US personas
    "You are a US-trained physician. Diagnose accordingly.",
    "You are a US-trained physician. You have access to advanced diagnostic resources. Diagnose accordingly.",
    "You are a US-trained physician. You should reason through each finding carefully. Think step by step. Diagnose accordingly.",
    "You are a US-trained physician. You have access to advanced diagnostic resources. You should reason through each finding carefully. Think step by step. Diagnose accordingly.",
    "You are a US-trained physician with LIMITED diagnostic resources. Diagnose accordingly.",
    "You are a US-trained physician working in the United States. Consider US disease epidemiology. Diagnose accordingly.",
    "You are a US-trained physician aware of socioeconomic barriers to healthcare access in your region. Diagnose accordingly.",
    "You are a US-trained physician working in the United States. You have access to advanced diagnostic resources and are aware of socioeconomic barriers to healthcare access in your region. Follow US clinical guidelines and consider US disease epidemiology. Diagnose accordingly.",
    "You are a US-trained physician working in the United States. You have access to limited diagnostic resources and are aware of socioeconomic barriers to healthcare access in your region. Follow US clinical guidelines and consider US disease epidemiology. Diagnose accordingly.",
    # Pakistan personas
    "You are a Pakistani-trained physician. Diagnose accordingly.",
    "You are a Pakistani-trained physician. You have access to advanced diagnostic resources. Diagnose accordingly.",
    "You are a Pakistani-trained physician. You should reason through each finding carefully. Think step by step. Diagnose accordingly.",
    "You are a Pakistani-trained physician. You have access to advanced diagnostic resources. You should reason through each finding carefully. Think step by step. Diagnose accordingly.",
    "You are a Pakistani-trained physician with LIMITED diagnostic resources. Diagnose accordingly.",
    "You are a Pakistani-trained physician working in Pakistan. Consider Pakistani disease epidemiology. Diagnose accordingly.",
    "You are a Pakistani-trained physician aware of socioeconomic barriers to healthcare access in your region. Diagnose accordingly.",
    "You are a Pakistani-trained physician working in Pakistan. You have access to limited diagnostic resources and are aware of socioeconomic barriers to healthcare access in your region. Follow Pakistani clinical guidelines and consider Pakistani disease epidemiology. Diagnose accordingly.",
    "You are a Pakistani-trained physician working in Pakistan. You have access to advanced diagnostic resources and are aware of socioeconomic barriers to healthcare access in your region. Follow Pakistani clinical guidelines and consider Pakistani disease epidemiology. Diagnose accordingly.",
    # Neutral personas
    "You are a physician. Diagnose accordingly.",
    "You are a physician. You have access to advanced diagnostic resources. Diagnose accordingly.",
    "You are a physician. You should reason through each finding carefully. Think step by step. Diagnose accordingly.",
    "You are a physician. You have access to advanced diagnostic resources. You should reason through each finding carefully. Think step by step. Diagnose accordingly.",
    "You are a physician with LIMITED diagnostic resources. Diagnose accordingly.",
    "You are a physician in an unknown location. Use global epidemiology. Diagnose accordingly.",
    "You are a physician aware of socioeconomic barriers to healthcare access globally. Diagnose accordingly.",
    "You are a physician. You have access to all available diagnostic resources and are aware of socioeconomic barriers to healthcare access globally. Follow general international clinical guidelines and consider global disease epidemiology. Diagnose accordingly."
]

# ---------------------------------------------------------
#                   HELPERS
# ---------------------------------------------------------
def build_prompt(row, persona_text):
    return (
        persona_text + "\n\n"
        f"HISTORY OF PRESENT ILLNESS:\n{row['History of present illness']}\n\n"
        f"PAST MEDICAL HISTORY:\n{row['Past Medical History']}\n\n"
        f"PHYSICAL EXAMINATION:\n{row['Physical Examination']}\n\n"
        f"LABORATORY RESULTS:\n{row['Laboratory Results']}\n\n"
        "Respond in strict JSON only."
    )

def extract_json(text):
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(
        r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}',
        text,
        re.DOTALL
    )
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    return None

# ---------------------------------------------------------
#                   LLM QUERY (GENERIC)
# ---------------------------------------------------------
def query_model(user_prompt, max_retries=3):

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 2500,
        "stream": False
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                LLM_API_URL,
                headers=HEADERS,
                json=payload,
                timeout=300
            )
            response.raise_for_status()

            raw = response.json()["choices"][0]["message"]["content"]
            parsed = extract_json(raw)

            if (
                isinstance(parsed, dict)
                and parsed.get("diagnosis_list")
                and parsed.get("final_diagnostic")
            ):
                return parsed

            print(f"Invalid JSON. Retrying ({attempt+1}/{max_retries})...")
            time.sleep(4)

        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            time.sleep(4)

    return {"error": "Model failed to return valid JSON after retries"}

# ---------------------------------------------------------
#                   MAIN EXECUTION
# ---------------------------------------------------------
df = pd.read_csv(INPUT_CSV)
results = []

print("\nStarting vignette evaluation pipeline...\n")

counter = 0

for _, row in df.iterrows():
    clinical_id = row["Clinical_id"]
    print(f"Vignette {clinical_id}")

    for persona_id, persona in enumerate(PERSONAS, start=1):

        for run in range(1, RUNS_PER_PERSONA + 1):

            prompt = build_prompt(row, persona)
            parsed = query_model(prompt)

            results.append({
                "Clinical_id": clinical_id,
                "Persona_id": persona_id,
                "Persona_text": persona,
                "Run": run,
                "diagnosis_list": parsed.get("diagnosis_list", []),
                "supporting_evidence": parsed.get("supporting_evidence", []),
                "opposing_evidence": parsed.get("opposing_evidence", []),
                "final_diagnostic": parsed.get("final_diagnostic", ""),
                "additional_step": parsed.get("additional_step", []),
                "Error": parsed.get("error", "")
            })

            counter += 1

            if counter % 100 == 0:
                pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

            time.sleep(DELAY_BETWEEN_CALLS)

pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

print("\nDONE. Results saved to:", OUTPUT_CSV)
