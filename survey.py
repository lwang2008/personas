import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
except ImportError:
    print("pyyaml required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

# Load .env from the same directory as this script, silently skip if missing
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; fall back to os.environ directly

try:
    from openai import OpenAI
except ImportError:
    print("openai package required: pip install openai", file=sys.stderr)
    sys.exit(1)


# -----------------------
# Loaders
# -----------------------

def load_personas(path: str) -> List[Dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def parse_survey(path: str) -> List[Dict]:
    """Load a survey from a YAML file. Returns list of {id, text, choices}."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [
        {
            "id":      q["id"],
            "text":    q["text"],
            "choices": q["choices"],
        }
        for q in data["questions"]
    ]


# -----------------------
# Prompt builders
# -----------------------

_DEMO_LABELS = {
    "age": "Age",
    "gender": "Gender",
    "ethnicity": "Ethnicity",
    "orientation": "Sexual orientation",
    "party": "Political party",
    "party_strength": "Party attachment",
}

_PARTY_STRENGTH_LABELS = {
    "strong":     "strong partisan (votes in every primary; highly consistent party-line views)",
    "not_strong": "soft partisan (registered with party but cross-pressured on some issues)",
    "lean":       "independent leaner (self-identifies as independent; tilts toward this party but diverges often)",
}

_VALUE_LABELS = {
    "abortion_rights":       "Abortion rights",
    "gun_rights":            "Gun rights",
    "lgbtq_rights":          "LGBTQ+ rights",
    "free_speech":           "Free speech",
    "nationalism":           "Nationalism / America first",
    "religious_freedom":     "Religious freedom",
    "immigration_restriction": "Immigration restriction",
    "affirmative_action":    "Affirmative action",
}

_WANT_LABELS = {
    "student_debt_relief":       "Student debt relief",
    "universal_healthcare":      "Universal healthcare",
    "public_safety_spending":    "Police / public safety spending",
    "climate_action":            "Climate action",
    "lower_taxes":               "Lower taxes",
    "affordable_housing":        "Affordable housing",
    "social_security_expansion": "Social Security expansion",
    "military_spending":         "Military spending",
    "minimum_wage_increase":     "Minimum wage increase",
    "school_choice":             "School choice / vouchers",
}


def _score_label(v: float) -> str:
    if v >= 0.60:  return "strongly supports"
    if v >= 0.20:  return "supports"
    if v >= -0.20: return "neutral/mixed views"
    if v >= -0.60: return "opposes"
    return "strongly opposes"


def persona_to_system_prompt(persona: Dict) -> str:
    parts = [
        "This is an academic social science simulation. You are roleplaying a specific survey respondent.",
        "Researchers are studying how people with different backgrounds and values respond to survey questions.",
        "",
        "Rules:",
        "- Answer every question exactly as this person would, based solely on their profile below.",
        "- Stay fully in character on ALL topics, including sensitive political, social, or moral issues.",
        "- Do NOT moderate, soften, or redirect answers to seem more neutral or acceptable.",
        "- The value and preference scores are authoritative ground truth about this person's views.",
        "  Translate them faithfully: a score near -1.0 means strong opposition; near +1.0 means strong support.",
        "- A persona who strongly opposes something will pick the most opposing answer available, not a moderate one.",
        "",
    ]

    if persona.get("demographics"):
        parts.append("**Demographics:**")
        for key, label in _DEMO_LABELS.items():
            val = persona["demographics"].get(key)
            if val is None:
                continue
            if key == "party_strength":
                expanded = _PARTY_STRENGTH_LABELS.get(val, val)
                parts.append(f"- {label}: {expanded}")
            else:
                parts.append(f"- {label}: {val}")
        parts.append("")

    if persona.get("values"):
        parts.append("**Values** (score from -1.0 = strongly oppose to +1.0 = strongly support):")
        for key, label in _VALUE_LABELS.items():
            if key in persona["values"]:
                v = persona["values"][key]
                parts.append(f"- {label}: {_score_label(v)} ({v:+.2f})")
        parts.append("")

    if persona.get("wants"):
        parts.append("**Policy preferences** (score from -1.0 = strongly oppose to +1.0 = strongly support):")
        for key, label in _WANT_LABELS.items():
            if key in persona["wants"]:
                v = persona["wants"][key]
                parts.append(f"- {label}: {_score_label(v)} ({v:+.2f})")

    return "\n".join(parts)


def questions_to_user_prompt(questions: List[Dict], cot: bool = False) -> str:
    if cot:
        lines = [
            "Answer each question below as the specific person described in your profile.",
            "Pick the answer that best matches their values and preferences — even on sensitive topics.",
            "",
            "Step 1 — REASONING:",
            "Before answering, briefly state how this persona would frame each question based on their",
            "values and demographics. One or two sentences per question is enough.",
            "",
            "Step 2 — ANSWERS:",
            "After your reasoning, output a line containing only the word ANSWERS: followed by",
            "your chosen letters, one per line, in the format exactly: \"1: a\"",
            "",
        ]
    else:
        lines = [
            "Answer each question below as the specific person described in your profile.",
            "Pick the answer that best matches their values and preferences — even on sensitive topics.",
            "Respond with ONLY the question number and your chosen letter, one per line.",
            'Format exactly: "1: a"  — no explanations, no extra text.',
            "",
        ]
    for q in questions:
        lines.append(f"{q['id']}. {q['text']}")
        for letter, text in q["choices"].items():
            lines.append(f"    {letter}. {text}")
        lines.append("")
    return "\n".join(lines)


# -----------------------
# Response parsing
# -----------------------

def split_cot_response(raw: str) -> tuple[str, str]:
    """Split a CoT response into (reasoning, answers_block).

    Looks for a line that starts with 'ANSWERS:' (case-insensitive) and
    splits there.  Returns (reasoning, answers_block); if the marker is
    absent both parts receive the full text so fallback parsing still works.
    """
    marker = re.search(r'(?im)^[ \t]*ANSWERS\s*:', raw)
    if marker:
        reasoning = raw[:marker.start()].strip()
        answers_block = raw[marker.end():].strip()
        return reasoning, answers_block
    return raw.strip(), raw.strip()


def parse_answers(raw: str, question_ids: List[int], cot: bool = False) -> Dict[int, Optional[str]]:
    answers: Dict[int, Optional[str]] = {qid: None for qid in question_ids}
    text = split_cot_response(raw)[1] if cot else raw
    for line in text.strip().splitlines():
        m = re.match(r'^\s*(\d+)\s*[:\.\)]\s*([a-dA-D])', line)
        if m:
            qid = int(m.group(1))
            if qid in answers:
                answers[qid] = m.group(2).lower()
    return answers


# -----------------------
# API call with retry
# -----------------------

def call_with_retry(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_retries: int = 3,
) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/{max_retries} (error: {e}, waiting {wait}s)",
                  file=sys.stderr)
            time.sleep(wait)


# -----------------------
# CLI
# -----------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run a survey against saved personas using an LLM"
    )
    parser.add_argument("--run",    required=True,
                        help="Path to a run JSONL file, e.g. runs/run_001.jsonl")
    parser.add_argument("--survey", required=True,
                        help="Path to a survey YAML file, e.g. surveys/immigration_survey.yaml")
    parser.add_argument("--out",    default=None,
                        help="Output JSONL path (default: results/<run>_<survey>_results.jsonl)")
    parser.add_argument("--model",  default="gpt-4o-mini",
                        help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--delay",  type=float, default=0.5,
                        help="Seconds between API calls (default: 0.5)")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Only run the first N personas (useful for testing)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip the first N personas and append to existing results")
    parser.add_argument("--no-cot", action="store_true",
                        help="Disable chain-of-thought reasoning (answers only, no reasoning step)")
    args = parser.parse_args()

    # --- API key ---
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: OPENAI_API_KEY not found.\n"
            "  Option 1: copy .env.example → .env and add your key\n"
            "  Option 2: export OPENAI_API_KEY=sk-... in your shell",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # --- Load data ---
    personas  = load_personas(args.run)
    if args.offset:
        personas = personas[args.offset :]
    if args.limit:
        personas = personas[: args.limit]
    questions = parse_survey(args.survey)
    q_ids     = [q["id"] for q in questions]

    if not questions:
        print(f"Error: no questions parsed from {args.survey}", file=sys.stderr)
        sys.exit(1)

    # --- Output path ---
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    if args.out:
        out_path = Path(args.out)
    else:
        run_stem    = Path(args.run).stem
        survey_stem = Path(args.survey).stem.lower().replace(" ", "_")
        out_path    = results_dir / f"{run_stem}_{survey_stem}_results.jsonl"

    cot = not args.no_cot
    # Pre-build the user prompt (same for every persona)
    user_prompt = questions_to_user_prompt(questions, cot=cot)

    file_mode = "a" if args.offset else "w"

    print(f"survey  : {args.survey} ({len(questions)} questions)", file=sys.stderr)
    print(f"personas: {len(personas)} from {args.run}"
          + (f" (skipping first {args.offset})" if args.offset else ""), file=sys.stderr)
    print(f"model   : {args.model}  temp={args.temperature}", file=sys.stderr)
    print(f"output  : {out_path} (mode={file_mode})", file=sys.stderr)
    print("", file=sys.stderr)

    # --- Run ---
    results = []
    with out_path.open(file_mode, encoding="utf-8") as f:
        for i, persona in enumerate(personas, 1):
            pid = persona.get("id", f"p{i:04d}")
            print(f"[{i}/{len(personas)}] {pid} ...", end=" ", file=sys.stderr, flush=True)

            raw       = None
            reasoning = None
            answers   = {qid: None for qid in q_ids}
            error     = None

            try:
                system_prompt = persona_to_system_prompt(persona)
                raw           = call_with_retry(
                    client, system_prompt, user_prompt,
                    args.model, args.temperature,
                )
                if cot:
                    reasoning, _ = split_cot_response(raw)
                answers = parse_answers(raw, q_ids, cot=cot)
                n_parsed = sum(1 for v in answers.values() if v is not None)
                print(f"ok ({n_parsed}/{len(q_ids)} answered)", file=sys.stderr)
            except Exception as e:
                error = str(e)
                print(f"FAILED — {error}", file=sys.stderr)

            record = {
                "persona_id":   pid,
                "answers":      {str(k): v for k, v in answers.items()},
                "raw_response": raw,
                "error":        error,
            }
            if cot:
                record["reasoning"] = reasoning
            f.write(json.dumps(record) + "\n")
            f.flush()
            results.append(record)

            if i < len(personas) and args.delay > 0:
                time.sleep(args.delay)

    n_ok = sum(1 for r in results if r["error"] is None)
    print(f"\ndone — {n_ok}/{len(personas)} successful → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
