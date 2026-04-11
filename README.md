# Personas

A probabilistic persona generator and LLM survey runner. Generates statistically realistic
synthetic Americans based on U.S. Census and Gallup demographic data, optionally scores them
on political values and policy preferences, then feeds them to an LLM to answer survey questions
as that persona.

---

## How it works

**Persona generation** (`main.py`) samples demographic attributes (age, gender, ethnicity,
sexual orientation, political party) from conditional probability distributions derived from
real polling data. It then scores each persona on 18 political issues using a product-of-experts
model — the more demographic signals that apply to an issue, the more strongly the persona's
score is pulled in that direction, with randomness added within a stance bucket.

**Survey running** (`survey.py`) takes a saved batch of personas, builds a system prompt
describing each one, and calls an OpenAI model to answer multiple-choice survey questions as
that persona. Results are saved to a JSONL file.

---

## Setup

**Requirements:** Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

Add your OpenAI API key in a `.env` file in this directory:

```
OPENAI_API_KEY=sk-...
```

Get a key at platform.openai.com/api-keys. The `.env` file is git-ignored.

---

## Generating personas

```bash
python main.py generate [options]
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `-n` | `10` | Number of personas to generate |
| `--fix attr=value` | — | Pin one demographic attribute |
| `--fix2 attr=value` | — | Pin a second demographic attribute |
| `--type 1\|2\|3` | `3` | Output type (see below) |
| `--seed N` | — | RNG seed for reproducibility |
| `--no-save` | off | Print to stdout only, skip saving a run file |

**Output types:**

| Type | Contains | Use when |
|---|---|---|
| `1` | demographics only | Fastest; no values/wants needed |
| `2` | values + wants only | Demographics used internally but hidden from output |
| `3` | demographics + values + wants | Full persona (default) |

**Examples:**

```bash
# 100 random personas, full output
python main.py generate -n 100

# 50 Republican personas
python main.py generate -n 50 --fix party=republican

# 30 young female Democrats
python main.py generate -n 30 --fix party=democrat --fix2 gender=female

# Demographics only, no API credits needed for scoring
python main.py generate -n 100 --type 1

# Reproducible run
python main.py generate -n 10 --seed 42
```

**Valid attribute values:**

| Attribute | Values |
|---|---|
| `age` | `18-29`, `30-44`, `45-64`, `65+` |
| `gender` | `male`, `female` |
| `ethnicity` | `hispanic`, `white`, `black`, `asian`, `native american/pacific islander`, `other` |
| `orientation` | `straight`, `gay`, `bi`, `other` |
| `party` | `republican`, `democrat`, `independent`, `other` |

**Managing runs:**

Each `generate` call auto-saves to `runs/run_NNN.jsonl` and increments the number.

```bash
# List all saved runs with persona counts
python main.py list

# Delete a specific run
python main.py clear run_001

# Delete all runs
python main.py clear --all
```

**Output format** (one JSON object per line):

```json
{
  "id": "p0001",
  "demographics": {
    "age": "18-29", "gender": "female", "ethnicity": "white",
    "orientation": "straight", "party": "democrat"
  },
  "values": {
    "abortion_rights": 0.847, "gun_rights": -0.612, "lgbtq_rights": 0.731,
    "free_speech": 0.290, "nationalism": -0.445, "religious_freedom": -0.318,
    "immigration_restriction": -0.534, "affirmative_action": 0.612
  },
  "wants": {
    "student_debt_relief": 0.923, "universal_healthcare": 0.801,
    "public_safety_spending": 0.243, "climate_action": 0.788,
    "lower_taxes": -0.201, "affordable_housing": 0.914,
    "social_security_expansion": 0.672, "military_spending": -0.134,
    "minimum_wage_increase": 0.845, "school_choice": 0.312
  }
}
```

Scores range from `-1.0` (strongly oppose) to `+1.0` (strongly support).

---

## Running a survey

```bash
python survey.py --run runs/run_001.jsonl --survey surveys/immigration_survey.yaml
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--run` | required | Path to a run JSONL file |
| `--survey` | required | Path to a survey YAML file |
| `--out` | auto | Output path (default: `results/<run>_<survey>_results.jsonl`) |
| `--model` | `gpt-4o-mini` | OpenAI model to use |
| `--temperature` | `0.7` | LLM sampling temperature |
| `--delay` | `0.5` | Seconds between API calls |
| `--limit N` | — | Only run the first N personas (useful for testing) |
| `--offset N` | — | Skip the first N personas and append to an existing results file (for resuming interrupted runs) |
| `--no-cot` | off | Disable chain-of-thought reasoning (answers only, no reasoning step) |

**Example:**

```bash
# Test with 3 personas before running the full batch
python survey.py --run runs/run_001.jsonl --survey surveys/immigration_survey.yaml --limit 3

# Full run (chain-of-thought on by default)
python survey.py --run runs/run_001.jsonl --survey surveys/immigration_survey.yaml

# Disable chain-of-thought (answers only)
python survey.py --run runs/run_001.jsonl --survey surveys/immigration_survey.yaml --no-cot
```

Results are written incrementally — if the run is interrupted, completed personas are preserved.

**Output format** (one JSON object per line in `results/`):

```json
{
  "persona_id": "p0001",
  "answers": {"1": "d", "2": "b", "3": "c", "4": "a", "5": "d", "6": "a", "7": "b"},
  "raw_response": "1: d\n2: b\n3: c\n4: a\n5: d\n6: a\n7: b",
  "error": null
}
```

**Chain-of-thought mode (on by default, disable with `--no-cot`)**

Instructs the model to briefly state how the persona would frame each question before
committing to an answer. This surfaces the persona's reasoning before RLHF defaults can
override it on the answer tokens, keeping the persona frame active throughout the response.

The model is asked to write a `REASONING:` section followed by an `ANSWERS:` block. Only
the `ANSWERS:` block is parsed for letter choices, so prose in the reasoning section cannot
accidentally influence parsing. A `"reasoning"` field is added to each output record:

```json
{
  "persona_id": "p0001",
  "answers": {"1": "d", "2": "b", "3": "c"},
  "raw_response": "...(full CoT + ANSWERS block)...",
  "reasoning": "1. As a strong Republican who strongly opposes immigration...\n2. ...",
  "error": null
}
```

**Managing results:**

```bash
# List all result files with record counts
python main.py list-results

# Delete a specific result file (extension optional)
python main.py clear-results run_001_immigration_survey_results

# Delete multiple result files at once
python main.py clear-results run_001_immigration_survey_results run_002_immigration_survey_results

# Delete all result files
python main.py clear-results --all
```

---

## Analyzing results

```bash
python analyze.py --results results/run_001_immigration_survey_results.jsonl \
  --survey surveys/immigration_survey.yaml
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--results` | required | Path to a results JSONL file |
| `--survey` | required | Path to the survey YAML used to generate the results |
| `--out` | auto | Output PNG path (default: same directory as results file) |
| `--percent` | off | Show percentages instead of raw counts |

**Example:**

```bash
# Generate a chart showing percentages
python analyze.py --results results/run_001_immigration_survey_results.jsonl \
  --survey surveys/immigration_survey.yaml --percent
```

Produces a multi-panel PNG with one bar chart per question showing the distribution of answers across all personas.

---

## Adding a survey

Create a YAML file in `surveys/`:

```yaml
title: "Survey Title"
source: "Source organization"

questions:
  - id: 1
    text: "Question text here."
    choices:
      a: "First choice"
      b: "Second choice"
      c: "Third choice"
      d: "Fourth choice"
  - id: 2
    ...
```

---

## Data files

| File | Description |
|---|---|
| `data/demographics.yaml` | Marginal and conditional probability tables for demographic generation (U.S. Census 2024, Gallup) |
| `data/values.yaml` | 8 political/moral issues with demographic-conditional stance distributions |
| `data/wants.yaml` | 10 policy preference issues with demographic-conditional stance distributions |
| `surveys/` | Survey files in YAML format |
| `runs/` | Saved persona batches (git-ignored) |
| `results/` | Survey results (git-ignored) |

---

## Project layout

```
personas/
├── main.py               # Persona generator and run manager
├── survey.py             # LLM survey runner
├── analyze.py            # Results visualization
├── requirements.txt      # Python dependencies
├── data/
│   ├── demographics.yaml
│   ├── values.yaml
│   └── wants.yaml
├── surveys/
│   └── immigration_survey.yaml
├── runs/                 # Auto-created, git-ignored
└── results/              # Auto-created, git-ignored
```
