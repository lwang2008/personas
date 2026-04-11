import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List

try:
    import yaml
except ImportError:
    raise SystemExit("pyyaml required: pip install pyyaml")

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    raise SystemExit("matplotlib required: pip install matplotlib")


# -----------------------
# Loaders
# -----------------------

def load_results(path: str) -> List[Dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def load_survey(path: str) -> Dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


# -----------------------
# Counting
# -----------------------

def count_answers(results: List[Dict], question_ids: List[int]) -> Dict[int, Counter]:
    counts = {qid: Counter() for qid in question_ids}
    for r in results:
        for qid_str, answer in r.get("answers", {}).items():
            qid = int(qid_str)
            if qid in counts and answer is not None:
                counts[qid][answer] += 1
    return counts


# -----------------------
# Plotting
# -----------------------

# Muted qualitative palette — works for up to 6 choices
_COLORS = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"]


def _truncate(text: str, n: int) -> str:
    return text[:n] + "..." if len(text) > n else text


def plot_results(
    counts: Dict[int, Counter],
    questions: List[Dict],
    survey_title: str,
    out_path: Path,
    use_percent: bool = False,
) -> None:
    n_questions = len(questions)
    ncols = 2
    nrows = math.ceil(n_questions / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(15, nrows * 3.8),
        constrained_layout=True,
    )
    fig.suptitle(survey_title, fontsize=13, fontweight="bold")

    # Always work with a flat list
    axes_flat = axes.flatten() if n_questions > 1 else [axes]

    n_respondents = sum(next(iter(counts.values())).values()) if counts else 0

    for i, q in enumerate(questions):
        ax = axes_flat[i]
        qid = q["id"]
        choices = q["choices"]
        letters = list(choices.keys())
        raw_labels = [choices[l] for l in letters]
        labels = [_truncate(lbl, 34) for lbl in raw_labels]

        raw_values = [counts[qid].get(l, 0) for l in letters]
        total = sum(raw_values) or 1
        values = [v / total * 100 for v in raw_values] if use_percent else raw_values
        value_fmt = "{:.1f}%" if use_percent else "{:d}"
        xlabel = "% of respondents" if use_percent else "respondents"

        # Horizontal bars, reversed so 'a' appears at top
        bars = ax.barh(
            labels[::-1],
            values[::-1],
            color=_COLORS[: len(letters)],
            edgecolor="white",
            linewidth=0.6,
        )

        # Value labels at end of each bar
        x_max = max(values) if values else 1
        for bar, val, raw in zip(bars, values[::-1], raw_values[::-1]):
            if raw > 0:
                label = f"{val:.1f}%" if use_percent else str(int(val))
                ax.text(
                    bar.get_width() + x_max * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    label,
                    va="center", ha="left", fontsize=8.5,
                )

        ax.set_title(
            f"Q{qid}. {_truncate(q['text'], 80)}",
            fontsize=9, fontweight="bold", loc="left",
        )
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlim(0, x_max * 1.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide any unused subplot slots
    for j in range(n_questions, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Footer
    mode = "%" if use_percent else "count"
    fig.text(
        0.5, -0.01,
        f"n = {n_respondents} respondents  |  showing {mode}",
        ha="center", fontsize=8, color="#666666",
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved → {out_path}")
    plt.close(fig)


# -----------------------
# CLI
# -----------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate charts from survey results"
    )
    parser.add_argument("--results", required=True,
                        help="Path to a results JSONL file, e.g. results/run_001_immigration_survey_results.jsonl")
    parser.add_argument("--survey",  required=True,
                        help="Path to the survey YAML that was used, e.g. surveys/immigration_survey.yaml")
    parser.add_argument("--out",     default=None,
                        help="Output PNG path (default: same location as results file, .png extension)")
    parser.add_argument("--percent", action="store_true",
                        help="Show percentages instead of raw counts")
    args = parser.parse_args()

    results = load_results(args.results)
    survey  = load_survey(args.survey)
    questions = survey["questions"]
    q_ids     = [q["id"] for q in questions]

    # Filter to personas that responded without error
    valid = [r for r in results if r.get("error") is None]
    if not valid:
        raise SystemExit("No successful results found in the file.")

    print(f"results : {len(valid)} successful / {len(results)} total")
    print(f"survey  : {survey.get('title', args.survey)} ({len(questions)} questions)")

    counts = count_answers(valid, q_ids)

    out_path = Path(args.out) if args.out else Path(args.results).with_suffix(".png")

    plot_results(
        counts=counts,
        questions=questions,
        survey_title=survey.get("title", Path(args.survey).stem),
        out_path=out_path,
        use_percent=args.percent,
    )


if __name__ == "__main__":
    main()
