import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# -----------------------
# Data loading utilities
# -----------------------

def load_demographics(path: str = "data/demographics.yaml") -> Dict[str, Any]:
    file_path = Path(__file__).parent / path
    if not file_path.exists():
        raise FileNotFoundError(f"Demographics file not found at {file_path}")
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")
    with file_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_issues(path: str) -> Dict[str, Any]:
    file_path = Path(__file__).parent / path
    if not file_path.exists():
        raise FileNotFoundError(f"Issues file not found at {file_path}")
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")
    with file_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -----------------------
# Probability assembly
# -----------------------

def normalize(dist: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, p) for p in dist.values())
    if total <= 0.0:
        # Fallback to uniform over keys
        n = max(1, len(dist))
        return {k: 1.0 / n for k in dist}
    return {k: max(0.0, p) / total for k, p in dist.items()}


def soft_product_distributions(d1: Dict[str, float], d2: Dict[str, float]) -> Dict[str, float]:
    # Product-of-experts: p ∝ p1 * p2 (on intersecting support), then normalize
    keys = set(d1.keys()) | set(d2.keys())
    out: Dict[str, float] = {}
    for k in keys:
        p1 = d1.get(k, 0.0)
        p2 = d2.get(k, 0.0)
        out[k] = p1 * p2 if (p1 > 0.0 and p2 > 0.0) else 0.0
    return normalize(out)


# How strongly each party_strength level pulls the party signal vs. the marginal.
# alpha=1.0 → full party conditional; alpha→0 → flat/marginal (no party signal).
_PARTY_STRENGTH_ALPHA = {"strong": 1.0, "not_strong": 0.55, "lean": 0.25}


def attenuate_distribution(dist: Dict[str, float], alpha: float) -> Dict[str, float]:
    """Raise each probability to power alpha then renormalize.

    alpha=1.0 leaves the distribution unchanged.
    alpha<1.0 flattens it toward uniform, reducing the signal strength.
    """
    out = {k: max(v, 1e-9) ** alpha for k, v in dist.items()}
    return normalize(out)


def uniform_over(values: Iterable[str]) -> Dict[str, float]:
    values = list(values)
    if not values:
        return {}
    p = 1.0 / len(values)
    return {v: p for v in values}


def build_conditionals_for_fixed(
    data: Dict[str, Any], fixed_attr: str
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Return conditionals[attr][fixed_value][attr_value] = prob
    using entries like conditionals['age|gender'] in YAML.
    For attrs lacking an entry, fall back to marginals.
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    cond = data.get("conditionals", {}) or {}
    schema = data.get("schema", {})
    marginals = data.get("marginals", {})

    for key, table in cond.items():
        parts = [p.strip() for p in key.split("|")]
        if len(parts) != 2:
            continue
        attr, given = parts
        if given != fixed_attr:
            continue
        out[attr] = table  # already {fixed_value: {attr_value: p}}

    for attr, values in schema.items():
        if attr == fixed_attr:
            continue
        if attr not in out:
            # fallback: same marginal regardless of fixed value
            m = marginals.get(attr)
            if isinstance(m, dict):
                # normalize marginals to be safe
                s = sum(m.get(v, 0.0) for v in values) or 1.0
                norm = {v: (m.get(v, 0.0) / s) for v in values}
                out[attr] = {fv: dict(norm) for fv in schema[fixed_attr]}
            else:
                # no marginals -> uniform
                u = uniform_over(values)
                out[attr] = {fv: dict(u) for fv in schema[fixed_attr]}
    return out


def combine_two_fixed_distributions(
    data: Dict[str, Any],
    fixed1: Tuple[str, str],
    fixed2: Tuple[str, str],
) -> Dict[str, Dict[str, float]]:
    """
    Build per-attribute distributions given two fixed attributes by
    product-of-experts: P(attr|f1,f2) ∝ P(attr|f1) * P(attr|f2).
    Returns per_attr_dist[attr][value] = prob for all attrs not fixed.
    """
    schema = data.get("schema", {})
    attr1, val1 = fixed1
    attr2, val2 = fixed2

    cond1 = build_conditionals_for_fixed(data, attr1)
    cond2 = build_conditionals_for_fixed(data, attr2)

    per_attr: Dict[str, Dict[str, float]] = {}
    for attr, values in schema.items():
        if attr in (attr1, attr2):
            continue
        d1 = cond1.get(attr, {}).get(val1)
        d2 = cond2.get(attr, {}).get(val2)
        if d1 and d2:
            per_attr[attr] = soft_product_distributions(d1, d2)
        elif d1:
            per_attr[attr] = normalize(dict(d1))
        elif d2:
            per_attr[attr] = normalize(dict(d2))
        else:
            marginal = data.get("marginals", {}).get(attr)
            if isinstance(marginal, dict):
                per_attr[attr] = normalize({v: marginal.get(v, 0.0) for v in values})
            else:
                per_attr[attr] = uniform_over(values)
    return per_attr

# -----------------------
# Issue scoring
# -----------------------

def score_issue(
    demographics: Dict[str, str],
    issue: Dict[str, Any],
    score_buckets: Dict[str, List[float]],
) -> float:
    """
    Sample a score in [-1.0, 1.0] for one issue given a persona's demographics.

    Algorithm:
      1. Start with the issue's marginal bucket distribution.
      2. For each demographic attribute that has a conditional table in this issue,
         retrieve P(bucket | attr=val) and combine with the running distribution
         using product-of-experts (same pattern as soft_product_distributions).
      3. Sample a bucket from the combined distribution.
      4. Sample a float uniformly within that bucket's [lo, hi] range.
    """
    combined = normalize(dict(issue["marginal"]))
    party_strength = demographics.get("party_strength", "strong")

    for attr, val in demographics.items():
        if attr == "party_strength":
            continue
        cond_tables = issue.get("conditionals", {})
        if attr in cond_tables and val in cond_tables[attr]:
            conditional_dist = normalize(dict(cond_tables[attr][val]))
            if attr == "party":
                alpha = _PARTY_STRENGTH_ALPHA.get(party_strength, 1.0)
                conditional_dist = attenuate_distribution(conditional_dist, alpha)
            combined = soft_product_distributions(combined, conditional_dist)

    bucket = sample_from(combined)
    lo, hi = score_buckets[bucket]
    return round(random.uniform(lo, hi), 3)


def assign_scores(
    demographics: Dict[str, str],
    issues_data: Dict[str, Any],
) -> Dict[str, float]:
    """Score every issue in an issues YAML file for one persona."""
    score_buckets = issues_data["score_buckets"]
    return {
        key: score_issue(demographics, issue, score_buckets)
        for key, issue in issues_data["issues"].items()
    }


# -----------------------
# Persona generator
# -----------------------

@dataclass
class Persona:
    id: str
    demographics: Dict[str, str]
    values: Dict[str, float]
    wants: Dict[str, float]


def sample_from(dist: Dict[str, float]) -> str:
    r = random.random()
    cum = 0.0
    for k, p in dist.items():
        cum += p
        if r <= cum:
            return k
    # numerical edge: return last key
    return next(reversed(dist.keys()))


def generate_personas(
    data: Dict[str, Any],
    n: int,
    fixed1: Tuple[str, str] | None = None,
    fixed2: Tuple[str, str] | None = None,
    values_data: Dict[str, Any] | None = None,
    wants_data: Dict[str, Any] | None = None,
) -> List[Persona]:
    schema = data.get("schema", {})
    if fixed1:
        a1, v1 = fixed1
        if a1 not in schema or v1 not in schema[a1]:
            raise ValueError(f"Invalid fixed1 {fixed1}")
    if fixed2:
        a2, v2 = fixed2
        if a2 not in schema or v2 not in schema[a2]:
            raise ValueError(f"Invalid fixed2 {fixed2}")
        if fixed1 and a2 == fixed1[0] and v2 != fixed1[1]:
            raise ValueError("Conflicting fixed values for the same attribute")

    personas: List[Persona] = []
    # Build per-attribute distributions
    if fixed1 and fixed2:
        per_attr = combine_two_fixed_distributions(data, fixed1, fixed2)
    elif fixed1:
        conds = build_conditionals_for_fixed(data, fixed1[0])
        per_attr = {attr: normalize(conds[attr][fixed1[1]]) for attr in schema if attr != fixed1[0]}
    else:
        # No fixed: use marginals
        marginals = data.get("marginals", {})
        per_attr = {}
        for attr, values in schema.items():
            m = marginals.get(attr)
            if isinstance(m, dict):
                per_attr[attr] = normalize({v: m.get(v, 0.0) for v in values})
            else:
                per_attr[attr] = uniform_over(values)

    for i in range(n):
        vals: Dict[str, str] = {}
        # Assign fixed values first
        if fixed1:
            vals[fixed1[0]] = fixed1[1]
        if fixed2:
            vals[fixed2[0]] = fixed2[1]
        for attr in schema.keys():
            if attr in vals:
                continue
            dist = per_attr.get(attr)
            if not dist:
                dist = uniform_over(schema[attr])
            choice = sample_from(dist)
            vals[attr] = choice
        personas.append(Persona(
            id=f"p{i + 1:04d}",
            demographics=vals,
            values=assign_scores(vals, values_data) if values_data else {},
            wants=assign_scores(vals, wants_data) if wants_data else {},
        ))
    return personas


# -----------------------
# Run file management
# -----------------------

RUNS_DIR    = Path(__file__).parent / "runs"
RESULTS_DIR = Path(__file__).parent / "results"


def list_results() -> List[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("*.jsonl"))


def next_run_path() -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    existing = sorted(RUNS_DIR.glob("run_*.jsonl"))
    nums = []
    for p in existing:
        try:
            nums.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    n = max(nums, default=0) + 1
    return RUNS_DIR / f"run_{n:03d}.jsonl"


def list_runs() -> List[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted(RUNS_DIR.glob("run_*.jsonl"))


def parse_fix_arg(s: str) -> Tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError("Must be in form attr=value")
    a, v = s.split("=", 1)
    return a.strip(), v.strip()


def main():
    parser = argparse.ArgumentParser(description="Persona tools")
    sub = parser.add_subparsers(dest="cmd")

    # Generate personas
    g = sub.add_parser("generate", help="Generate personas with one or two fixed attributes")
    g.add_argument("--data", default="data/demographics.yaml")
    g.add_argument("--values", default="data/values.yaml", help="Path to values issues YAML")
    g.add_argument("--wants", default="data/wants.yaml", help="Path to wants issues YAML")
    g.add_argument("-n", type=int, default=10, help="Number of personas")
    g.add_argument("--fix", type=parse_fix_arg, help="Fixed attr=value")
    g.add_argument("--fix2", type=parse_fix_arg, help="Second fixed attr=value")
    g.add_argument("--seed", type=int, default=None, help="RNG seed (omit for non-deterministic runs)")
    g.add_argument("--no-save", action="store_true", help="Print to stdout only, do not save a run file")
    g.add_argument("--type", type=int, choices=[1, 2, 3], default=3,
                   help="1=demographics only, 2=values/wants only (demographics hidden), 3=full (default)")

    # List saved runs
    sub.add_parser("list", help="List saved persona runs")

    # Clear saved runs
    cl = sub.add_parser("clear", help="Delete saved run file(s)")
    cl.add_argument("run", nargs="?", help="Run name to delete, e.g. run_001 (omit with --all to delete everything)")
    cl.add_argument("--all", action="store_true", help="Delete all saved runs")

    # List result files
    sub.add_parser("list-results", help="List saved survey result files")

    # Clear result files
    cr = sub.add_parser("clear-results", help="Delete survey result file(s)")
    cr.add_argument("result", nargs="*",
                    help="Result filename(s) to delete, e.g. run_001_immigration_survey_results "
                         "(omit with --all to delete everything)")
    cr.add_argument("--all", action="store_true", help="Delete all result files")

    args = parser.parse_args()

    if args.cmd == "generate":
        if args.seed is not None:
            random.seed(args.seed)
        data = load_demographics(args.data)
        values_data = load_issues(args.values) if args.type in (2, 3) else None
        wants_data  = load_issues(args.wants)  if args.type in (2, 3) else None
        personas = generate_personas(
            data=data,
            n=args.n,
            fixed1=args.fix,
            fixed2=args.fix2,
            values_data=values_data,
            wants_data=wants_data,
        )
        lines = []
        for p in personas:
            if args.type == 1:
                record = {"id": p.id, "demographics": p.demographics}
            elif args.type == 2:
                record = {"id": p.id, "values": p.values, "wants": p.wants}
            else:
                record = {"id": p.id, "demographics": p.demographics, "values": p.values, "wants": p.wants}
            lines.append(json.dumps(record))
        for line in lines:
            print(line)
        if not args.no_save:
            run_path = next_run_path()
            run_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"saved {len(personas)} personas → {run_path}", file=sys.stderr)
        return

    if args.cmd == "list":
        runs = list_runs()
        if not runs:
            print("no saved runs")
        for path in runs:
            count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            print(f"{path.name}  ({count} personas)")
        return

    if args.cmd == "clear":
        if args.all:
            runs = list_runs()
            for path in runs:
                path.unlink()
            print(f"deleted {len(runs)} run(s)")
        elif args.run:
            name = args.run if args.run.endswith(".jsonl") else args.run + ".jsonl"
            path = RUNS_DIR / name
            if not path.exists():
                print(f"not found: {path}")
            else:
                path.unlink()
                print(f"deleted {path.name}")
        else:
            print("specify a run name (e.g. run_001) or use --all")
        return

    if args.cmd == "list-results":
        results = list_results()
        if not results:
            print("no saved results")
        for path in results:
            count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            print(f"{path.name}  ({count} records)")
        return

    if args.cmd == "clear-results":
        if args.all:
            results = list_results()
            images  = sorted(RESULTS_DIR.glob("*.png")) if RESULTS_DIR.exists() else []
            for path in results + images:
                path.unlink()
            print(f"deleted {len(results)} result file(s) and {len(images)} image(s)")
        elif args.result:
            for name in args.result:
                name = name if name.endswith(".jsonl") else name + ".jsonl"
                path = RESULTS_DIR / name
                if not path.exists():
                    print(f"not found: {path.name}")
                else:
                    path.unlink()
                    print(f"deleted {path.name}")
        else:
            print("specify result filename(s) or use --all")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
