import argparse
import json
import math
import random
from dataclasses import dataclass
from itertools import product
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
        # Get P(attr|attr1=val1)
        d1 = cond1.get(attr, {}).get(val1)
        # Get P(attr|attr2=val2)
        d2 = cond2.get(attr, {}).get(val2)
        if d1 and d2:
            per_attr[attr] = soft_product_distributions(d1, d2)
        elif d1:
            per_attr[attr] = normalize(dict(d1))
        elif d2:
            per_attr[attr] = normalize(dict(d2))
        else:
            per_attr[attr] = uniform_over(values)
    return per_attr


# -----------------------
# Scoring and sampling
# -----------------------

def compute_conditional_distribution(
    fixed_param_name: str,
    fixed_param_value: str,
    schema: Dict[str, List[str]],
    conditionals: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[Tuple[Tuple[str, str], ...], float]:
    remaining_attrs: List[str] = [a for a in schema.keys() if a != fixed_param_name]
    for attr in remaining_attrs:
        if attr not in conditionals or fixed_param_value not in conditionals[attr]:
            raise ValueError(
                f"Missing P({attr}|{fixed_param_name}={fixed_param_value}) table"
            )

    combos: List[Tuple[str, ...]] = list(
        product(*[schema[attr] for attr in remaining_attrs])
    )

    def log_score(choice_tuple: Tuple[str, ...]) -> float:
        total = 0.0
        for attr, val in zip(remaining_attrs, choice_tuple):
            p = conditionals[attr][fixed_param_value].get(val, 0.0)
            if p <= 0.0:
                return -1e12
            total += math.log(p)
        return total

    logs = [log_score(c) for c in combos]
    m = max(logs)
    exps = [math.exp(x - m) for x in logs]
    Z = sum(exps)
    probs = [e / Z for e in exps]

    results: Dict[Tuple[Tuple[str, str], ...], float] = {}
    for combo, p in zip(combos, probs):
        key = tuple((attr, val) for attr, val in zip(remaining_attrs, combo))
        results[key] = p
    return results


def build_conditionals_from_yaml(data: Dict[str, Any], fixed_param: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    cond = data.get("conditionals", {})
    for key, table in cond.items():
        parts = [p.strip() for p in key.split("|")]
        if len(parts) != 2:
            continue
        attr, given = parts
        if given != fixed_param:
            continue
        out[attr] = table
    schema = data.get("schema", {})
    marginals = data.get("marginals", {})
    for attr, values in schema.items():
        if attr == fixed_param:
            continue
        if attr not in out and attr in marginals:
            m = marginals[attr]
            s = sum(m.get(v, 0.0) for v in values) or 1.0
            uniformized = {v: (m.get(v, 0.0) / s) for v in values}
            out[attr] = {fv: dict(uniformized) for fv in data["schema"][fixed_param]}
    return out


# Note: no pairwise adjustments parsed from YAML; computation uses only available conditionals/marginals.


# -----------------------
# Persona generator
# -----------------------

@dataclass
class Persona:
    values: Dict[str, str]


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

    for _ in range(n):
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
        personas.append(Persona(values=vals))
    return personas


# -----------------------
# CLI
# -----------------------

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
    g.add_argument("-n", type=int, default=10, help="Number of personas")
    g.add_argument("--fix", type=parse_fix_arg, help="Fixed attr=value")
    g.add_argument("--fix2", type=parse_fix_arg, help="Second fixed attr=value")
    g.add_argument("--seed", type=int, default=None, help="RNG seed (omit for non-deterministic runs)")

    # Top-k combinations for a single fixed attribute (existing conceptually)
    k = sub.add_parser("topk", help="Show top-k combinations given one fixed attribute")
    k.add_argument("--data", default="data/demographics.yaml")
    k.add_argument("--fixed", required=True, help="Fixed attribute name (e.g., gender)")
    k.add_argument("--value", required=True, help="Fixed attribute value (e.g., male)")
    k.add_argument("--top", type=int, default=20, help="Show top-k combinations")

    args = parser.parse_args()

    if args.cmd == "generate":
        if args.seed is not None:
            random.seed(args.seed)
        data = load_demographics(args.data)
        personas = generate_personas(
            data=data,
            n=args.n,
            fixed1=args.fix,
            fixed2=args.fix2,
        )
        for p in personas:
            print(json.dumps(p.values))
        return

    if args.cmd == "topk":
        data = load_demographics(args.data)
        schema = data["schema"]
        if args.fixed not in schema:
            raise ValueError(f"Unknown fixed attribute '{args.fixed}'. Available: {list(schema.keys())}")
        if args.value not in schema[args.fixed]:
            raise ValueError(f"Unknown value '{args.value}' for {args.fixed}. Available: {schema[args.fixed]}")

        conditionals = build_conditionals_from_yaml(data, fixed_param=args.fixed)
        dist = compute_conditional_distribution(
            fixed_param_name=args.fixed,
            fixed_param_value=args.value,
            schema=schema,
            conditionals=conditionals,
        )
        items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[: args.top]
        for combo, p in items:
            combo_str = ", ".join([f"{a}={v}" for a, v in combo])
            print(f"{p:.6f}  {combo_str}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
