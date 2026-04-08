import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ADULT_MIN_AGE = 18
DECEMBER = 12


@dataclass(frozen=True)
class RaceEthCategory:
    key: str
    male_column: str
    female_column: str
    description: str


# Column names taken from the CSV header
CATEGORIES: List[RaceEthCategory] = [
    RaceEthCategory(
        key="nonhispanic_white_alone",
        male_column="NHWA_MALE",
        female_column="NHWA_FEMALE",
        description="Non-Hispanic White alone",
    ),
    RaceEthCategory(
        key="nonhispanic_black_alone",
        male_column="NHBA_MALE",
        female_column="NHBA_FEMALE",
        description="Non-Hispanic Black or African American alone",
    ),
    RaceEthCategory(
        key="nonhispanic_asian_alone",
        male_column="NHAA_MALE",
        female_column="NHAA_FEMALE",
        description="Non-Hispanic Asian alone",
    ),
    RaceEthCategory(
        key="nonhispanic_native_hawaiian_pacific_islander_alone",
        male_column="NHNA_MALE",
        female_column="NHNA_FEMALE",
        description="Non-Hispanic Native Hawaiian and Other Pacific Islander alone",
    ),
    RaceEthCategory(
        key="nonhispanic_native_american_alone",
        male_column="NHIA_MALE",
        female_column="NHIA_FEMALE",
        description="Non-Hispanic American Indian and Alaska Native alone",
    ),
    RaceEthCategory(
        key="hispanic_any_race",
        male_column="H_MALE",
        female_column="H_FEMALE",
        description="Hispanic or Latino (of any race)",
    ),
    RaceEthCategory(
        key="nonhispanic_multiracial_two_or_more",
        male_column="NHTOM_MALE",
        female_column="NHTOM_FEMALE",
        description="Non-Hispanic Two or More races",
    ),
]


def _parse_int(value: str) -> int:
    value = value.strip()
    return int(value) if value else 0


def _row_is_adult_december(row: Dict[str, str]) -> bool:
    try:
        month = _parse_int(row["MONTH"])
        age = _parse_int(row["AGE"])
    except KeyError as exc:
        raise KeyError(f"Missing expected column in CSV: {exc}") from exc
    # AGE 999 is the 'all ages' summary row; exclude it. Adults are AGE >= 18.
    return month == DECEMBER and age >= ADULT_MIN_AGE and age != 999


def _sum_columns(row: Dict[str, str], male_col: str, female_col: str) -> int:
    return _parse_int(row.get(male_col, "0")) + _parse_int(row.get(female_col, "0"))


def compute_adult_totals_month(
    csv_path: str | Path,
    month: int = DECEMBER,
    adult_min_age: int = ADULT_MIN_AGE,
    include_categories: Iterable[RaceEthCategory] = CATEGORIES,
) -> Dict[str, int]:
    """
    Compute adult (age >= adult_min_age) totals for the given month by race/ethnicity categories.
    Excludes 'all ages' summary rows where AGE == 999.
    """
    totals: Dict[str, int] = {c.key: 0 for c in include_categories}

    def row_match(row: Dict[str, str]) -> bool:
        m = _parse_int(row.get("MONTH", "0"))
        a = _parse_int(row.get("AGE", "0"))
        return m == month and a >= adult_min_age and a != 999

    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row_match(row):
                continue
            for cat in include_categories:
                totals[cat.key] += _sum_columns(row, cat.male_column, cat.female_column)
    return totals


def compute_adult_totals_december(csv_path: str | Path) -> Dict[str, int]:
    """
    Convenience wrapper for December (month 12).
    """
    return compute_adult_totals_month(csv_path, month=DECEMBER, adult_min_age=ADULT_MIN_AGE)


def pretty_print_totals(totals: Dict[str, int]) -> str:
    lines: List[str] = []
    for cat in CATEGORIES:
        if cat.key in totals:
            lines.append(f"{cat.key}: {totals[cat.key]:,d}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Example CLI usage:
    csv_file = Path(__file__).parent / "nc-est2024-alldata-r-file12.csv"
    result = compute_adult_totals_december(csv_file)
    print(pretty_print_totals(result))
