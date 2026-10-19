
"""
Skill Extractor for AI Resume Analyzer.

Loads a skill dictionary from Data/Skill_dictionary.csv,
detects skills in extracted resume text, and returns
standardized skill names with categories and match counts.
"""

import csv
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DICTIONARY_PATH = (
    BASE_DIR / "Data" / "Skill_dictionary.csv"
)


# ============================================================
# LOAD SKILL DICTIONARY
# ============================================================

def load_skill_dictionary(dictionary_path=None):
    """Load canonical skills, categories, and aliases from CSV."""

    path = Path(dictionary_path) if dictionary_path else (
        DEFAULT_DICTIONARY_PATH
    )

    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.exists():
        raise FileNotFoundError(
            f"Skill dictionary not found: {path}. "
            "Create Data/Skill_dictionary.csv first."
        )

    skills = []

    with path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        required_columns = {
            "skill",
            "category",
            "aliases",
        }

        if not reader.fieldnames or not required_columns.issubset(
            set(reader.fieldnames)
        ):
            raise ValueError(
                "Skill_dictionary.csv must contain these columns: "
                "skill, category, aliases"
            )

        for row in reader:
            skill_name = (row.get("skill") or "").strip()
            category = (row.get("category") or "").strip()
            aliases_text = (row.get("aliases") or "").strip()

            if not skill_name:
                continue

            # Always match the canonical skill name as well.
            aliases = [skill_name]

            aliases.extend(
                alias.strip()
                for alias in aliases_text.split("|")
                if alias.strip()
            )

            # Deduplicate aliases without case sensitivity.
            unique_aliases = []
            seen = set()

            for alias in aliases:
                key = alias.casefold()

                if key not in seen:
                    seen.add(key)
                    unique_aliases.append(alias)

            skills.append({
                "skill": skill_name,
                "category": category or "Other",
                "aliases": unique_aliases,
            })

    if not skills:
        raise ValueError(
            "The skill dictionary contains no valid skills."
        )

    return skills


# ============================================================
# MATCH SKILL ALIASES
# ============================================================

def build_alias_pattern(alias):
    """
    Match a complete alias rather than part of a longer word.

    Flexible whitespace allows phrases with multiple spaces
    or line breaks to be recognized.
    """

    words = re.split(r"\s+", alias.strip())

    phrase_pattern = r"\s+".join(
        re.escape(word) for word in words
    )

    return re.compile(
        r"(?<!\w)" + phrase_pattern + r"(?!\w)",
        flags=re.IGNORECASE,
    )


# ============================================================
# EXTRACT SKILLS FROM TEXT
# ============================================================

def extract_skills(text, dictionary_path=None):
    """
    Detect skills in extracted resume text.

    Returns a list of dictionaries containing:
        skill
        category
        mentions
        matched_aliases
    """

    if not isinstance(text, str):
        raise TypeError("Resume text must be a string.")

    if not text.strip():
        return []

    dictionary = load_skill_dictionary(dictionary_path)

    # Collect all possible matches first.
    candidates = []

    for entry in dictionary:

        for alias in entry["aliases"]:

            pattern = build_alias_pattern(alias)

            for match in pattern.finditer(text):

                candidates.append({
                    "start": match.start(),
                    "end": match.end(),
                    "skill": entry["skill"],
                    "category": entry["category"],
                    "alias": match.group(0),
                })

    # Prefer longer matches when phrases overlap.
    # Example: C++ should be preferred over C.
    candidates.sort(
        key=lambda item: (
            -(item["end"] - item["start"]),
            item["start"],
        )
    )

    accepted_matches = []

    for candidate in candidates:

        overlaps = any(
            candidate["start"] < existing["end"]
            and candidate["end"] > existing["start"]
            for existing in accepted_matches
        )

        if not overlaps:
            accepted_matches.append(candidate)

    # Restore document order.
    accepted_matches.sort(key=lambda item: item["start"])

    # Group occurrences by canonical skill.
    grouped = {}

    for match in accepted_matches:

        skill_name = match["skill"]

        if skill_name not in grouped:
            grouped[skill_name] = {
                "skill": skill_name,
                "category": match["category"],
                "mentions": 0,
                "matched_aliases": [],
                "_first_position": match["start"],
            }

        grouped[skill_name]["mentions"] += 1

        alias = match["alias"]

        if alias.casefold() not in {
            item.casefold()
            for item in grouped[skill_name]["matched_aliases"]
        }:
            grouped[skill_name]["matched_aliases"].append(alias)

    # Return skills in order of first appearance.
    results = sorted(
        grouped.values(),
        key=lambda item: item["_first_position"],
    )

    for result in results:
        result.pop("_first_position", None)

    return results


# ============================================================
# EXTRACT SKILLS FROM PARSED RESUME
# ============================================================

def extract_skills_from_resume(
    parsed_resume,
    dictionary_path=None,
):
    """
    Accept the dictionary returned by resume_parser.py
    and extract skills from the complete resume text.
    """

    if not isinstance(parsed_resume, dict):
        raise TypeError(
            "parsed_resume must be a dictionary."
        )

    text = (
        parsed_resume.get("raw_text")
        or parsed_resume.get("skills_text")
        or ""
    )

    results = extract_skills(
        text,
        dictionary_path=dictionary_path,
    )

    return {
        "filename": parsed_resume.get(
            "filename", "resume"
        ),
        "skills_found": len(results),
        "skills": results,
    }
