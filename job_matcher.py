
"""
Job Matcher for AI Resume Analyzer.

Compares extracted candidate skills against job requirements.

Scoring:
- Required skills: 80% of the score
- Preferred skills: 20% of the score

The score measures skill overlap. It is not a probability
of getting hired or a complete assessment of a candidate.
"""

import csv
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_JOB_ROLES_PATH = BASE_DIR / "Data" / "Job_roles.csv"


# ============================================================
# HELPERS
# ============================================================

def normalize_skill(skill: str) -> str:
    """Normalize a skill name for case-insensitive matching."""

    return " ".join(str(skill).strip().casefold().split())


def split_skill_list(value: str) -> list[str]:
    """Split a pipe-separated skill list from the CSV."""

    if not value:
        return []

    skills = []
    seen = set()

    for item in value.split("|"):
        skill = item.strip()

        if not skill:
            continue

        key = normalize_skill(skill)

        if key not in seen:
            seen.add(key)
            skills.append(skill)

    return skills


def parse_minimum_experience(value: str, job_role: str) -> int:
    """Read a non-negative minimum experience requirement."""

    try:
        experience = int(value.strip())
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"Invalid minimum_experience_years for "
            f"job role '{job_role}': {value!r}"
        ) from exc

    if experience < 0:
        raise ValueError(
            f"Minimum experience cannot be negative for "
            f"job role '{job_role}'."
        )

    return experience


# ============================================================
# LOAD JOB ROLE DATASET
# ============================================================

def load_job_roles(job_roles_path=None) -> list[dict[str, Any]]:
    """Load and validate the job-role CSV."""

    path = (
        Path(job_roles_path)
        if job_roles_path is not None
        else DEFAULT_JOB_ROLES_PATH
    )

    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.is_file():
        raise FileNotFoundError(
            f"Job roles dataset not found: {path}. "
            "Create Data/Job_roles.csv first."
        )

    required_columns = {
        "job_role",
        "category",
        "description",
        "required_skills",
        "preferred_skills",
        "minimum_experience_years",
    }

    roles = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:

        reader = csv.DictReader(
            line for line in csv_file if line.strip()
        )

        headers = set(reader.fieldnames or [])

        if not required_columns.issubset(headers):
            missing = sorted(required_columns - headers)

            raise ValueError(
                "Job_roles.csv is missing columns: "
                + ", ".join(missing)
            )

        for row in reader:
            role_name = (row.get("job_role") or "").strip()

            if not role_name:
                continue

            required_skills = split_skill_list(
                row.get("required_skills", "")
            )

            preferred_skills = split_skill_list(
                row.get("preferred_skills", "")
            )

            if not required_skills and not preferred_skills:
                raise ValueError(
                    f"Job role '{role_name}' has no required "
                    "or preferred skills."
                )

            roles.append({
                "job_role": role_name,
                "category": (row.get("category") or "").strip(),
                "description": (
                    row.get("description") or ""
                ).strip(),
                "required_skills": required_skills,
                "preferred_skills": preferred_skills,
                "minimum_experience_years": (
                    parse_minimum_experience(
                        row.get("minimum_experience_years", ""),
                        role_name,
                    )
                ),
            })

    if not roles:
        raise ValueError(
            f"No valid job roles found in {path}."
        )

    return roles


# ============================================================
# NORMALIZE EXTRACTED RESUME SKILLS
# ============================================================

def normalize_resume_skills(resume_skills) -> set[str]:
    """
    Accept either:
      - A list of strings
      - A list of dictionaries containing 'skill'
      - The dictionary returned by extract_skills_from_resume()
    """

    if isinstance(resume_skills, dict):
        if "skills" not in resume_skills:
            raise ValueError(
                "Expected extracted skills. Pass the output of "
                "extract_skills() or extract_skills_from_resume(), "
                "not the raw parsed resume."
            )

        resume_skills = resume_skills["skills"]

    if isinstance(resume_skills, str):
        resume_skills = [resume_skills]

    if not isinstance(resume_skills, (list, tuple, set)):
        raise TypeError(
            "resume_skills must be a list, tuple, set, "
            "or extracted-skills dictionary."
        )

    normalized = set()

    for item in resume_skills:

        if isinstance(item, str):
            skill_name = item

        elif isinstance(item, dict):
            skill_name = item.get("skill", "")

        else:
            continue

        key = normalize_skill(skill_name)

        if key:
            normalized.add(key)

    return normalized


# ============================================================
# MATCH ONE JOB ROLE
# ============================================================

def calculate_job_match(
    candidate_skills: set[str],
    role: dict[str, Any],
    experience_years: int | None = None,
) -> dict[str, Any]:
    """Calculate an explainable skill-match result for one role."""

    required = role["required_skills"]
    preferred = role["preferred_skills"]

    matched_required = [
        skill for skill in required
        if normalize_skill(skill) in candidate_skills
    ]

    missing_required = [
        skill for skill in required
        if normalize_skill(skill) not in candidate_skills
    ]

    matched_preferred = [
        skill for skill in preferred
        if normalize_skill(skill) in candidate_skills
    ]

    missing_preferred = [
        skill for skill in preferred
        if normalize_skill(skill) not in candidate_skills
    ]

    required_coverage = (
        len(matched_required) / len(required)
        if required else None
    )

    preferred_coverage = (
        len(matched_preferred) / len(preferred)
        if preferred else None
    )

    # Give required skills more weight than preferred skills.
    if required and preferred:
        score = (
            0.80 * required_coverage
            + 0.20 * preferred_coverage
        )

    elif required:
        score = required_coverage

    elif preferred:
        score = preferred_coverage

    else:
        score = 0.0

    minimum_experience = role["minimum_experience_years"]

    if experience_years is None:
        experience_requirement_met = None
    else:
        experience_requirement_met = (
            experience_years >= minimum_experience
        )

    return {
        "job_role": role["job_role"],
        "category": role["category"],
        "description": role["description"],

        # Skill-based score, from 0 to 100.
        "match_score": round(score * 100, 2),

        "required_skill_coverage_percent": (
            round(required_coverage * 100, 2)
            if required_coverage is not None else None
        ),

        "preferred_skill_coverage_percent": (
            round(preferred_coverage * 100, 2)
            if preferred_coverage is not None else None
        ),

        "matched_required_skills": matched_required,
        "missing_required_skills": missing_required,
        "matched_preferred_skills": matched_preferred,
        "missing_preferred_skills": missing_preferred,

        "total_required_skills": len(required),
        "total_preferred_skills": len(preferred),
        "total_matched_skills": (
            len(matched_required) + len(matched_preferred)
        ),

        "minimum_experience_years": minimum_experience,
        "candidate_experience_years": experience_years,
        "experience_requirement_met": experience_requirement_met,
    }


# ============================================================
# MATCH AND RANK ALL JOB ROLES
# ============================================================

def match_jobs(
    resume_skills,
    job_roles_path=None,
    experience_years: int | None = None,
    top_n: int | None = 5,
) -> list[dict[str, Any]]:
    """
    Compare extracted resume skills with every job role.

    Returns roles sorted by skill-match score, highest first.
    """

    if experience_years is not None:
        if (
            isinstance(experience_years, bool)
            or not isinstance(experience_years, int)
            or experience_years < 0
        ):
            raise ValueError(
                "experience_years must be a non-negative integer "
                "or None."
            )

    if top_n is not None:
        if (
            isinstance(top_n, bool)
            or not isinstance(top_n, int)
            or top_n < 1
        ):
            raise ValueError(
                "top_n must be a positive integer or None."
            )

    candidate_skills = normalize_resume_skills(resume_skills)

    roles = load_job_roles(job_roles_path)

    results = [
        calculate_job_match(
            candidate_skills=candidate_skills,
            role=role,
            experience_years=experience_years,
        )
        for role in roles
    ]

    # Rank by score, then required-skill coverage,
    # then preferred-skill coverage, then role name.
    results.sort(
        key=lambda result: (
            -result["match_score"],
            -(result["required_skill_coverage_percent"] or 0),
            -(result["preferred_skill_coverage_percent"] or 0),
            result["job_role"].casefold(),
        )
    )

    return results[:top_n] if top_n is not None else results
