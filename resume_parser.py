"""
Resume Parser
-------------
Extracts structured information from resumes.

Supported source formats are handled by:
processors/document_processor.py
"""

import re
from typing import Any

from processors.document_processor import extract_document_text


# ============================================================
# RESUME SECTION DEFINITIONS
# ============================================================

SECTION_ALIASES = {
    "summary": {
        "summary",
        "professional summary",
        "profile",
        "professional profile",
        "career summary",
        "objective",
        "career objective",
        "about me",
    },
    "skills": {
        "skills",
        "technical skills",
        "key skills",
        "core skills",
        "technical expertise",
        "skill set",
        "competencies",
    },
    "experience": {
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history",
        "internship experience",
        "internships",
    },
    "education": {
        "education",
        "educational qualification",
        "educational qualifications",
        "academic background",
        "academic qualifications",
        "academics",
    },
    "projects": {
        "projects",
        "academic projects",
        "personal projects",
        "key projects",
        "project experience",
    },
    "certifications": {
        "certifications",
        "certificates",
        "licenses and certifications",
        "courses and certifications",
        "professional certifications",
    },
    "achievements": {
        "achievements",
        "awards",
        "honors",
        "honours",
        "accomplishments",
    },
    "languages": {
        "languages",
        "language proficiency",
    },
    "publications": {
        "publications",
        "research publications",
        "papers published",
    },
    "interests": {
        "interests",
        "hobbies",
        "interests and hobbies",
    },
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """Normalize line endings and unnecessary whitespace."""

    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines = []

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(line)

    # Keep line breaks because they help detect resume sections.
    text = "\n".join(cleaned_lines)

    # Avoid excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_heading(line: str) -> str:
    """Normalize a line before comparing it with section headings."""

    line = line.strip().strip(":")
    line = re.sub(r"[\s|]+$", "", line)
    line = re.sub(r"[^a-zA-Z0-9\s&/()\-]", "", line)

    return re.sub(r"\s+", " ", line).strip().lower()


def identify_section_heading(line: str) -> str | None:
    """Return the recognized section name, if the line is a heading."""

    normalized = normalize_heading(line)

    for section_name, aliases in SECTION_ALIASES.items():
        if normalized in aliases:
            return section_name

    return None


# ============================================================
# CONTACT INFORMATION
# ============================================================

def extract_email(text: str) -> str | None:
    """Find the first email address."""

    pattern = r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)

    return match.group(0) if match else None


def extract_phone(text: str) -> str | None:
    """
    Find a plausible phone number.

    This is a heuristic, not a guarantee that the number is valid.
    """

    pattern = r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3,5}\)?[\s.-]?)?\d{5,10}(?!\w)"

    for match in re.finditer(pattern, text):
        candidate = match.group(0).strip()
        digits = re.sub(r"\D", "", candidate)

        if 10 <= len(digits) <= 13:
            return candidate

    return None


def extract_profile_url(text: str, platform: str) -> str | None:
    """Find a LinkedIn or GitHub profile URL."""

    patterns = {
        "linkedin": r"(?:https?://)?(?:www\.)?linkedin\.com/[^\s,;]+",
        "github": r"(?:https?://)?(?:www\.)?github\.com/[^\s,;]+",
    }

    pattern = patterns.get(platform.lower())

    if not pattern:
        return None

    match = re.search(pattern, text, flags=re.IGNORECASE)

    return match.group(0).rstrip(".)]") if match else None


# ============================================================
# NAME DETECTION
# ============================================================

def extract_candidate_name(text: str) -> str | None:
    """
    Estimate the candidate's name from the first few lines.

    The result is a heuristic and should be verified when accuracy
    is important.
    """

    excluded_words = {
        "resume",
        "curriculum vitae",
        "cv",
        "portfolio",
        "professional summary",
        "contact",
        "education",
        "experience",
        "skills",
        "projects",
    }

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines[:8]:
        normalized = line.lower().strip(": ")

        if normalized in excluded_words:
            continue

        if "@" in line or "://" in line or "www." in line.lower():
            continue

        if re.search(r"\d", line):
            continue

        # Avoid treating long headings or sentences as names.
        words = line.split()

        if not 1 <= len(words) <= 5:
            continue

        if len(line) > 70:
            continue

        if identify_section_heading(line):
            continue

        if not re.search(r"[A-Za-z]", line):
            continue

        return line

    return None


# ============================================================
# SECTION EXTRACTION
# ============================================================

def extract_resume_sections(text: str) -> dict[str, str]:
    """
    Group text under recognized headings.

    Text that appears before any recognized heading is stored under
    'other'. Heading recognition is heuristic, so unusual formats
    may require additional aliases.
    """

    sections: dict[str, list[str]] = {}
    current_section = "other"

    for line in text.splitlines():

        section_name = identify_section_heading(line)

        if section_name:
            current_section = section_name
            sections.setdefault(current_section, [])
            continue

        if line.strip():
            sections.setdefault(current_section, []).append(line.strip())

    return {
        name: "\n".join(lines).strip()
        for name, lines in sections.items()
        if lines
    }


# ============================================================
# MAIN TEXT PARSER
# ============================================================

def parse_resume_text(
    text: str,
    filename: str = "resume",
) -> dict[str, Any]:
    """Convert extracted resume text into a structured dictionary."""

    cleaned_text = normalize_text(text)

    if not cleaned_text:
        raise ValueError("The resume contains no readable text.")

    sections = extract_resume_sections(cleaned_text)

    result = {
        "filename": filename,
        "candidate_name": extract_candidate_name(cleaned_text),
        "email": extract_email(cleaned_text),
        "phone": extract_phone(cleaned_text),
        "linkedin": extract_profile_url(cleaned_text, "linkedin"),
        "github": extract_profile_url(cleaned_text, "github"),
        "professional_summary": sections.get("summary", ""),
        "skills_text": sections.get("skills", ""),
        "experience_text": sections.get("experience", ""),
        "education_text": sections.get("education", ""),
        "projects_text": sections.get("projects", ""),
        "certifications_text": sections.get("certifications", ""),
        "achievements_text": sections.get("achievements", ""),
        "languages_text": sections.get("languages", ""),
        "sections": sections,
        "raw_text": cleaned_text,
        "word_count": len(cleaned_text.split()),
        "character_count": len(cleaned_text),
    }

    return result


# ============================================================
# MAIN FILE-TO-RESUME PIPELINE
# ============================================================

def parse_resume_file(
    filename: str,
    file_bytes: bytes,
) -> dict[str, Any]:
    """
    Extract text from an uploaded document and parse the resume.

    This function connects the document processor with the parser.
    """

    extracted = extract_document_text(
        filename,
        file_bytes,
    )

    return parse_resume_text(
        text=extracted["text"],
        filename=filename,
    )
