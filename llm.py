import os
import re
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_RESUME_CHARS = 10000


@lru_cache(maxsize=1)
def get_llm():
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key or api_key == "your_actual_groq_api_key":
        raise RuntimeError(
            "Groq API key is missing. Add it to your local .env file."
        )

    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL).strip()

    return ChatGroq(
        model=model,
        temperature=0.2,
        max_tokens=1000,
        api_key=api_key,
        timeout=45,
        max_retries=2,
    )


def remove_contact_details(text: str) -> str:
    """Remove common email addresses and phone numbers before AI processing."""
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL REDACTED]",
        text,
    )
    text = re.sub(
        r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)",
        "[PHONE REDACTED]",
        text,
    )
    return text


def ask_groq(system_prompt: str, user_prompt: str) -> str:
    try:
        response = get_llm().invoke([
            ("system", system_prompt),
            ("human", user_prompt),
        ])

        content = response.content

        if isinstance(content, str):
            answer = content.strip()
        else:
            answer = "\n".join(
                str(part.get("text", ""))
                if isinstance(part, dict)
                else str(part)
                for part in content
            ).strip()

        if not answer:
            raise RuntimeError("Groq returned an empty response.")

        return answer

    except Exception as exc:
        # Avoid displaying the API key or raw request details.
        raise RuntimeError(
            "AI request failed. Check your API key, model availability, "
            "internet connection, and Groq usage limits."
        ) from exc


def generate_resume_summary(
    resume_text: str,
    extracted_skills: list[Any] | None = None,
) -> str:
    if not isinstance(resume_text, str) or not resume_text.strip():
        raise ValueError("No resume text is available to summarize.")

    safe_text = remove_contact_details(
        resume_text.strip()[:MAX_RESUME_CHARS]
    )

    skills = []
    for item in extracted_skills or []:
        if isinstance(item, dict):
            skill = item.get("skill") or item.get("name")
        else:
            skill = item

        if skill and str(skill).strip():
            skills.append(str(skill).strip())

    verified_skills = ", ".join(dict.fromkeys(skills)) or "Not supplied"

    return ask_groq(
        system_prompt=(
            "You are a professional resume analysis assistant. "
            "Summarize only evidence present in the supplied resume. "
            "Do not invent qualifications, experience, achievements, "
            "or skills. Clearly distinguish missing information."
        ),
        user_prompt=f"""
Create a professional resume analysis with these sections:
1. Candidate profile
2. Key strengths
3. Relevant projects or experience
4. Potential job-role directions
5. Areas that need more evidence

Detected skills from the application's skill extractor:
{verified_skills}

Resume text:
{safe_text}

Keep the response clear, specific, and concise.
""",
    )


def explain_job_match(match_result: dict[str, Any]) -> str:
    if not isinstance(match_result, dict):
        raise ValueError("The job match must be a dictionary.")

    role = (
        match_result.get("job_role")
        or match_result.get("role")
        or match_result.get("job_title")
        or "Recommended role"
    )

    score = (
        match_result.get("match_score")
        or match_result.get("fit_score")
        or match_result.get("score")
        or "Not provided"
    )

    matched_required = match_result.get(
        "matched_required_skills", []
    )
    missing_required = match_result.get(
        "missing_required_skills", []
    )
    matched_preferred = match_result.get(
        "matched_preferred_skills", []
    )
    missing_preferred = match_result.get(
        "missing_preferred_skills", []
    )

    return ask_groq(
        system_prompt=(
            "You are a career guidance assistant. Explain the supplied "
            "job match honestly. Never change or recalculate the supplied "
            "score. A skill-match score is not a probability of being hired. "
            "Do not claim that missing skills are definitely absent from "
            "the candidate's real abilities; they may simply be missing "
            "from the resume."
        ),
        user_prompt=f"""
Explain this resume-to-job-role match.

Role: {role}
Application's skill-fit score: {score}

Matched required skills: {matched_required}
Missing required skills: {missing_required}
Matched preferred skills: {matched_preferred}
Missing preferred skills: {missing_preferred}

Use these sections:
1. Why this role may fit
2. Strongest matching skills
3. Most important skills to learn next
4. Practical next steps

Prioritize missing required skills before preferred skills.
Do not invent job requirements or candidate experience.
""",
    )


def is_api_configured() -> bool:
    key = os.getenv("GROQ_API_KEY", "").strip()
    return bool(
        key
        and key != "your_actual_groq_api_key"
    )
