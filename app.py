"""AI Resume Analyzer - integrated Streamlit application.

This UI connects the project's existing parser, skill extractor, job matcher,
and roadmap generator. Role-fit scores are skill-coverage estimates, not hiring
probabilities.
"""

from __future__ import annotations

import csv
import inspect
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from components.styles import load_css
from components.uploader import render_uploader
from processors.document_processor import extract_document_text

BASE_DIR = Path(__file__).resolve().parent

# Load backend modules defensively. The app can still open and show diagnostics
# if a module has an import problem.
def _import_optional_module(module_name: str):
    try:
        module = __import__(module_name)
        return module, ""
    except Exception as exc:  # shown in the app if the feature is needed
        return None, str(exc)


resume_parser_module, RESUME_PARSER_IMPORT_ERROR = _import_optional_module("resume_parser")
skill_extractor_module, SKILL_IMPORT_ERROR = _import_optional_module("skill_extractor")
job_matcher_module, JOB_IMPORT_ERROR = _import_optional_module("job_matcher")
roadmap_generator_module, ROADMAP_IMPORT_ERROR = _import_optional_module("roadmap_generator")
llm_service_module, LLM_IMPORT_ERROR = _import_optional_module("llm_service")
chatbot_service_module, CHATBOT_IMPORT_ERROR = _import_optional_module(
    "chatbot_service"
)


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()


# -----------------------------------------------------------------------------
# Compatibility and normalization helpers
# -----------------------------------------------------------------------------
def find_function(module: Any, names: tuple[str, ...]):
    if module is None:
        return None
    for name in names:
        value = getattr(module, name, None)
        if callable(value):
            return value
    return None


def call_compatible(function, values: dict[str, Any], fallback_kwargs: dict[str, Any]):
    """Call a function with the subset of known keyword arguments it accepts."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(**fallback_kwargs)

    parameters = signature.parameters
    has_var_keyword = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    )
    kwargs = {
        name: values[name]
        for name, param in parameters.items()
        if name in values
        and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    missing = [
        param.name
        for param in parameters.values()
        if param.default is inspect.Parameter.empty
        and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and param.name not in kwargs
    ]
    if missing:
        raise TypeError(
            f"Could not map required parameter(s) for {function.__name__}: "
            + ", ".join(missing)
        )
    if has_var_keyword and not kwargs:
        kwargs = fallback_kwargs
    return function(**kwargs)


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def normalize_columns(row: dict[str, Any]) -> dict[str, Any]:
    return {
        re.sub(r"[^a-z0-9]+", "_", str(key or "").strip().casefold()).strip("_"): value
        for key, value in row.items()
        if key is not None
    }


def field_value(row: dict[str, Any], *names: str) -> Any:
    normalized = normalize_columns(row)
    for name in names:
        key = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")
        value = normalized.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def split_skill_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        value = value.get("skills", list(value.values()))
    if isinstance(value, (list, tuple, set)):
        parts = list(value)
    else:
        parts = re.split(r"[|;,\n]+", str(value))

    result = []
    seen = set()
    for part in parts:
        if isinstance(part, dict):
            part = part.get("skill") or part.get("name") or part.get("skill_name") or ""
        item = str(part).strip()
        if item and normalize_key(item) not in seen:
            seen.add(normalize_key(item))
            result.append(item)
    return result


def get_document_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("text", "raw_text", "content", "extracted_text", "resume_text"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def normalize_skills(extracted: Any) -> list[dict[str, str]]:
    if isinstance(extracted, dict):
        items = (
            extracted.get("skills")
            or extracted.get("extracted_skills")
            or extracted.get("detected_skills")
            or []
        )
    elif isinstance(extracted, (list, tuple, set)):
        items = list(extracted)
    else:
        items = []

    output = []
    seen = set()
    for item in items:
        if isinstance(item, str):
            name, category = item.strip(), "Other"
        elif isinstance(item, dict):
            name = str(
                item.get("skill") or item.get("name") or item.get("skill_name")
                or item.get("technology") or ""
            ).strip()
            category = str(item.get("category") or item.get("type") or "Other").strip()
        else:
            name, category = str(item).strip(), "Other"
        key = normalize_key(name)
        if name and key not in seen:
            seen.add(key)
            output.append({"skill": name, "category": category or "Other"})
    return output


def display_skill_list(items: Any) -> list[str]:
    if items is None:
        return []
    if isinstance(items, dict):
        items = items.get("skills", list(items.values()))
    return split_skill_values(items)


def score_as_percent(value: Any) -> float | None:
    """Normalize common score representations to a value from 0 to 100."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
        if not match:
            return None
        try:
            number = float(match.group())
        except ValueError:
            return None
    if 0 <= number <= 1:
        number *= 100
    return max(0.0, min(100.0, number))


def get_job_title(job: dict[str, Any]) -> str:
    return str(
        job.get("job_role") or job.get("role") or job.get("job_title")
        or job.get("title") or job.get("name") or "Recommended role"
    ).strip()


def normalize_job_matches(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        for key in ("matches", "jobs", "recommendations", "job_matches", "results"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
        else:
            value = [value] if any(
                key in value for key in ("job_role", "role", "title", "job_title")
            ) else []
    if not isinstance(value, (list, tuple)):
        return []
    jobs = []
    for item in value:
        if isinstance(item, dict):
            jobs.append(dict(item))
        elif isinstance(item, str):
            jobs.append({"job_role": item})
    return jobs


# -----------------------------------------------------------------------------
# CSV catalog: used to explain recommendations and recover gracefully when the
# job matcher returns a role without its required/preferred skill lists.
# -----------------------------------------------------------------------------
def find_csv_path(filename: str) -> Path | None:
    candidates = [
        BASE_DIR / "data" / filename,
        BASE_DIR / "Data" / filename,
    ]
    return next((path for path in candidates if path.exists()), None)


def load_csv_rows(filename: str) -> list[dict[str, Any]]:
    path = find_csv_path(filename)
    if path is None:
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return [normalize_columns(row) for row in csv.DictReader(stream)]
    except (OSError, csv.Error):
        return []


SKILL_DICTIONARY_ROWS = load_csv_rows("Skill_dictionary.csv")
JOB_CATALOG_ROWS = load_csv_rows("Job_roles.csv")


def build_alias_map() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for row in SKILL_DICTIONARY_ROWS:
        canonical = str(row.get("skill") or row.get("skill_name") or "").strip()
        if not canonical:
            continue
        aliases[normalize_key(canonical)] = canonical
        for alias in split_skill_values(row.get("aliases")):
            aliases[normalize_key(alias)] = canonical
    return aliases


SKILL_ALIAS_MAP = build_alias_map()


def canonical_skill(value: str) -> str:
    key = normalize_key(value)
    return SKILL_ALIAS_MAP.get(key, str(value).strip())


def canonical_skill_set(values: list[str]) -> set[str]:
    return {normalize_key(canonical_skill(value)) for value in values if str(value).strip()}


def catalog_role_map() -> dict[str, dict[str, Any]]:
    mapping = {}
    for row in JOB_CATALOG_ROWS:
        title = field_value(row, "job_role", "role", "job_title", "title")
        if title:
            mapping[normalize_key(title)] = row
    return mapping


JOB_CATALOG_BY_TITLE = catalog_role_map()


def role_requirements(job: dict[str, Any]) -> tuple[list[str], list[str]]:
    catalog = JOB_CATALOG_BY_TITLE.get(normalize_key(get_job_title(job)), {})
    required_raw = (
        field_value(job, "required_skills", "essential_skills", "skills_required", "must_have_skills")
        or field_value(catalog, "required_skills", "essential_skills", "skills_required", "must_have_skills")
    )
    preferred_raw = (
        field_value(job, "preferred_skills", "desirable_skills", "nice_to_have_skills")
        or field_value(catalog, "preferred_skills", "desirable_skills", "nice_to_have_skills")
    )
    return split_skill_values(required_raw), split_skill_values(preferred_raw)


def calculate_skill_fit(
    required: list[str], preferred: list[str], candidate_skills: list[str]
) -> tuple[float | None, list[str], list[str], list[str], list[str]]:
    candidate_set = canonical_skill_set(candidate_skills)

    def partition(values: list[str]) -> tuple[list[str], list[str]]:
        matched, missing = [], []
        for skill in values:
            if normalize_key(canonical_skill(skill)) in candidate_set:
                matched.append(skill)
            else:
                missing.append(skill)
        return matched, missing

    matched_required, missing_required = partition(required)
    matched_preferred, missing_preferred = partition(preferred)

    required_score = (
        100 * len(matched_required) / len(required) if required else None
    )
    preferred_score = (
        100 * len(matched_preferred) / len(preferred) if preferred else None
    )
    if required_score is not None and preferred_score is not None:
        # Required skills matter more than preferred skills.
        score = 0.8 * required_score + 0.2 * preferred_score
    elif required_score is not None:
        score = required_score
    elif preferred_score is not None:
        score = preferred_score
    else:
        score = None
    return score, matched_required, missing_required, matched_preferred, missing_preferred


def enrich_job_matches(jobs: list[dict[str, Any]], candidate_skills: list[str]) -> list[dict[str, Any]]:
    enriched = []
    for original in jobs:
        job = dict(original)
        title = get_job_title(job)
        if not title:
            continue
        job["job_role"] = title
        required, preferred = role_requirements(job)

        # Prefer calculated skill coverage when the dataset contains a skill list:
        # that ensures the graph's percentages mean the same thing for every role.
        if required or preferred:
            score, matched_req, missing_req, matched_pref, missing_pref = calculate_skill_fit(
                required, preferred, candidate_skills
            )
            job["fit_score"] = score
            job["score_basis"] = "Calculated from the role skills in the job dataset"
            job["required_skills"] = required
            job["preferred_skills"] = preferred
            job["matched_required_skills"] = matched_req
            job["missing_required_skills"] = missing_req
            job["matched_preferred_skills"] = matched_pref
            job["missing_preferred_skills"] = missing_pref
        else:
            # If role requirements are unavailable, use the backend matcher's score.
            original_score = next(
                (job.get(key) for key in ("match_score", "score", "match_percentage", "fit_score") if job.get(key) is not None),
                None,
            )
            job["fit_score"] = score_as_percent(original_score)
            job["score_basis"] = "Provided by the job-matching module" if original_score is not None else "No score or role-skill data available"
            for key in (
                "matched_required_skills", "missing_required_skills",
                "matched_preferred_skills", "missing_preferred_skills",
            ):
                job[key] = display_skill_list(job.get(key))
        enriched.append(job)

    # Keep the strongest fit at the top. Missing scores remain below scored roles.
    enriched.sort(key=lambda item: item.get("fit_score") if item.get("fit_score") is not None else -1, reverse=True)
    return enriched


def fallback_jobs_from_catalog(candidate_skills: list[str], limit: int = 10) -> list[dict[str, Any]]:
    """Create transparent data-driven recommendations if the matcher fails/returns nothing."""
    jobs = []
    for row in JOB_CATALOG_ROWS:
        title = field_value(row, "job_role", "role", "job_title", "title")
        if not title:
            continue
        required, preferred = role_requirements({"job_role": title})
        if not required and not preferred:
            continue
        score, matched_req, missing_req, matched_pref, missing_pref = calculate_skill_fit(
            required, preferred, candidate_skills
        )
        jobs.append({
            "job_role": str(title),
            "description": field_value(row, "description", "job_description", "summary") or "",
            "required_skills": required,
            "preferred_skills": preferred,
            "matched_required_skills": matched_req,
            "missing_required_skills": missing_req,
            "matched_preferred_skills": matched_pref,
            "missing_preferred_skills": missing_pref,
            "fit_score": score,
            "score_basis": "Calculated from the role skills in the job dataset",
        })
    jobs.sort(key=lambda item: item.get("fit_score") if item.get("fit_score") is not None else -1, reverse=True)
    return jobs[:limit]


def describe_job(job: dict[str, Any]) -> str:
    return str(job.get("description") or job.get("job_description") or "").strip()


# -----------------------------------------------------------------------------
# Backend workflow
# -----------------------------------------------------------------------------
def analyze_resume_text(filename: str, text: str, parsed_metadata: Any = None) -> dict[str, Any]:
    output: dict[str, Any] = {
        "skills": [], "skill_result": None, "jobs": [], "roadmaps": {}, "errors": []
    }

    parsed_resume = dict(parsed_metadata) if isinstance(parsed_metadata, dict) else {}
    parsed_resume.update({
        "filename": filename,
        "text": text,
        "raw_text": text,
        "resume_text": text,
    })

    extract_skills_func = find_function(
        skill_extractor_module,
        ("extract_skills_from_resume", "extract_skills", "get_resume_skills"),
    )
    if extract_skills_func is None:
        output["errors"].append(
            "Skill extraction is unavailable. " + (SKILL_IMPORT_ERROR or "No supported extraction function was found.")
        )
        return output

    try:
        skill_result = call_compatible(
            extract_skills_func,
            {
                "parsed_resume": parsed_resume,
                "resume_data": parsed_resume,
                "resume": text,
                "resume_text": text,
                "text": text,
                "raw_text": text,
                "document_text": text,
                "document": parsed_resume,
                "content": text,
            },
            {"parsed_resume": parsed_resume},
        )
        output["skill_result"] = skill_result
        output["skills"] = normalize_skills(skill_result)
    except Exception as exc:
        output["errors"].append(f"Skill extraction failed: {exc}")
        return output

    candidate_skill_names = [item["skill"] for item in output["skills"]]

    match_func = find_function(
        job_matcher_module,
        ("match_jobs", "match_job_roles", "recommend_jobs", "find_matching_jobs"),
    )
    matcher_jobs: list[dict[str, Any]] = []
    if match_func is not None:
        try:
            matches = call_compatible(
                match_func,
                {
                    "resume_skills": skill_result,
                    "extracted_skills": skill_result,
                    "skills": skill_result,
                    "skill_result": skill_result,
                    "experience_years": parsed_resume.get("experience_years"),
                    "top_n": 10,
                    "limit": 10,
                    "n": 10,
                },
                {"resume_skills": skill_result, "experience_years": parsed_resume.get("experience_years"), "top_n": 10},
            )
            matcher_jobs = normalize_job_matches(matches)
        except Exception as exc:
            output["errors"].append(f"Job matching module failed; trying the role CSV instead: {exc}")
    else:
        output["errors"].append(
            "Job matcher unavailable; trying the role CSV instead. " + (JOB_IMPORT_ERROR or "No supported matching function was found.")
        )

    output["jobs"] = enrich_job_matches(matcher_jobs, candidate_skill_names)
    if not output["jobs"]:
        output["jobs"] = fallback_jobs_from_catalog(candidate_skill_names, limit=10)
        if output["jobs"]:
            output["errors"].append(
                "Recommendations were generated directly from data/Job_roles.csv because the matcher returned no roles."
            )

    # Build a roadmap for every recommended role, so users can compare learning paths.
    roadmap_func = find_function(
        roadmap_generator_module,
        ("generate_roadmap", "create_roadmap", "build_roadmap"),
    )
    if output["jobs"] and roadmap_func is None:
        output["errors"].append(
            "The roadmap generator is unavailable. Missing-skill lists are still shown for every role. "
            + (ROADMAP_IMPORT_ERROR or "No supported roadmap function was found.")
        )
    elif roadmap_func is not None:
        for job in output["jobs"]:
            try:
                roadmap = call_compatible(
                    roadmap_func,
                    {
                        "match_result": job,
                        "job_match": job,
                        "best_match": job,
                        "match": job,
                        "job": job,
                        "weeks": 6,
                        "duration_weeks": 6,
                        "num_weeks": 6,
                    },
                    {"match_result": job, "weeks": 6},
                )
                output["roadmaps"][get_job_title(job)] = roadmap
            except Exception as exc:
                output["errors"].append(f"Roadmap for {get_job_title(job)} could not be built: {exc}")

    return output


def render_roadmap(roadmap: Any) -> None:
    if roadmap is None:
        st.info("No generated weekly roadmap is available for this role. Use the missing-skill list above as your learning checklist.")
        return

    format_func = find_function(roadmap_generator_module, ("format_roadmap",))
    if format_func:
        try:
            formatted = call_compatible(
                format_func,
                {"roadmap": roadmap, "plan": roadmap, "result": roadmap},
                {"roadmap": roadmap},
            )
            if isinstance(formatted, str) and formatted.strip():
                st.markdown(formatted)
                return
        except Exception:
            pass

    if isinstance(roadmap, str):
        st.markdown(roadmap)
    elif isinstance(roadmap, dict):
        for key in ("target_role", "duration_weeks", "status", "summary"):
            if roadmap.get(key) is not None:
                st.write(f"**{key.replace('_', ' ').title()}:** {roadmap[key]}")
        weekly_plan = roadmap.get("weekly_plan") or roadmap.get("weeks") or roadmap.get("plan")
        if isinstance(weekly_plan, list):
            for index, week in enumerate(weekly_plan, start=1):
                if isinstance(week, dict):
                    heading = week.get("week") or week.get("title") or f"Week {index}"
                    with st.expander(str(heading), expanded=(index == 1)):
                        st.markdown("\n".join(
                            f"- **{str(key).replace('_', ' ').title()}:** {value}"
                            for key, value in week.items()
                        ))
                else:
                    with st.expander(f"Week {index}", expanded=(index == 1)):
                        st.write(week)
        elif weekly_plan:
            st.write(weekly_plan)
        else:
            st.json(roadmap)
    elif isinstance(roadmap, (list, tuple)):
        for index, item in enumerate(roadmap, start=1):
            st.write(f"{index}. {item}")
    else:
        st.write(roadmap)


# -----------------------------------------------------------------------------
# Sidebar and header
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="font-size:1.35rem;font-weight:700;margin-bottom:0.25rem;">
            📄 AI Resume Analyzer
        </div>
        <div style="color:#94A3B8;font-size:0.82rem;margin-bottom:1.4rem;">
            Intelligent Recruitment Platform
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Analysis workflow")
    st.markdown(
        "1. Upload resume\n2. Extract resume text\n3. Detect skills\n4. Score job-role fit\n5. Review skill gaps\n6. Compare learning roadmaps"
    )
    st.divider()
    st.caption("AI Resume Analyzer · Major Project · 2026")
    st.caption("Use fictional/sample resumes for testing.")

st.markdown(
    """
    <div class="main-header">
        <h1>AI Resume Analyzer</h1>
        <p>Understand your skills, discover suitable roles, and plan what to learn next.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="upload-card">
        <div class="upload-title">Upload Resume</div>
        <div class="upload-description">
            Upload a text-based PDF or DOCX resume. The analyzer will extract skills,
            compare them with job-role requirements, and show suggested learning paths.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_files = render_uploader()

if uploaded_files:
    left, right = st.columns([3, 1])
    with left:
        st.success(f"{len(uploaded_files)} resume file(s) ready for analysis.")
    with right:
        analyze_button = st.button(
            "Analyze Resume(s) →", type="primary", use_container_width=True
        )

    if analyze_button:
        processed_documents = []
        progress = st.progress(0, text="Preparing resume analysis…")

        for index, uploaded_file in enumerate(uploaded_files):
            try:
                file_bytes = uploaded_file.getvalue()
                document_result = extract_document_text(uploaded_file.name, file_bytes)
                text = get_document_text(document_result)

                parsed_metadata: dict[str, Any] = {}
                parser_func = find_function(
                    resume_parser_module,
                    ("parse_resume_file", "parse_resume", "parse_document"),
                )
                if parser_func is not None:
                    try:
                        parsed = call_compatible(
                            parser_func,
                            {
                                "filename": uploaded_file.name,
                                "file_name": uploaded_file.name,
                                "file_bytes": file_bytes,
                                "file_content": file_bytes,
                                "content": file_bytes,
                                "uploaded_file": uploaded_file,
                            },
                            {"filename": uploaded_file.name, "file_bytes": file_bytes},
                        )
                        if isinstance(parsed, dict):
                            parsed_metadata = parsed
                        if not text.strip():
                            text = get_document_text(parsed)
                    except Exception as parser_error:
                        parsed_metadata = {"parser_note": str(parser_error)}

                if not text.strip():
                    raise ValueError("No readable text was extracted. If this is a scanned PDF, OCR may be needed.")

                if isinstance(document_result, dict):
                    document = dict(document_result)
                else:
                    document = {}
                document.setdefault("filename", uploaded_file.name)
                document.setdefault("extension", uploaded_file.name.rsplit(".", 1)[-1].lower())
                document["text"] = text
                document["word_count"] = int(document.get("word_count") or len(text.split()))
                document["character_count"] = int(document.get("character_count") or len(text))
                for key in ("candidate_name", "email", "phone", "experience_years", "education"):
                    if parsed_metadata.get(key) is not None:
                        document[key] = parsed_metadata[key]
                if parsed_metadata.get("parser_note"):
                    document["parser_note"] = parsed_metadata["parser_note"]

                document["analysis"] = analyze_resume_text(
                    uploaded_file.name, text, parsed_metadata
                )
                processed_documents.append(document)
            except Exception as exc:
                st.error(f"Could not process {uploaded_file.name}: {exc}")
            finally:
                progress.progress((index + 1) / len(uploaded_files), text="Analyzing resumes…")

        st.session_state["processed_documents"] = processed_documents
        st.session_state.pop("ai_summary", None)
        st.session_state.pop("ai_match_explanation", None)
        if processed_documents:
            st.success("Analysis finished. Your results are below.")
        else:
            st.warning("No resume could be analyzed. Check the file type and backend module setup.")


# -----------------------------------------------------------------------------
# Results dashboard. Results are kept in session state across Streamlit reruns.
# -----------------------------------------------------------------------------
processed_documents = st.session_state.get("processed_documents", [])
if not processed_documents:
    st.markdown("### What you will get")
    card1, card2, card3 = st.columns(3)
    with card1:
        st.markdown("#### 🎯 Role recommendations")
        st.write("Compare your current skills with job roles in your local job dataset.")
    with card2:
        st.markdown("#### 📊 Match graphs")
        st.write("See skill-fit estimates and matched-versus-missing required skills by role.")
    with card3:
        st.markdown("#### 🧭 Learning roadmap")
        st.write("Find skill gaps and review a week-by-week plan for the role you choose.")

    if not JOB_CATALOG_ROWS:
        st.warning("Job-role data was not found. Check that data/Job_roles.csv exists and contains role names and required/preferred skills.")
    if not SKILL_DICTIONARY_ROWS:
        st.caption("Skill dictionary was not found or is empty. Check data/Skill_dictionary.csv.")
else:
    st.divider()
    st.markdown("## Resume Analysis Dashboard")
    total_words = sum(int(doc.get("word_count", 0) or 0) for doc in processed_documents)
    all_skills = {
        row["skill"].casefold(): row["skill"]
        for doc in processed_documents
        for row in doc.get("analysis", {}).get("skills", [])
    }
    total_recommendations = sum(len(doc.get("analysis", {}).get("jobs", [])) for doc in processed_documents)

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Resumes analyzed", len(processed_documents))
    metric2.metric("Total words", f"{total_words:,}")
    metric3.metric("Unique skills found", len(all_skills))
    metric4.metric("Role recommendations", total_recommendations)

    selected_filename = st.selectbox(
        "Choose a resume to inspect",
        options=[doc["filename"] for doc in processed_documents],
        key="selected_resume_filename",
    )
    selected_document = next(
        doc for doc in processed_documents if doc["filename"] == selected_filename
    )
    analysis = selected_document.get("analysis", {})
    skill_rows = analysis.get("skills", [])
    candidate_skills = [row["skill"] for row in skill_rows]
    job_matches = analysis.get("jobs", [])

    overview_tab, skills_tab, roles_tab, roadmap_tab, text_tab, chatbot_tab = st.tabs(
    [
        "Overview",
        "Skills & Categories",
        "Job Fit & Skill Gaps",
        "Learning Roadmaps",
        "Extracted Text",
        "AI Career Assistant",
    ]
)

    # Overview
    with overview_tab:
        st.markdown(f"### {selected_document['filename']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Word count", f"{int(selected_document.get('word_count', 0) or 0):,}")
        c2.metric("Character count", f"{int(selected_document.get('character_count', 0) or 0):,}")
        c3.metric("Detected skills", len(skill_rows))
        if selected_document.get("candidate_name"):
            st.write(f"**Candidate:** {selected_document['candidate_name']}")
        if selected_document.get("experience_years") is not None:
            st.write(f"**Experience found by parser:** {selected_document['experience_years']} years")
        if selected_document.get("parser_note"):
            st.caption(f"Parser note: {selected_document['parser_note']}")

        if job_matches:
            best = job_matches[0]
            best_score = best.get("fit_score")
            st.markdown("#### Strongest role match in this dataset")
            st.markdown(f"**{get_job_title(best)}**")
            if best_score is not None:
                st.progress(best_score / 100)
                st.metric("Skill-fit estimate", f"{best_score:.1f}%")
                st.caption(best.get("score_basis", ""))
            st.caption("This score measures skill overlap with the role data; it is not a probability of being hired.")
            if best.get("missing_required_skills"):
                st.markdown("**Priority skills to work on:** " + ", ".join(best["missing_required_skills"][:8]))
        else:
            st.warning("No job recommendations are available. Check the role dataset and job-matcher module below.")

        for message in analysis.get("errors", []):
            st.warning(message)

    # Skills and category charts
    with skills_tab:
        st.markdown("### Skills detected in your resume")
        if skill_rows:
            st.dataframe(pd.DataFrame(skill_rows), use_container_width=True, hide_index=True)
            category_counts = pd.DataFrame(skill_rows).groupby("category").size().sort_values(ascending=False)
            st.markdown("#### Skills by category")
            st.bar_chart(category_counts)
            csv_data = pd.DataFrame(skill_rows).to_csv(index=False)
            st.download_button(
                "Download detected skills (CSV)",
                data=csv_data,
                file_name="detected_skills.csv",
                mime="text/csv",
            )
        else:
            st.info("No skills were returned. Check data/Skill_dictionary.csv and the skill_extractor.py module.")

    # Job scores and per-role gap analysis
    with roles_tab:
        st.markdown("### Which job roles match your current skills?")
        st.caption("Higher skill-fit means that more of the skills listed for that role were found in your resume. It does not guarantee eligibility or hiring.")
        scored_jobs = [job for job in job_matches if job.get("fit_score") is not None]
        if scored_jobs:
            score_frame = pd.DataFrame([
                {"Job role": get_job_title(job), "Skill fit (%)": round(float(job["fit_score"]), 1)}
                for job in scored_jobs
            ]).drop_duplicates(subset=["Job role"], keep="first").set_index("Job role")
            st.markdown("#### Role fit comparison")
            st.bar_chart(score_frame)
            st.caption("Scoring: when both lists exist, fit = 80% required-skill coverage + 20% preferred-skill coverage. If only one list exists, coverage uses that list; if role skill lists are unavailable, the matcher score is used.")

            gap_rows = []
            for job in job_matches:
                gap_rows.append({
                    "Job role": get_job_title(job),
                    "Matched required": len(job.get("matched_required_skills", [])),
                    "Missing required": len(job.get("missing_required_skills", [])),
                })
            gap_frame = pd.DataFrame(gap_rows).drop_duplicates(subset=["Job role"], keep="first").set_index("Job role")
            if not gap_frame.empty and (gap_frame["Matched required"].sum() + gap_frame["Missing required"].sum() > 0):
                st.markdown("#### Required skills: matched vs. missing")
                st.bar_chart(gap_frame)
        elif not job_matches:
            st.warning("No job matches returned. Confirm data/Job_roles.csv has required_skills/preferred_skills and that job_matcher.py is working.")

        if job_matches:
            st.markdown("#### Detailed fit for every recommended role")
            for rank, job in enumerate(job_matches, start=1):
                title = get_job_title(job)
                score = job.get("fit_score")
                score_label = f" — {score:.1f}% skill fit" if score is not None else " — score unavailable"
                with st.expander(f"{rank}. {title}{score_label}", expanded=(rank == 1)):
                    description = describe_job(job)
                    if description:
                        st.write(description)
                    if score is not None:
                        st.progress(float(score) / 100)
                    basis = job.get("score_basis")
                    if basis:
                        st.caption(basis)

                    a, b = st.columns(2)
                    with a:
                        st.markdown("**Skills already found**")
                        matched = job.get("matched_required_skills", []) + job.get("matched_preferred_skills", [])
                        st.write(", ".join(dict.fromkeys(matched)) if matched else "No matched-skill details available.")
                        st.markdown("**Required skills matched**")
                        st.write(", ".join(job.get("matched_required_skills", [])) or "None listed / none matched")
                    with b:
                        st.markdown("**Skills to learn next**")
                        missing = job.get("missing_required_skills", []) + job.get("missing_preferred_skills", [])
                        st.write(", ".join(dict.fromkeys(missing)) if missing else "No listed skill gaps for this role.")
                        st.markdown("**Missing required skills (prioritize these)**")
                        st.write(", ".join(job.get("missing_required_skills", [])) or "None listed")
                    if job.get("minimum_experience_years") is not None:
                        st.caption(f"Dataset minimum experience: {job['minimum_experience_years']} years. Verify actual job postings before applying.")
        else:
            st.info("No job recommendations available yet.")

        if not JOB_CATALOG_ROWS:
            st.info("For full per-role skill gap explanations, add job roles and skills to data/Job_roles.csv.")

    # Roadmaps selectable for every recommended role
    with roadmap_tab:
        st.markdown("### Build your next-step learning plan")
        if job_matches:
            titles = [get_job_title(job) for job in job_matches]
            selected_role = st.selectbox(
                "Choose a target job role",
                options=titles,
                key=f"roadmap_role_{selected_filename}",
            )
            chosen_job = next(job for job in job_matches if get_job_title(job) == selected_role)
            missing_required = chosen_job.get("missing_required_skills", [])
            missing_preferred = chosen_job.get("missing_preferred_skills", [])
            st.markdown(f"#### Learning priorities for {selected_role}")
            if missing_required:
                st.markdown("**Priority 1 — Required skills**")
                for skill in missing_required:
                    st.markdown(f"- [ ] {skill}")
            if missing_preferred:
                st.markdown("**Priority 2 — Preferred skills**")
                for skill in missing_preferred:
                    st.markdown(f"- [ ] {skill}")
            if not missing_required and not missing_preferred:
                st.success("No skill gaps were returned for this role. Review job descriptions to find role-specific expectations.")
            st.divider()
            st.markdown("#### Suggested weekly roadmap")
            render_roadmap(analysis.get("roadmaps", {}).get(selected_role))
        else:
            st.info("Run analysis and select a recommended role to see its learning plan.")

    # Extracted text for transparency/debugging
    with text_tab:
        st.markdown("### Extracted resume text")
        st.caption(
            f"File type: {str(selected_document.get('extension', '')).upper()} · "
            f"{int(selected_document.get('word_count', 0) or 0):,} words · "
            f"{int(selected_document.get('character_count', 0) or 0):,} characters"
        )
        st.text_area(
            "Text extracted from the selected resume",
            value=selected_document.get("text", ""),
            height=360,
            key=f"selected_extracted_text_{selected_filename}",
        )
        st.download_button(
            "Download extracted text",
            data=selected_document.get("text", ""),
            file_name=f"{selected_filename}.txt",
            mime="text/plain",
        )
            # ==========================================
    # AI CAREER ASSISTANT
    # ==========================================

    with chatbot_tab:

        st.markdown("## 🤖 AI Resume & Career Assistant")

        st.write(
            "Ask questions about your resume, detected skills, "
            "recommended job roles, missing skills, and learning plans."
        )

        st.caption(
            "Powered by Groq and grounded in your resume analysis."
        )

        # Prepare the actual analysis as chatbot context.
        chat_context = {
            "resume_text": selected_document.get("text", ""),
            "word_count": selected_document.get("word_count", 0),
            "skills": candidate_skills,
            "job_matches": [
                {
                    "job_role": get_job_title(job),
                    "fit_score": job.get("fit_score"),
                    "score_basis": job.get("score_basis"),
                    "description": describe_job(job),
                    "matched_required_skills": job.get(
                        "matched_required_skills", []
                    ),
                    "missing_required_skills": job.get(
                        "missing_required_skills", []
                    ),
                    "matched_preferred_skills": job.get(
                        "matched_preferred_skills", []
                    ),
                    "missing_preferred_skills": job.get(
                        "missing_preferred_skills", []
                    ),
                }
                for job in job_matches
            ],
            "roadmaps": analysis.get("roadmaps", {}),
        }

        # Maintain a separate conversation for each selected resume.
        chat_suffix = normalize_key(selected_filename)
        chat_key = f"career_chat_{chat_suffix}"

        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        st.markdown("### Suggested questions")

        st.markdown(
            """
            - Why is this job role a good match for me?
            - Which skills should I learn first?
            - What are the biggest gaps in my resume?
            - How can I improve my resume summary?
            - Give me interview questions for my strongest role.
            """
        )

        # Display previous messages.
        for message in st.session_state[chat_key]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input.
        user_prompt = st.chat_input(
            "Ask anything about your resume or career...",
            key=f"career_chat_input_{chat_suffix}",
        )

        if user_prompt and user_prompt.strip():

            # Save and display the current question.
            user_prompt = user_prompt.strip()

            previous_history = list(st.session_state[chat_key])

            st.session_state[chat_key].append({
                "role": "user",
                "content": user_prompt,
            })

            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):

                if chatbot_service_module is None:

                    st.error(
                        "The chatbot backend could not be imported."
                    )

                    if CHATBOT_IMPORT_ERROR:
                        st.code(CHATBOT_IMPORT_ERROR)

                else:

                    try:
                        with st.spinner(
                            "Reviewing your resume and preparing an answer..."
                        ):

                            answer = (
                                chatbot_service_module.answer_resume_question(
                                    user_message=user_prompt,
                                    resume_context=chat_context,
                                    chat_history=previous_history,
                                )
                            )

                        st.markdown(answer)

                        st.session_state[chat_key].append({
                            "role": "assistant",
                            "content": answer,
                        })

                    except Exception as exc:
                        st.error(f"Chatbot error: {exc}")

        st.caption(
            "For privacy, test with fictional resumes. Common emails and "
            "phone numbers are redacted before chatbot requests, but this "
            "is not complete removal of personal information."
        )

    
    # ==========================================
    # DOWNLOADABLE ANALYSIS REPORT
    # ==========================================

    st.divider()
    st.markdown("## Download Your Analysis Report")
    st.write(
        "Download a combined report containing detected skills, "
        "recommended job roles, fit scores, missing skills, and learning roadmaps."
    )

    target_role = st.session_state.get(
        f"roadmap_role_{selected_filename}",
        "Not selected",
    )

    report_lines = [
        "# AI Resume Analyzer — Analysis Report",
        "",
        f"Resume: {selected_filename}",
        f"Target role: {target_role}",
        f"Words analyzed: {selected_document.get('word_count', 0)}",
        f"Characters analyzed: {selected_document.get('character_count', 0)}",
        "",
        "## 1. Extracted Skills",
        "",
    ]

    if skill_rows:
        for item in skill_rows:
            report_lines.append(
                f"- {item.get('skill', 'Unknown')} "
                f"(Category: {item.get('category', 'Other')})"
            )
    else:
        report_lines.append("No skills were detected.")

    report_lines.extend([
        "",
        "## 2. Recommended Job Roles",
        "",
    ])

    if job_matches:
        for rank, job in enumerate(job_matches, start=1):
            title = get_job_title(job)
            score = job.get("fit_score")

            report_lines.extend([
                f"### {rank}. {title}",
                "",
                f"Skill-fit estimate: "
                f"{f'{score:.1f}%' if score is not None else 'Unavailable'}",
                f"Score basis: {job.get('score_basis', 'Not specified')}",
                "",
                f"Description: {describe_job(job) or 'Not provided'}",
                "",
                "**Matched required skills:**",
                ", ".join(job.get("matched_required_skills", [])) or "None listed",
                "",
                "**Missing required skills:**",
                ", ".join(job.get("missing_required_skills", [])) or "None listed",
                "",
                "**Matched preferred skills:**",
                ", ".join(job.get("matched_preferred_skills", [])) or "None listed",
                "",
                "**Missing preferred skills:**",
                ", ".join(job.get("missing_preferred_skills", [])) or "None listed",
                "",
                "**Learning roadmap:**",
            ])

            roadmap = analysis.get("roadmaps", {}).get(title)
            formatted_roadmap = None

            format_func = find_function(
                roadmap_generator_module,
                ("format_roadmap",),
            )

            if roadmap is not None and format_func:
                try:
                    formatted_roadmap = call_compatible(
                        format_func,
                        {
                            "roadmap": roadmap,
                            "plan": roadmap,
                            "result": roadmap,
                        },
                        {"roadmap": roadmap},
                    )
                except Exception:
                    formatted_roadmap = None

            if isinstance(formatted_roadmap, str) and formatted_roadmap.strip():
                report_lines.append(formatted_roadmap)
            elif roadmap is not None:
                report_lines.append(str(roadmap))
            else:
                report_lines.append(
                    "No weekly roadmap was generated. Use the missing-skill "
                    "lists above as learning priorities."
                )

            report_lines.append("")
    else:
        report_lines.append("No job-role recommendations were available.")

    report_lines.extend([
        "",
        "## 3. Interpretation",
        "",
        "The skill-fit estimate measures overlap with the skills listed "
        "in the application's job-role dataset. It is not a probability "
        "of getting hired or a guarantee of eligibility.",
        "",
        "Verify required skills, experience, and qualifications against "
        "the actual job description before applying.",
    ])

    # Include AI insights if they have already been generated.
    if st.session_state.get("ai_summary"):
        report_lines.extend([
            "",
            "## 4. AI Resume Summary",
            "",
            st.session_state["ai_summary"],
        ])

    if st.session_state.get("ai_match_explanation"):
        report_lines.extend([
            "",
            "## 5. AI Job-Match Explanation",
            "",
            st.session_state["ai_match_explanation"],
        ])

    report_text = "\n".join(report_lines)
        # Display the report directly in Streamlit
    st.divider()
    st.markdown("## 📊 Complete Resume Analysis Report")

    with st.container(border=True):
        st.markdown(report_text)

    role_report_rows = []
    for job in job_matches:
        role_report_rows.append({
            "job_role": get_job_title(job),
            "skill_fit_percent": job.get("fit_score"),
            "matched_required_skills": ", ".join(
                job.get("matched_required_skills", [])
            ),
            "missing_required_skills": ", ".join(
                job.get("missing_required_skills", [])
            ),
            "matched_preferred_skills": ", ".join(
                job.get("matched_preferred_skills", [])
            ),
            "missing_preferred_skills": ", ".join(
                job.get("missing_preferred_skills", [])
            ),
        })

    safe_filename = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        Path(selected_filename).stem,
    )

    download_col1, download_col2 = st.columns(2)

    with download_col1:
        st.download_button(
            "📄 Download Complete Analysis Report",
            data=report_text,
            file_name=f"{safe_filename}_analysis_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with download_col2:
        st.download_button(
            "📊 Download Job Comparison (CSV)",
            data=pd.DataFrame(role_report_rows).to_csv(index=False),
            file_name=f"{safe_filename}_job_comparison.csv",
            mime="text/csv",
            use_container_width=True,
        )
# Optional AI functions. Core role scoring and charts do not require a key.
    st.divider()
    st.markdown("## Optional AI insights")
    st.caption("AI summaries need llm_service.py, its dependencies, internet access, and a valid GROQ_API_KEY in your local .env file.")
    summary_func = find_function(llm_service_module, ("generate_resume_summary",))
    explanation_func = find_function(llm_service_module, ("explain_job_match",))

    if summary_func is None:
        st.info("AI insights are optional; resume parsing, skills, role-fit graphs, and learning-gap views can run without them.")
        if LLM_IMPORT_ERROR:
            with st.expander("AI module setup details"):
                st.code(LLM_IMPORT_ERROR)
    else:
        ai_left, ai_right = st.columns(2)
        with ai_left:
            if st.button("Generate AI resume summary", use_container_width=True):
                try:
                    summary = call_compatible(
                        summary_func,
                        {
                            "resume_text": selected_document.get("text", ""),
                            "text": selected_document.get("text", ""),
                            "extracted_skills": candidate_skills,
                            "skills": candidate_skills,
                        },
                        {"resume_text": selected_document.get("text", ""), "extracted_skills": candidate_skills},
                    )
                    st.session_state["ai_summary"] = str(summary)
                except Exception as exc:
                    st.error(f"Could not generate summary: {exc}")
            if st.session_state.get("ai_summary"):
                st.markdown("**AI-generated summary**")
                st.write(st.session_state["ai_summary"])

        with ai_right:
            if st.button("Explain strongest job match", use_container_width=True):
                if not job_matches:
                    st.warning("No job match is available to explain yet.")
                elif explanation_func is None:
                    st.warning("explain_job_match() was not found in llm_service.py.")
                else:
                    try:
                        explanation = call_compatible(
                            explanation_func,
                            {"match_result": job_matches[0], "job_match": job_matches[0], "match": job_matches[0]},
                            {"match_result": job_matches[0]},
                        )
                        st.session_state["ai_match_explanation"] = str(explanation)
                    except Exception as exc:
                        st.error(f"Could not explain job match: {exc}")
            if st.session_state.get("ai_match_explanation"):
                st.markdown("**AI explanation**")
                st.write(st.session_state["ai_match_explanation"])

    st.caption("Privacy note: resume text can contain personal information. Use fictional resumes during development, and do not send personal resume text to external AI services without appropriate permission.")

