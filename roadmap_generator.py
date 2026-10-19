"""
Roadmap Generator for AI Resume Analyzer.

Creates a structured learning roadmap from the missing skills
identified by job_matcher.py.

This baseline is rule-based and does not require an LLM or API key.
"""

from typing import Any


# ============================================================
# LEARNING ACTIVITY TEMPLATES
# ============================================================

SKILL_ACTIVITIES = {
    "python": {
        "activity": "Study core concepts and build a small Python program.",
        "practice": "Create a script that accepts input, processes data, and produces output.",
    },
    "sql": {
        "activity": "Practise SQL filtering, sorting, joins, and aggregation.",
        "practice": "Create sample tables and answer five business questions using SQL.",
    },
    "pandas": {
        "activity": "Learn DataFrames, filtering, grouping, and data transformation.",
        "practice": "Load a dataset, clean missing values, and summarize its columns.",
    },
    "machine learning": {
        "activity": "Study supervised learning, model training, and evaluation.",
        "practice": "Train a simple classifier and evaluate it on held-out data.",
    },
    "statistics": {
        "activity": "Review descriptive statistics, distributions, and basic inference.",
        "practice": "Analyse a dataset and explain its mean, median, spread, and patterns.",
    },
    "data cleaning": {
        "activity": "Practise handling missing values, duplicates, and inconsistent data.",
        "practice": "Clean a deliberately messy dataset and document each transformation.",
    },
    "exploratory data analysis": {
        "activity": "Learn how to inspect datasets and discover useful patterns.",
        "practice": "Perform an exploratory analysis and summarize three findings.",
    },
    "data visualization": {
        "activity": "Study chart selection, labels, and clear visual communication.",
        "practice": "Create three visualizations that explain patterns in a dataset.",
    },
    "power bi": {
        "activity": "Learn data import, relationships, measures, and dashboard design.",
        "practice": "Build a dashboard with appropriate charts and key performance indicators.",
    },
    "tableau": {
        "activity": "Practise data connections, charts, filters, and dashboards.",
        "practice": "Build an interactive dashboard from a public dataset.",
    },
    "git": {
        "activity": "Learn repositories, commits, branches, and version control.",
        "practice": "Create a repository and commit a small project with a clear README.",
    },
    "deep learning": {
        "activity": "Study neural networks, loss functions, training, and evaluation.",
        "practice": "Train a small neural-network model using a suitable dataset.",
    },
    "natural language processing": {
        "activity": "Learn tokenization, text preprocessing, and text representations.",
        "practice": "Build a small text-classification or text-analysis experiment.",
    },
    "large language models": {
        "activity": "Study LLM capabilities, limitations, and grounded prompting.",
        "practice": "Build a small application that answers questions using supplied text.",
    },
    "prompt engineering": {
        "activity": "Practise clear instructions, structured outputs, and examples.",
        "practice": "Compare several prompts on the same task and record their results.",
    },
    "embeddings": {
        "activity": "Learn semantic embeddings and similarity measures.",
        "practice": "Encode sample sentences and compare their semantic similarity.",
    },
    "retrieval-augmented generation": {
        "activity": "Study document chunking, retrieval, context construction, and generation.",
        "practice": "Build a small question-answering pipeline over a set of documents.",
    },
    "langchain": {
        "activity": "Learn how to compose model, prompt, and retrieval components.",
        "practice": "Create a small reusable chain and test it with multiple inputs.",
    },
    "faiss": {
        "activity": "Learn vector indexing, similarity search, and metadata handling.",
        "practice": "Create an index for sample embeddings and retrieve relevant items.",
    },
    "vector databases": {
        "activity": "Study vector storage, nearest-neighbour search, and retrieval design.",
        "practice": "Store sample vectors and test retrieval against several queries.",
    },
    "streamlit": {
        "activity": "Learn Streamlit state, inputs, forms, and result display.",
        "practice": "Build a small data application with validation and clear error messages.",
    },
    "fastapi": {
        "activity": "Learn API routes, request validation, and response models.",
        "practice": "Create and test a small API with validated input and structured output.",
    },
    "docker": {
        "activity": "Study images, containers, Dockerfiles, and environment configuration.",
        "practice": "Containerize a small application and document how to run it.",
    },
    "communication": {
        "activity": "Practise explaining technical decisions and findings clearly.",
        "practice": "Prepare a short presentation explaining a project, its results, and limitations.",
    },
    "problem solving": {
        "activity": "Practise breaking problems into smaller steps and testing solutions.",
        "practice": "Solve a small technical problem and document the reasoning and tests.",
    },
}


# ============================================================
# HELPERS
# ============================================================

def normalize_name(value: str) -> str:
    """Normalize a name for consistent comparison."""

    return " ".join(str(value).strip().casefold().split())


def get_learning_activity(skill: str) -> dict[str, str]:
    """Return a specific activity or a safe generic fallback."""

    key = normalize_name(skill)

    if key in SKILL_ACTIVITIES:
        return SKILL_ACTIVITIES[key]

    return {
        "activity": (
            f"Learn the fundamentals and common applications of {skill}."
        ),
        "practice": (
            f"Complete a small hands-on exercise using {skill} "
            "and document the result."
        ),
    }


def validate_match_result(match_result: dict[str, Any]) -> None:
    """Ensure the input has the fields returned by job_matcher.py."""

    if not isinstance(match_result, dict):
        raise TypeError("match_result must be a dictionary.")

    if not match_result.get("job_role"):
        raise ValueError("The job-match result must contain job_role.")

    required_fields = (
        "missing_required_skills",
        "missing_preferred_skills",
    )

    for field in required_fields:
        if field not in match_result:
            raise ValueError(
                f"Missing '{field}' in the job-match result."
            )

        if not isinstance(match_result[field], list):
            raise TypeError(f"'{field}' must be a list.")


# ============================================================
# ROADMAP GENERATION
# ============================================================

def generate_roadmap(
    match_result: dict[str, Any],
    weeks: int = 6,
) -> dict[str, Any]:
    """
    Generate a learning roadmap from one job-match result.

    Required skill gaps receive higher priority than preferred
    skill gaps.

    The duration is a suggested planning structure, not a guarantee
    of how long a person will need to master a skill.
    """

    validate_match_result(match_result)

    if isinstance(weeks, bool) or not isinstance(weeks, int) or weeks < 1:
        raise ValueError("weeks must be a positive integer.")

    role = match_result["job_role"]
    missing_required = match_result["missing_required_skills"]
    missing_preferred = match_result["missing_preferred_skills"]

    # Deduplicate while retaining priority order.
    tasks_by_skill = {}

    for skill in missing_required:
        skill = str(skill).strip()

        if skill:
            tasks_by_skill[normalize_name(skill)] = {
                "skill": skill,
                "priority": "High",
                "reason": "Listed as a required skill for this role.",
            }

    for skill in missing_preferred:
        skill = str(skill).strip()
        key = normalize_name(skill)

        if skill and key not in tasks_by_skill:
            tasks_by_skill[key] = {
                "skill": skill,
                "priority": "Medium",
                "reason": "Listed as a preferred skill for this role.",
            }

    tasks = list(tasks_by_skill.values())

    # If no gaps were found, provide a maintenance plan instead.
    if not tasks:
        return {
            "target_role": role,
            "match_score": match_result.get("match_score"),
            "duration_weeks": weeks,
            "status": "No listed skill gaps found",
            "summary": (
                "The supplied job match identifies no missing required "
                "or preferred skills. Validate proficiency through "
                "projects and practical assessments."
            ),
            "weekly_plan": [
                {
                    "week": week,
                    "focus": "Practical assessment and portfolio improvement",
                    "tasks": [
                        {
                            "skill": "Existing skills",
                            "priority": "Maintenance",
                            "reason": "Continue demonstrating relevant skills.",
                            "learning_activity": (
                                "Review concepts and practise independently."
                            ),
                            "practice_task": (
                                "Improve a project and record evidence of "
                                "what you can accomplish."
                            ),
                        }
                    ],
                }
                for week in range(1, weeks + 1)
            ],
            "missing_required_skills": [],
            "missing_preferred_skills": [],
            "next_steps": [
                "Validate proficiency with practical exercises.",
                "Review the role requirements periodically.",
                "Keep relevant projects and documentation updated.",
            ],
        }

    weekly_plan = []

    # Allocate all identified skill gaps across the requested weeks.
    # Required skills are scheduled before preferred skills.
    total_tasks = len(tasks)

    for week in range(1, weeks + 1):
        start = (week - 1) * total_tasks // weeks
        end = week * total_tasks // weeks

        week_tasks = []

        for task in tasks[start:end]:
            activity = get_learning_activity(task["skill"])

            week_tasks.append({
                "skill": task["skill"],
                "priority": task["priority"],
                "reason": task["reason"],
                "learning_activity": activity["activity"],
                "practice_task": activity["practice"],
            })

        weekly_plan.append({
            "week": week,
            "focus": (
                "Required skill development"
                if any(t["priority"] == "High" for t in week_tasks)
                else "Preferred skills and practical reinforcement"
            ),
            "tasks": week_tasks,
        })

    high_priority_count = sum(
        1 for task in tasks if task["priority"] == "High"
    )

    medium_priority_count = sum(
        1 for task in tasks if task["priority"] == "Medium"
    )

    score = match_result.get("match_score")

    return {
        "target_role": role,
        "match_score": score,
        "duration_weeks": weeks,
        "status": "Learning roadmap generated",
        "summary": (
            f"The roadmap covers {high_priority_count} missing required "
            f"skill(s) and {medium_priority_count} missing preferred "
            f"skill(s) for the {role} role. Complete the practical tasks "
            "and reassess your skill match."
        ),
        "weekly_plan": weekly_plan,
        "missing_required_skills": missing_required.copy(),
        "missing_preferred_skills": missing_preferred.copy(),
        "next_steps": [
            "Complete the practical tasks and save evidence of your work.",
            "Build or improve a project demonstrating the newly practised skills.",
            "Re-run the skill extractor and job matcher after updating your resume.",
        ],
    }


# ============================================================
# DISPLAY ROADMAP AS PLAIN TEXT
# ============================================================

def format_roadmap(roadmap: dict[str, Any]) -> str:
    """Convert the roadmap into readable text for testing or export."""

    lines = [
        f"LEARNING ROADMAP: {roadmap['target_role']}",
        "=" * 60,
        f"Status: {roadmap['status']}",
        f"Suggested duration: {roadmap['duration_weeks']} weeks",
        "",
        roadmap["summary"],
        "",
    ]

    for week in roadmap["weekly_plan"]:
        lines.append(f"WEEK {week['week']} — {week['focus']}")

        if not week["tasks"]:
            lines.append("  No new skill tasks scheduled.")
            lines.append("")
            continue

        for task in week["tasks"]:
            lines.extend([
                f"  Skill: {task['skill']}",
                f"  Priority: {task['priority']}",
                f"  Why: {task['reason']}",
                f"  Learn: {task['learning_activity']}",
                f"  Practise: {task['practice_task']}",
                "",
            ])

    lines.append("NEXT STEPS")

    for step in roadmap["next_steps"]:
        lines.append(f"- {step}")

    return "\n".join(lines)
