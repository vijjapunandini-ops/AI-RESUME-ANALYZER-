"""
End-to-end integration test for AI Resume Analyzer.

Pipeline:
1. Extract document text.
2. Parse the resume.
3. Extract skills.
4. Match jobs.
5. Generate a learning roadmap.

Supported test documents: PDF and DOCX.
"""

from pathlib import Path

from resume_parser import parse_resume_file
from skill_extractor import extract_skills_from_resume
from job_matcher import match_jobs
from roadmap_generator import generate_roadmap, format_roadmap


BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "Sample_resumes"

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def run_resume_pipeline(resume_path: Path) -> dict:
    """Run the complete backend pipeline for one resume."""

    print("\n" + "=" * 70)
    print(f"TESTING RESUME: {resume_path.name}")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. DOCUMENT EXTRACTION + RESUME PARSING
    # --------------------------------------------------------

    parsed_resume = parse_resume_file(
        filename=resume_path.name,
        file_bytes=resume_path.read_bytes(),
    )

    assert parsed_resume.get("raw_text", "").strip(), (
        "Document extraction returned no readable text."
    )

    print("\n[PASS] Document extraction and resume parsing")

    candidate_name = parsed_resume.get("candidate_name")

    if candidate_name:
        print(f"Candidate name detected: {candidate_name}")
    else:
        print("Candidate name: Not detected")

    print(f"Extracted words: {parsed_resume['word_count']}")

    # --------------------------------------------------------
    # 2. SKILL EXTRACTION
    # --------------------------------------------------------

    extracted = extract_skills_from_resume(parsed_resume)
    skills = extracted.get("skills", [])

    if not skills:
        raise AssertionError(
            "No skills were extracted. Check the resume content "
            "and data/Skill_dictionary.csv."
        )

    print("\n[PASS] Skill extraction")
    print(f"Unique skills identified: {len(skills)}")

    for item in skills:
        print(f"  - {item['skill']} ({item['category']})")

    # --------------------------------------------------------
    # 3. JOB MATCHING
    # --------------------------------------------------------

    # We have not yet implemented reliable experience extraction.
    # Pass None rather than assuming the candidate has zero years.
    matches = match_jobs(
        resume_skills=extracted,
        experience_years=None,
        top_n=5,
    )

    if not matches:
        raise AssertionError(
            "No job matches returned. Check data/Job_roles.csv."
        )

    print("\n[PASS] Job matching")
    print("\nTop job matches:")

    for rank, match in enumerate(matches, start=1):
        print(
            f"{rank}. {match['job_role']} "
            f"— {match['match_score']:.2f}% skill match"
        )

        print(
            "   Missing required skills: "
            + (
                ", ".join(match["missing_required_skills"])
                or "None"
            )
        )

    # --------------------------------------------------------
    # 4. ROADMAP GENERATION
    # --------------------------------------------------------

    best_match = matches[0]

    roadmap = generate_roadmap(
        match_result=best_match,
        weeks=6,
    )

    assert roadmap.get("target_role") == best_match["job_role"]
    assert len(roadmap.get("weekly_plan", [])) == 6

    print("\n[PASS] Learning roadmap generation")
    print(format_roadmap(roadmap))

    return {
        "filename": resume_path.name,
        "skills_found": len(skills),
        "jobs_ranked": len(matches),
        "best_match": best_match["job_role"],
        "match_score": best_match["match_score"],
        "roadmap_status": roadmap["status"],
    }


def main():
    """Find sample resumes and run the complete pipeline."""

    if not SAMPLE_DIR.is_dir():
        raise FileNotFoundError(
            f"Sample resume folder not found: {SAMPLE_DIR}\n"
            "Create a folder named Sample_resumes and place a "
            "sample PDF or DOCX resume inside it."
        )

    resume_files = sorted(
        path
        for path in SAMPLE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not resume_files:
        raise FileNotFoundError(
            f"No PDF or DOCX resumes were found in {SAMPLE_DIR}."
        )

    successful = []
    failures = []

    for resume_path in resume_files:
        try:
            result = run_resume_pipeline(resume_path)
            successful.append(result)

        except Exception as exc:
            failures.append((resume_path.name, str(exc)))

            print(f"\n[FAIL] {resume_path.name}")
            print(f"Reason: {exc}")

    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 70)

    for result in successful:
        print(
            f"[PASS] {result['filename']} | "
            f"Skills: {result['skills_found']} | "
            f"Best role: {result['best_match']} | "
            f"Match: {result['match_score']:.2f}%"
        )

    for filename, message in failures:
        print(f"[FAIL] {filename}: {message}")

    print(f"\nPassed: {len(successful)}")
    print(f"Failed: {len(failures)}")

    if failures:
        raise SystemExit(1)

    print("\nALL RESUME PIPELINE INTEGRATION TESTS PASSED.")


if __name__ == "__main__":
    main()
