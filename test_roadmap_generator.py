"""
End-to-end backend test:
skill extraction -> job matching -> roadmap generation.
"""

from skill_extractor import extract_skills
from job_matcher import match_jobs
from roadmap_generator import generate_roadmap, format_roadmap


SAMPLE_RESUME = """
Technical Skills
Python, SQL, Pandas, NumPy, Git, Streamlit.

Projects
Built a small data analysis dashboard using Python and Pandas.
"""


def main():
    print("STEP 1: Extract resume skills")

    extracted_skills = extract_skills(SAMPLE_RESUME)

    for item in extracted_skills:
        print(f"  - {item['skill']}")

    if not extracted_skills:
        raise AssertionError(
            "No skills were extracted. Check Skill_dictionary.csv."
        )

    print("\nSTEP 2: Match skills against job roles")

    matches = match_jobs(
        extracted_skills,
        experience_years=0,
        top_n=5,
    )

    if not matches:
        raise AssertionError(
            "No job matches were returned. Check Job_roles.csv."
        )

    for index, match in enumerate(matches, start=1):
        print(
            f"{index}. {match['job_role']}: "
            f"{match['match_score']}%"
        )

    print("\nSTEP 3: Generate a roadmap for the best-ranked role")

    roadmap = generate_roadmap(
        match_result=matches[0],
        weeks=6,
    )

    print(format_roadmap(roadmap))

    # Basic automated checks.
    assert roadmap["target_role"] == matches[0]["job_role"]
    assert len(roadmap["weekly_plan"]) == 6
    assert roadmap["status"]

    print("\nALL ROADMAP TESTS PASSED.")


if __name__ == "__main__":
    main()
