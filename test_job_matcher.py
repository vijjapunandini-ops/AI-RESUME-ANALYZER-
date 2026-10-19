from skill_extractor import extract_skills
from job_matcher import match_jobs


sample_resume = """
Technical Skills
Python, SQL, Pandas, NumPy, Machine Learning,
Data Cleaning, Exploratory Data Analysis,
Statistics, Git, Streamlit.
"""

# First extract canonical skills using your existing extractor.
extracted_skills = extract_skills(sample_resume)

# Then compare those skills with the jobs dataset.
matches = match_jobs(
    extracted_skills,
    experience_years=0,
    top_n=5,
)

print("\nTOP JOB MATCHES")
print("=" * 60)

for index, job in enumerate(matches, start=1):
    print(f"\n{index}. {job['job_role']}")
    print(f"   Match score: {job['match_score']}%")
    print(
        "   Required skills matched: "
        f"{len(job['matched_required_skills'])}/"
        f"{job['total_required_skills']}"
    )
    print(
        "   Preferred skills matched: "
        f"{len(job['matched_preferred_skills'])}/"
        f"{job['total_preferred_skills']}"
    )
    print(
        "   Missing required skills:",
        ", ".join(job["missing_required_skills"]) or "None",
    )

print("\nTest completed.")
