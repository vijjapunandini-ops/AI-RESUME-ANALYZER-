from skill_extractor import extract_skills


sample_resume = """
Professional Summary
Software developer with experience in Python and SQL.

Technical Skills
Python, Pandas, NumPy, Machine Learning, NLP, Git.

Projects
Built a RAG application using LangChain and FAISS.
Used Streamlit to develop the user interface.
"""

results = extract_skills(sample_resume)

print("\nEXTRACTED SKILLS")
print("=" * 40)

for item in results:
    print(
        f"{item['skill']} | "
        f"{item['category']} | "
        f"Mentions: {item['mentions']}"
    )

print("=" * 40)
print("Total unique skills:", len(results))
