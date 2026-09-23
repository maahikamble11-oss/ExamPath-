"""
Generates the sample INPUT data for the ExamPath project:
    data/students.csv
    data/questions.csv
    data/performance.csv

"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------
# 1. STUDENTS
# ---------------------------------------------------------------------
students = pd.DataFrame({
    "student_id": [f"S{str(i).zfill(3)}" for i in range(1, 11)],
    "name": ["Aarav Sharma", "Diya Patel", "Vihaan Gupta", "Ananya Singh", "Kabir Rao",
             "Myra Verma", "Reyansh Mehta", "Isha Nair", "Arjun Iyer", "Saanvi Joshi"],
    "class": ["12-A", "12-A", "12-B", "12-B", "12-A",
              "12-C", "12-B", "12-C", "12-A", "12-C"],
    "email": [f"student{i}@exampath.edu" for i in range(1, 11)],
})
students.to_csv("data/students.csv", index=False)

# ---------------------------------------------------------------------
# 2. QUESTIONS (question bank, 5 per concept)
# ---------------------------------------------------------------------
concepts = {
    "Physics": ["Kinematics", "Laws of Motion", "Thermodynamics", "Optics", "Electrostatics"],
    "Chemistry": ["Atomic Structure", "Chemical Bonding", "Organic Reactions", "Equilibrium", "Thermochemistry"],
    "Mathematics": ["Algebra", "Calculus", "Trigonometry", "Probability", "Coordinate Geometry"],
}

rows = []
qid = 1
difficulties = ["Easy", "Medium", "Hard"]
for subject, clist in concepts.items():
    for concept in clist:
        for n in range(1, 6):
            diff = difficulties[(n - 1) % 3]
            correct_letter = rng.choice(["A", "B", "C", "D"])
            rows.append({
                "question_id": f"Q{str(qid).zfill(4)}",
                "subject": subject,
                "concept": concept,
                "difficulty": diff,
                "question_text": f"[{concept}] Sample question #{n} on {concept} ({diff} level).",
                "option_a": f"{concept} option A-{n}",
                "option_b": f"{concept} option B-{n}",
                "option_c": f"{concept} option C-{n}",
                "option_d": f"{concept} option D-{n}",
                "correct_option": correct_letter,
            })
            qid += 1
questions = pd.DataFrame(rows)
questions.to_csv("data/questions.csv", index=False)

# ---------------------------------------------------------------------
# 3. PERFORMANCE LOG (simulated attempt history -> ML training data)
# ---------------------------------------------------------------------
all_concepts = [c for clist in concepts.values() for c in clist]

# give every student a hidden "true skill" per concept (0-1) so data is realistic
skill = {
    sid: {c: rng.beta(2, 2) for c in all_concepts}
    for sid in students["student_id"]
}
# deliberately make a couple of concepts weak for most students (for a good demo)
for sid in students["student_id"]:
    for weak_c in rng.choice(all_concepts, size=3, replace=False):
        skill[sid][weak_c] *= 0.35

diff_penalty = {"Easy": 0.10, "Medium": 0.0, "Hard": -0.15}

perf_rows = []
today = datetime.now().date()
for sid in students["student_id"]:
    n_attempts = rng.integers(25, 45)
    picked_q = questions.sample(
        n=int(n_attempts),
        replace=True,
        random_state=int(rng.integers(0, 10_000)),
    )
    days_back = sorted(rng.integers(1, 45, size=n_attempts), reverse=True)
    for (_, q), d in zip(picked_q.iterrows(), days_back):
        p_correct = np.clip(skill[sid][q["concept"]] + diff_penalty[q["difficulty"]], 0.02, 0.98)
        correct = int(rng.random() < p_correct)
        time_taken = round(rng.normal(60 - 20 * p_correct, 10), 1)
        time_taken = max(8.0, time_taken)
        perf_rows.append({
            "student_id": sid,
            "question_id": q["question_id"],
            "concept": q["concept"],
            "difficulty": q["difficulty"],
            "correct": correct,
            "time_taken": time_taken,  # seconds
            "date": (today - timedelta(days=int(d))).isoformat(),
        })

performance = pd.DataFrame(perf_rows).sort_values(["student_id", "date"])
performance.to_csv("data/performance.csv", index=False)

print("Generated:")
print(" data/students.csv   ->", students.shape)
print(" data/questions.csv  ->", questions.shape)
print(" data/performance.csv->", performance.shape)
