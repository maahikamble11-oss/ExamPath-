# 🧭 ExamPath — AI Study Companion (ML + Streamlit project)

ExamPath analyzes a student's quiz history, predicts **weak concepts** with a
machine-learning model, generates a **targeted practice quiz**, **scores** the
attempt, and schedules **spaced-repetition revisions** with in-browser alarms.

All data is stored as plain **CSV files** (as if exported from Excel) —
**no SQL / database** is used anywhere in this project.

## Project structure
```
ExamPath/
├── app.py                     # Streamlit app (all pages/UI)
├── generate_sample_data.py    # Creates the sample input CSVs
├── requirements.txt
├── data/
│   ├── students.csv           # input: student master data
│   ├── questions.csv          # input: question bank
│   └── performance.csv        # input: attempt history (ML training data)
└── utils/
    ├── ml_engine.py           # RandomForestClassifier + KMeans
    └── spaced_repetition.py   # SM-2 style revision scheduler
```

## How the ML works
1. **Weak concept detection** — a `RandomForestClassifier` is trained per
   session on each student's attempt history. Features: question difficulty,
   time taken, and rolling (pre-attempt) accuracy on that concept. The
   model's `predict_proba` for "answered correctly" becomes the **mastery %**
   for that concept. Anything below the sidebar threshold (default 50%) is
   flagged weak.
2. **Learner segmentation** — `KMeans` (k=3) clusters students by overall
   accuracy and average response time into *Needs Support / Developing /
   Advanced* tiers, shown on the Dashboard.
3. **Spaced repetition** — a simplified SM-2 algorithm decides the next
   revision date per concept: weak concepts return the next day, strong ones
   get pushed further out using a growing "ease factor".
4. **Alarms** — the Revision Schedule page uses the browser's real
   `Notification` API (via an injected `<script>`) to pop native alarms for
   concepts due today, alongside an in-app alert banner and `st.toast`.

## Run it
```bash
pip install -r requirements.txt
python generate_sample_data.py   # only needed once, regenerates sample CSVs
streamlit run app.py
```

## Bring your own data
Go to **📤 Upload Data** in the sidebar and upload your own
`students.csv` / `questions.csv` / `performance.csv` (Replace or Append mode).
Templates with the exact required columns are downloadable on that page.

## CSV schemas
**students.csv**: `student_id, name, class, email`

**questions.csv**: `question_id, subject, concept, difficulty, question_text,
option_a, option_b, option_c, option_d, correct_option`

**performance.csv**: `student_id, question_id, concept, difficulty, correct,
time_taken, date`
