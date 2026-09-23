# 🧭 ExamPath — AI Study Companion (ML + Streamlit project)

Problem Statement & the Need: Most students revise reactively — they re-read the same chapters, take generic mock tests, and don't know which specific concepts are actually weak until they lose marks in the real exam. Teachers and coaching institutes also struggle to track individual weak areas across large batches without manual effort. The result: wasted revision time on topics already mastered, and last-minute panic on topics that were never actually reinforced. There's a need for a lightweight, data-driven tool that pinpoints weak concepts early and builds a personalized, repeatable revision loop around them — without requiring heavy infrastructure like a school ERP or database.

Proposed Product & How It Works
ExamPath is a Streamlit-based AI study companion that ingests a student's quiz-attempt history (from simple CSV files — no database needed), and:

Trains a RandomForestClassifier on attempt patterns (difficulty, time taken, rolling accuracy) to predict a mastery score per concept.
Flags concepts below a mastery threshold as "weak."
Auto-generates a targeted practice quiz from the question bank on those weak concepts.
Scores the quiz instantly and logs the new attempt back into the history.
Feeds the result into a spaced-repetition scheduler (SM-2 style) that decides when each concept should be revisited, and fires browser alarms when a revision is due.

Target Users:
School/coaching students (classes 9–12, JEE/NEET aspirants) who want a personal weak-spot tracker.
Teachers/tutors managing small-to-medium batches who want a quick way to see which students are weak in which topics without building a full LMS.
Coaching institutes / edtech startups who want a lightweight add-on module rather than a full analytics platform.

Key Features & Value Proposition:
Weak-concept detection powered by real ML, not just averages
Auto-built, self-scoring practice quizzes targeted at weak areas
Spaced-repetition revision calendar with real alarms
Learner segmentation (KMeans) to group students by ability
Fully CSV-driven — works with data exported from Excel/Google Sheets, no database setup
Value prop: less time spent guessing what to study, more time spent studying the right things at the right intervals.

What Makes It Innovative / Different:
Combines three techniques that are usually separate products — mastery prediction, adaptive quizzing, and spaced repetition — into one lightweight tool.
Zero infrastructure overhead: runs entirely on CSV files, so any school or tutor can adopt it without IT support or a database.
Alarms use the actual browser Notification API, giving a real nudge rather than a passive dashboard nobody checks.
Model retrains per student in real time from live data, rather than shipping a fixed, generic "weak topics" list.

Basic Business / Revenue Model:
Freemium SaaS: free tier for individual students (limited question bank/history); paid tier for advanced analytics, larger question banks, and unlimited history.
B2B licensing to coaching institutes/schools: per-student or per-batch subscription, with a teacher dashboard add-on.
White-label/API licensing: edtech platforms could plug the ML engine (weak-concept detection + spaced repetition) into their own products.
Content marketplace: question-bank packs for specific exams (JEE, NEET, boards) sold as add-ons.

Go-to-Market Strategy:
Launch a free version targeting individual students via social media/exam-prep communities (Instagram, Telegram groups, Reddit exam subs) to build initial usage and word-of-mouth.
Pilot with 2–3 local coaching institutes (like ones in your city) offering the teacher dashboard free for a term in exchange for feedback and testimonials.
Use pilot data/case studies to approach mid-size coaching chains for paid batch licenses.
Expand into a mobile-friendly/app version and partner with edtech content providers for question banks.

Prototype:
The working prototype is the Streamlit app already built in this conversation — it demonstrates the full core loop end to end: CSV data upload → ML-based weak-concept analysis → auto-generated practice quiz → live scoring → spaced-repetition schedule → browser alarm. You have the zipped project (ExamPath.zip) and sample datasets from earlier in this chat to run and showcase it.

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
