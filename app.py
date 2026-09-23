"""
ExamPath — an ML-powered exam-prep companion.

Features
--------
- Loads student / question / performance data from plain CSV files
  (no SQL / database anywhere in the project).
- Lets a teacher/student manually upload their own CSV data.
- Uses a RandomForestClassifier to predict per-concept "mastery" and
  surface weak concepts, plus KMeans to segment students into learner tiers.
- Auto-builds a practice quiz targeted at a student's weakest concepts,
  scores it, and logs the attempt.
- Runs a spaced-repetition (SM-2 style) revision scheduler and fires
  in-browser alarms for concepts due today.

Run with:  streamlit run app.py
"""
import json
from datetime import datetime, date

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px

from utils import ml_engine, spaced_repetition as sr

# --------------------------------------------------------------------------
# PAGE CONFIG + STYLE
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="ExamPath | AI Study Companion",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.main { background-color: #0f1117; }
h1, h2, h3 { font-family: 'Trebuchet MS', sans-serif; }
.exampath-header {
    background: linear-gradient(90deg, #4f46e5, #06b6d4);
    padding: 1.4rem 1.8rem;
    border-radius: 14px;
    color: white;
    margin-bottom: 1.2rem;
}
.exampath-header h1 { margin: 0; font-size: 2rem; }
.exampath-header p { margin: 0.2rem 0 0 0; opacity: 0.9; }
div[data-testid="stMetric"] {
    background: rgba(79, 70, 229, 0.08);
    border: 1px solid rgba(79, 70, 229, 0.25);
    padding: 0.8rem 1rem;
    border-radius: 12px;
}
.weak-badge {
    display:inline-block; padding:3px 10px; border-radius:999px;
    background:#fee2e2; color:#b91c1c; font-size:0.8rem; font-weight:600; margin:2px;
}
.mid-badge {
    display:inline-block; padding:3px 10px; border-radius:999px;
    background:#fef3c7; color:#92400e; font-size:0.8rem; font-weight:600; margin:2px;
}
.strong-badge {
    display:inline-block; padding:3px 10px; border-radius:999px;
    background:#dcfce7; color:#166534; font-size:0.8rem; font-weight:600; margin:2px;
}
.alarm-box {
    background:#fff1f2; border-left:5px solid #e11d48; padding:0.9rem 1.1rem;
    border-radius:8px; margin-bottom:0.8rem; color:#881337; font-weight:500;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DATA_DIR = "data"


# --------------------------------------------------------------------------
# DATA LOADING (CSV only — no SQL)
# --------------------------------------------------------------------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


def init_state():
    if "students" not in st.session_state:
        st.session_state.students = load_csv(f"{DATA_DIR}/students.csv")
    if "questions" not in st.session_state:
        st.session_state.questions = load_csv(f"{DATA_DIR}/questions.csv")
    if "performance" not in st.session_state:
        st.session_state.performance = load_csv(f"{DATA_DIR}/performance.csv")
    if "schedule" not in st.session_state:
        try:
            st.session_state.schedule = load_csv(f"{DATA_DIR}/revision_schedule.csv")
        except FileNotFoundError:
            st.session_state.schedule = sr.empty_schedule()


init_state()


def fire_browser_alarm(messages):
    """Fires real browser (Notification API) alarms — the 'simultaneous alert'."""
    if not messages:
        return
    msgs = json.dumps(messages)
    components.html(f"""
    <script>
    const msgs = {msgs};
    if (window.Notification) {{
        if (Notification.permission !== "granted" && Notification.permission !== "denied") {{
            Notification.requestPermission().then(() => fire());
        }} else {{
            fire();
        }}
        function fire() {{
            if (Notification.permission === "granted") {{
                msgs.forEach((m, i) => {{
                    setTimeout(() => {{
                        new Notification("🔔 ExamPath Revision Alarm", {{ body: m }});
                    }}, i * 400);
                }});
            }}
        }}
    }}
    </script>
    """, height=0)


# --------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------
st.sidebar.markdown("## 🧭 ExamPath")
st.sidebar.caption("AI-powered exam preparation companion")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Dashboard", "📤 Upload Data", "🧠 Weak Concept Analysis",
     "📝 Practice & Score", "📅 Revision Schedule & Alarms", "👤 Student Profile"],
)

st.sidebar.divider()
mastery_threshold = st.sidebar.slider("Weak-concept threshold (mastery %)", 20, 80, 50, 5)
st.sidebar.caption("Concepts predicted below this mastery % are flagged as weak.")

students_df = st.session_state.students
questions_df = st.session_state.questions
performance_df = st.session_state.performance


# --------------------------------------------------------------------------
# PAGE: DASHBOARD
# --------------------------------------------------------------------------
if page == "🏠 Dashboard":
    st.markdown(
        '<div class="exampath-header"><h1>🧭 ExamPath</h1>'
        '<p>Data-driven revision — find your weak spots before the exam finds them.</p></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", len(students_df))
    c2.metric("Questions in bank", len(questions_df))
    c3.metric("Logged attempts", len(performance_df))
    overall_acc = performance_df["correct"].mean() * 100 if len(performance_df) else 0
    c4.metric("Overall accuracy", f"{overall_acc:.1f}%")

    st.divider()
    colA, colB = st.columns([1.3, 1])

    with colA:
        st.subheader("📊 Concept-wise average accuracy (all students)")
        if len(performance_df):
            concept_acc = (
                performance_df.groupby("concept")["correct"].mean().mul(100).reset_index()
                .sort_values("correct")
            )
            fig = px.bar(
                concept_acc, x="correct", y="concept", orientation="h",
                labels={"correct": "Accuracy (%)", "concept": ""},
                color="correct", color_continuous_scale="RdYlGn", range_color=[0, 100],
            )
            fig.update_layout(height=520, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance data yet — upload some in 📤 Upload Data.")

    with colB:
        st.subheader("🎯 Learner segments (KMeans)")
        seg = ml_engine.cluster_students(performance_df)
        if "cluster_label" in seg.columns and len(seg):
            seg_named = seg.merge(students_df[["student_id", "name"]], on="student_id", how="left")
            counts = seg_named["cluster_label"].value_counts().reset_index()
            counts.columns = ["Segment", "Students"]
            fig2 = px.pie(counts, names="Segment", values="Students", hole=0.5,
                          color="Segment",
                          color_discrete_map={"Needs Support": "#ef4444", "Developing": "#f59e0b",
                                              "Advanced": "#22c55e"})
            fig2.update_layout(height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(
                seg_named[["name", "avg_accuracy", "avg_time", "cluster_label"]]
                .rename(columns={"name": "Student", "avg_accuracy": "Avg Accuracy",
                                  "avg_time": "Avg Time (s)", "cluster_label": "Segment"})
                .assign(**{"Avg Accuracy": lambda d: (d["Avg Accuracy"] * 100).round(1)})
                .style.format({"Avg Time (s)": "{:.1f}"}),
                use_container_width=True, height=220,
            )
        else:
            st.info("Need at least 3 students with attempts for clustering.")


# --------------------------------------------------------------------------
# PAGE: UPLOAD DATA
# --------------------------------------------------------------------------
elif page == "📤 Upload Data":
    st.header("📤 Upload manual data")
    st.write(
        "Bring your own data as CSV (exported from Excel or Google Sheets). "
        "No database/SQL is required — everything here runs on CSV files."
    )

    tab1, tab2, tab3 = st.tabs(["👩‍🎓 Students", "❓ Questions", "📈 Performance / Attempts"])

    with tab1:
        st.caption("Required columns: student_id, name, class, email")
        st.download_button("Download template", students_df.head(0).to_csv(index=False),
                            "students_template.csv")
        up = st.file_uploader("Upload students.csv", type="csv", key="up_students")
        mode = st.radio("Mode", ["Replace", "Append"], horizontal=True, key="mode_students")
        if up is not None:
            new_df = pd.read_csv(up)
            st.session_state.students = new_df if mode == "Replace" else pd.concat(
                [students_df, new_df]).drop_duplicates("student_id")
            st.success(f"Loaded {len(new_df)} rows.")
            st.dataframe(st.session_state.students, use_container_width=True)

    with tab2:
        st.caption("Required columns: question_id, subject, concept, difficulty, question_text, "
                   "option_a, option_b, option_c, option_d, correct_option")
        st.download_button("Download template", questions_df.head(0).to_csv(index=False),
                            "questions_template.csv")
        up2 = st.file_uploader("Upload questions.csv", type="csv", key="up_questions")
        mode2 = st.radio("Mode", ["Replace", "Append"], horizontal=True, key="mode_questions")
        if up2 is not None:
            new_df = pd.read_csv(up2)
            st.session_state.questions = new_df if mode2 == "Replace" else pd.concat(
                [questions_df, new_df]).drop_duplicates("question_id")
            st.success(f"Loaded {len(new_df)} rows.")
            st.dataframe(st.session_state.questions, use_container_width=True)

    with tab3:
        st.caption("Required columns: student_id, question_id, concept, difficulty, correct, "
                   "time_taken, date")
        st.download_button("Download template", performance_df.head(0).to_csv(index=False),
                            "performance_template.csv")
        up3 = st.file_uploader("Upload performance.csv", type="csv", key="up_perf")
        mode3 = st.radio("Mode", ["Replace", "Append"], horizontal=True, key="mode_perf")
        if up3 is not None:
            new_df = pd.read_csv(up3)
            st.session_state.performance = new_df if mode3 == "Replace" else pd.concat(
                [performance_df, new_df], ignore_index=True)
            st.success(f"Loaded {len(new_df)} rows.")
            st.dataframe(st.session_state.performance.tail(20), use_container_width=True)

    st.divider()
    st.subheader("Current data snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Students", len(st.session_state.students))
    c2.metric("Questions", len(st.session_state.questions))
    c3.metric("Attempts", len(st.session_state.performance))


# --------------------------------------------------------------------------
# PAGE: WEAK CONCEPT ANALYSIS (ML)
# --------------------------------------------------------------------------
elif page == "🧠 Weak Concept Analysis":
    st.header("🧠 ML-based weak concept analysis")
    st.caption("A RandomForestClassifier is trained on each student's attempt history "
               "(difficulty, time taken, rolling accuracy) to predict mastery per concept.")

    student_name = st.selectbox("Select student", students_df["name"])
    student_id = students_df.loc[students_df["name"] == student_name, "student_id"].iloc[0]

    model, le, feat_df = ml_engine.train_mastery_model(performance_df)
    if model is None:
        st.warning("Not enough attempt data to train the model yet (need 15+ logged attempts).")
    else:
        mastery_df = ml_engine.predict_mastery(model, le, feat_df, student_id)
        weak = mastery_df[mastery_df["mastery"] < mastery_threshold]
        mid = mastery_df[(mastery_df["mastery"] >= mastery_threshold) & (mastery_df["mastery"] < 75)]
        strong = mastery_df[mastery_df["mastery"] >= 75]

        st.markdown("**Weak:** " + ("".join(f'<span class="weak-badge">{c}</span>' for c in weak["concept"]) or "none 🎉"),
                    unsafe_allow_html=True)
        st.markdown("**Developing:** " + ("".join(f'<span class="mid-badge">{c}</span>' for c in mid["concept"]) or "—"),
                    unsafe_allow_html=True)
        st.markdown("**Strong:** " + ("".join(f'<span class="strong-badge">{c}</span>' for c in strong["concept"]) or "—"),
                    unsafe_allow_html=True)

        st.divider()
        fig = px.bar(mastery_df, x="mastery", y="concept", orientation="h",
                     color="mastery", color_continuous_scale="RdYlGn", range_color=[0, 100],
                     labels={"mastery": "Predicted mastery (%)", "concept": ""})
        fig.add_vline(x=mastery_threshold, line_dash="dash", line_color="red")
        fig.update_layout(height=480, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            mastery_df.rename(columns={"mastery": "Mastery %", "attempts": "Attempts",
                                        "accuracy": "Raw Accuracy %", "avg_time": "Avg Time (s)",
                                        "concept": "Concept"}),
            use_container_width=True,
        )
        st.session_state["last_weak_concepts"] = weak["concept"].tolist()
        st.session_state["last_student_id"] = student_id


# --------------------------------------------------------------------------
# PAGE: PRACTICE & SCORE
# --------------------------------------------------------------------------
elif page == "📝 Practice & Score":
    st.header("📝 Practice quiz built from your weak concepts")

    student_name = st.selectbox("Student", students_df["name"], key="quiz_student")
    student_id = students_df.loc[students_df["name"] == student_name, "student_id"].iloc[0]

    model, le, feat_df = ml_engine.train_mastery_model(performance_df)
    if model is not None:
        mastery_df = ml_engine.predict_mastery(model, le, feat_df, student_id)
        weak_concepts = ml_engine.weak_concepts_for_student(mastery_df, mastery_threshold)
    else:
        weak_concepts = []

    available_concepts = sorted(questions_df["concept"].unique())
    default_pick = weak_concepts if weak_concepts else available_concepts[:3]
    chosen = st.multiselect("Concepts to practice (auto-suggested from weak areas)",
                             available_concepts, default=default_pick)
    n_q = st.slider("Number of questions", 3, 15, 5)

    pool = questions_df[questions_df["concept"].isin(chosen)]
    if pool.empty:
        st.info("Pick at least one concept.")
    else:
        if "quiz_set" not in st.session_state or st.button("🔄 Generate new quiz"):
            st.session_state.quiz_set = pool.sample(n=min(n_q, len(pool)), random_state=None).reset_index(drop=True)

        quiz = st.session_state.quiz_set
        with st.form("quiz_form"):
            answers = {}
            for i, row in quiz.iterrows():
                st.markdown(f"**Q{i+1}. ({row['concept']} — {row['difficulty']}):** {row['question_text']}")
                answers[row["question_id"]] = st.radio(
                    "Choose:",
                    options=["A", "B", "C", "D"],
                    format_func=lambda x, r=row: f"{x}. {r[f'option_{x.lower()}']}",
                    key=f"ans_{i}", horizontal=True, index=None,
                )
                st.markdown("---")
            submitted = st.form_submit_button("✅ Submit quiz")

        if submitted:
            results = []
            score = 0
            new_attempts = []
            today_str = date.today().isoformat()
            for i, row in quiz.iterrows():
                chosen_opt = answers[row["question_id"]]
                is_correct = int(chosen_opt == row["correct_option"])
                score += is_correct
                results.append({"Question": row["question_text"], "Your answer": chosen_opt,
                                 "Correct answer": row["correct_option"],
                                 "Result": "✅" if is_correct else "❌"})
                new_attempts.append({
                    "student_id": student_id, "question_id": row["question_id"],
                    "concept": row["concept"], "difficulty": row["difficulty"],
                    "correct": is_correct, "time_taken": float(np.random.uniform(15, 55)),
                    "date": today_str,
                })

            pct = round(score / len(quiz) * 100, 1)
            st.subheader(f"🏆 Score: {score}/{len(quiz)}  ({pct}%)")
            st.progress(pct / 100)
            st.dataframe(pd.DataFrame(results), use_container_width=True)

            st.session_state.performance = pd.concat(
                [st.session_state.performance, pd.DataFrame(new_attempts)], ignore_index=True)

            for concept in chosen:
                concept_rows = [a for a in new_attempts if a["concept"] == concept]
                if concept_rows:
                    c_acc = float(np.mean([a["correct"] for a in concept_rows]) * 100)
                    st.session_state.schedule = sr.update_schedule(
                        st.session_state.schedule, student_id, concept, c_acc)

            st.success("Attempt logged and revision schedule updated — check 📅 Revision Schedule.")


# --------------------------------------------------------------------------
# PAGE: REVISION SCHEDULE & ALARMS
# --------------------------------------------------------------------------
elif page == "📅 Revision Schedule & Alarms":
    st.header("📅 Spaced-repetition revision schedule")
    st.caption("Uses a simplified SM-2 algorithm: weak concepts come back tomorrow, "
               "strong ones get pushed further out.")

    student_name = st.selectbox("Student", students_df["name"], key="sched_student")
    student_id = students_df.loc[students_df["name"] == student_name, "student_id"].iloc[0]

    schedule_df = st.session_state.schedule
    my_schedule = schedule_df[schedule_df["student_id"] == student_id] if len(schedule_df) else schedule_df

    due = sr.due_today_or_overdue(schedule_df, student_id)
    due_msgs = [f"{row['concept']} is due for revision (was scheduled {row['next_review']})"
                for _, row in due.iterrows()] if len(due) else []

    if due_msgs:
        for m in due_msgs:
            st.markdown(f'<div class="alarm-box">🔔 {m}</div>', unsafe_allow_html=True)
        if st.button("🔔 Enable browser alarm for these"):
            fire_browser_alarm(due_msgs)
            st.toast("Alarm fired! Allow browser notifications if prompted.", icon="🔔")
    else:
        st.success("No revisions due today — you're on track! ✅")

    st.divider()
    st.subheader("Full schedule")
    if len(my_schedule):
        show = my_schedule[["concept", "last_reviewed", "next_review", "interval_days", "ease_factor"]]\
            .sort_values("next_review").rename(columns={
                "concept": "Concept", "last_reviewed": "Last reviewed",
                "next_review": "Next review", "interval_days": "Interval (days)",
                "ease_factor": "Ease factor"})
        st.dataframe(show, use_container_width=True)
    else:
        st.info("No schedule yet — complete a practice quiz on the 📝 Practice & Score page first.")


# --------------------------------------------------------------------------
# PAGE: STUDENT PROFILE
# --------------------------------------------------------------------------
elif page == "👤 Student Profile":
    st.header("👤 Student profile & history")
    student_name = st.selectbox("Student", students_df["name"], key="profile_student")
    student_row = students_df.loc[students_df["name"] == student_name].iloc[0]
    student_id = student_row["student_id"]

    st.markdown(f"**ID:** {student_id} &nbsp;|&nbsp; **Class:** {student_row['class']} "
                f"&nbsp;|&nbsp; **Email:** {student_row['email']}")

    hist = performance_df[performance_df["student_id"] == student_id].copy()
    if hist.empty:
        st.info("No attempts logged for this student yet.")
    else:
        hist["date"] = pd.to_datetime(hist["date"])
        daily = hist.groupby("date")["correct"].mean().mul(100).reset_index()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total attempts", len(hist))
        c2.metric("Overall accuracy", f"{hist['correct'].mean()*100:.1f}%")
        c3.metric("Avg time / question", f"{hist['time_taken'].mean():.1f}s")

        fig = px.line(daily, x="date", y="correct", markers=True,
                      labels={"correct": "Accuracy (%)", "date": "Date"},
                      title="Accuracy trend over time")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Concept-wise breakdown")
        cbd = hist.groupby("concept")["correct"].agg(["mean", "count"]).reset_index()
        cbd["mean"] = (cbd["mean"] * 100).round(1)
        cbd = cbd.rename(columns={"concept": "Concept", "mean": "Accuracy %", "count": "Attempts"})
        st.dataframe(cbd.sort_values("Accuracy %"), use_container_width=True)

        csv = hist.to_csv(index=False)
        st.download_button("⬇️ Download this student's attempt log (CSV)", csv,
                            f"{student_id}_performance.csv")
