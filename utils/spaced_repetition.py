"""
A simplified SM-2 style spaced-repetition scheduler.

Every time a student attempts a quiz on a concept, we recompute:
  - ease_factor : how easily the concept "sticks" (grows with good performance)
  - interval_days : how many days until the next revision
  - next_review : last_reviewed + interval_days

Weak concepts (mastery < 50%) are always scheduled for revision tomorrow.
Everything is kept in a plain pandas DataFrame backed by a CSV file
(data/revision_schedule.csv) - no SQL/database used.
"""
import pandas as pd
from datetime import datetime, timedelta

SCHEDULE_COLUMNS = [
    "student_id", "concept", "last_reviewed", "next_review",
    "interval_days", "ease_factor",
]


def empty_schedule() -> pd.DataFrame:
    return pd.DataFrame(columns=SCHEDULE_COLUMNS)


def update_schedule(schedule_df: pd.DataFrame, student_id: str, concept: str,
                     mastery_pct: float, today=None) -> pd.DataFrame:
    today = today or datetime.now().date()
    mask = (schedule_df["student_id"] == student_id) & (schedule_df["concept"] == concept)

    if mask.any():
        row = schedule_df.loc[mask].iloc[0]
        ease = float(row["ease_factor"])
        interval = float(row["interval_days"])
    else:
        ease = 2.5
        interval = 1.0

    if mastery_pct < 50:
        interval = 1
        ease = max(1.3, ease - 0.2)
    elif mastery_pct < 75:
        interval = max(1, round(interval * 1.2))
        ease = max(1.3, ease - 0.05)
    else:
        interval = max(1, round(interval * ease))
        ease = ease + 0.1

    next_review = today + timedelta(days=int(interval))
    new_row = {
        "student_id": student_id,
        "concept": concept,
        "last_reviewed": today.isoformat(),
        "next_review": next_review.isoformat(),
        "interval_days": int(interval),
        "ease_factor": round(ease, 2),
    }

    schedule_df = schedule_df[~mask]
    schedule_df = pd.concat([schedule_df, pd.DataFrame([new_row])], ignore_index=True)
    return schedule_df


def due_today_or_overdue(schedule_df: pd.DataFrame, student_id: str, today=None) -> pd.DataFrame:
    today = today or datetime.now().date()
    if schedule_df.empty:
        return schedule_df
    df = schedule_df[schedule_df["student_id"] == student_id].copy()
    if df.empty:
        return df
    df["next_review_date"] = pd.to_datetime(df["next_review"]).dt.date
    return df[df["next_review_date"] <= today].sort_values("next_review_date")
