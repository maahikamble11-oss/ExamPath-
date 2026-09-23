"""
ML core of ExamPath.

1. train_mastery_model() -> a RandomForestClassifier trained on the student's
   attempt history (performance.csv) that predicts P(answer correct) for a
   given concept/difficulty/recent-accuracy combination. This probability is
   used as a "mastery score" (0-100%) per concept.

2. cluster_students() -> KMeans clustering of students into
   Needs Support / Developing / Advanced learner segments based on overall
   accuracy and speed.

"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

DIFF_MAP = {"Easy": 1, "Medium": 2, "Hard": 3}
MIN_ROWS_FOR_MODEL = 15


def build_features(performance_df: pd.DataFrame) -> pd.DataFrame:
    """Adds numeric difficulty + a rolling (pre-attempt) accuracy feature."""
    df = performance_df.copy()
    df["difficulty_num"] = df["difficulty"].map(DIFF_MAP).fillna(2)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["student_id", "concept", "date"])
    df["rolling_accuracy"] = (
        df.groupby(["student_id", "concept"])["correct"]
        .transform(lambda x: x.shift().expanding().mean())
    )
    df["rolling_accuracy"] = df["rolling_accuracy"].fillna(0.5)
    return df


def train_mastery_model(performance_df: pd.DataFrame):
    """Returns (model, label_encoder, feature_df) or (None, None, df) if too little data."""
    df = build_features(performance_df)
    if len(df) < MIN_ROWS_FOR_MODEL or df["correct"].nunique() < 2:
        return None, None, df

    le = LabelEncoder()
    # Build the encoded column explicitly so static type checking does not need
    # to resolve sklearn's ArrayLike return type.
    le.fit(df["concept"])
    concept_codes = {concept: index for index, concept in enumerate(le.classes_)}
    df["concept_enc"] = pd.Series(
        [concept_codes[concept] for concept in df["concept"]],
        index=df.index,
        dtype="int64",
    )
    feature_cols = ["difficulty_num", "time_taken", "rolling_accuracy", "concept_enc"]
    X = df[feature_cols]
    y = df["correct"]

    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=2, random_state=42
    )
    model.fit(X, y)
    return model, le, df


def predict_mastery(model, le, feature_df: pd.DataFrame, student_id: str) -> pd.DataFrame:
    """Predicted mastery % per concept for one student, sorted weakest-first."""
    student_df = feature_df[feature_df["student_id"] == student_id]
    if student_df.empty:
        return pd.DataFrame(columns=["concept", "mastery", "attempts", "accuracy", "avg_time"])

    results = []
    for concept, g in student_df.groupby("concept"):
        feat = pd.DataFrame([{
            "difficulty_num": g["difficulty_num"].mean(),
            "time_taken": g["time_taken"].mean(),
            "rolling_accuracy": g["correct"].mean(),
            "concept_enc": le.transform([concept])[0] if concept in le.classes_ else 0,
        }])
        prob = model.predict_proba(feat)[0][1] if model is not None else g["correct"].mean()
        results.append({
            "concept": concept,
            "mastery": round(float(prob) * 100, 1),
            "attempts": int(len(g)),
            "accuracy": round(float(g["correct"].mean()) * 100, 1),
            "avg_time": round(float(g["time_taken"].mean()), 1),
        })
    return pd.DataFrame(results).sort_values("mastery").reset_index(drop=True)


def cluster_students(performance_df: pd.DataFrame) -> pd.DataFrame:
    """Segments students into learner tiers using KMeans on accuracy & speed."""
    summary = performance_df.groupby("student_id").agg(
        avg_accuracy=("correct", "mean"),
        avg_time=("time_taken", "mean"),
        attempts=("correct", "count"),
    ).reset_index()

    if len(summary) < 3:
        summary["cluster_label"] = "Not enough data"
        return summary

    km = KMeans(n_clusters=3, n_init=10, random_state=42)
    summary["cluster"] = km.fit_predict(summary[["avg_accuracy", "avg_time"]])

    order = summary.groupby("cluster")["avg_accuracy"].mean().sort_values().index.tolist()
    label_map = {order[0]: "Needs Support", order[1]: "Developing", order[2]: "Advanced"}
    summary["cluster_label"] = summary["cluster"].map(label_map)
    return summary


def weak_concepts_for_student(mastery_df: pd.DataFrame, threshold: float = 50.0) -> list:
    """List of concept names below the mastery threshold."""
    if mastery_df.empty:
        return []
    return mastery_df[mastery_df["mastery"] < threshold]["concept"].tolist()
