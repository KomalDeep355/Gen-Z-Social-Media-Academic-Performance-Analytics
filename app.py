"""
Gen Z Social Media & Academic Performance Analytics — Streamlit Dashboard
Deployable on Hugging Face Spaces (SDK: streamlit).
"""

import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Gen Z Social Media & Academic Performance",
    page_icon="📱",
    layout="wide",
)

# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/student_social_media_data.csv")
    return df


@st.cache_resource
def train_models(df):
    df_ml = df.copy()
    le_platform = LabelEncoder()
    le_year = LabelEncoder()
    le_gender = LabelEncoder()
    df_ml["platform_enc"] = le_platform.fit_transform(df_ml["platform_preference"])
    df_ml["year_enc"] = le_year.fit_transform(df_ml["year"])
    df_ml["gender_enc"] = le_gender.fit_transform(df_ml["gender"])

    features = ["daily_screen_time_hrs", "study_hours", "sleep_hours",
                "attendance_pct", "assignment_completion_pct",
                "platform_enc", "year_enc", "gender_enc"]

    X = df_ml[features]
    rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
    rf.fit(X, df_ml["cgpa"])

    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
    gb.fit(X, df_ml["at_risk"])

    return rf, gb, le_platform, le_year, le_gender, features


df = load_data()
rf_model, gb_model, le_platform, le_year, le_gender, FEATURES = train_models(df)

# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
st.sidebar.title("📱 Filters")
platforms_sel = st.sidebar.multiselect(
    "Platform", sorted(df["platform_preference"].unique()),
    default=sorted(df["platform_preference"].unique())
)
years_sel = st.sidebar.multiselect(
    "Year", sorted(df["year"].unique()), default=sorted(df["year"].unique())
)
screen_range = st.sidebar.slider(
    "Daily Screen Time (hrs)",
    float(df["daily_screen_time_hrs"].min()),
    float(df["daily_screen_time_hrs"].max()),
    (float(df["daily_screen_time_hrs"].min()), float(df["daily_screen_time_hrs"].max()))
)

filtered = df[
    df["platform_preference"].isin(platforms_sel)
    & df["year"].isin(years_sel)
    & df["daily_screen_time_hrs"].between(*screen_range)
]

# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------
st.title("📱 Gen Z Social Media & Academic Performance Analytics")
st.caption("Does Screen Time Really Hurt Your CGPA? — Built on simulated student behavioral data.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Students (filtered)", len(filtered))
c2.metric("Avg CGPA", f"{filtered['cgpa'].mean():.2f}")
c3.metric("Avg Screen Time", f"{filtered['daily_screen_time_hrs'].mean():.1f} hrs/day")
c4.metric("Avg Exam Score", f"{filtered['exam_score'].mean():.1f}")

st.divider()

# ---------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 EDA Dashboard", "🧮 SQL Explorer", "🤖 CGPA Predictor", "⚠️ Risk Classifier"]
)

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Screen Time Distribution")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        sns.histplot(filtered["daily_screen_time_hrs"], bins=20, kde=True, ax=ax, color="#6C5CE7")
        st.pyplot(fig)

        st.subheader("Average CGPA by Platform")
        avg_plat = filtered.groupby("platform_preference")["cgpa"].mean().sort_values(ascending=False)
        st.bar_chart(avg_plat)

    with col2:
        st.subheader("Correlation Heatmap")
        num_cols = ["daily_screen_time_hrs", "study_hours", "sleep_hours",
                    "attendance_pct", "assignment_completion_pct", "exam_score", "cgpa"]
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.heatmap(filtered[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

        st.subheader("CGPA by Screen Time Group")
        order = ["<2h", "2-4h", "4-6h", "6-8h", "8h+"]
        avg_group = (filtered.groupby("screen_time_group", observed=True)["cgpa"]
                     .mean().reindex(order))
        st.bar_chart(avg_group)

with tab2:
    st.subheader("Run SQL Queries Against the Dataset")
    st.caption("Table name: `students`")
    default_query = """SELECT platform_preference AS platform,
       ROUND(AVG(cgpa), 2) AS avg_cgpa,
       ROUND(AVG(exam_score), 1) AS avg_exam_score,
       COUNT(*) AS n_students
FROM students
GROUP BY platform_preference
ORDER BY avg_cgpa DESC;"""
    query = st.text_area("SQL query", value=default_query, height=160)

    if st.button("Run Query"):
        try:
            conn = sqlite3.connect(":memory:")
            filtered.to_sql("students", conn, index=False, if_exists="replace")
            result = pd.read_sql(query, conn)
            st.dataframe(result, use_container_width=True)
        except Exception as e:
            st.error(f"Query error: {e}")

with tab3:
    st.subheader("Predict CGPA from Behavioral Inputs")
    colA, colB, colC = st.columns(3)
    with colA:
        in_screen = st.slider("Daily screen time (hrs)", 0.0, 14.0, 5.5)
        in_study = st.slider("Study hours (hrs/day)", 0.0, 9.0, 3.0)
    with colB:
        in_sleep = st.slider("Sleep hours", 3.0, 10.0, 7.0)
        in_attendance = st.slider("Attendance %", 40.0, 100.0, 85.0)
    with colC:
        in_assignment = st.slider("Assignment completion %", 10.0, 100.0, 70.0)
        in_platform = st.selectbox("Platform preference", sorted(df["platform_preference"].unique()))
        in_year = st.selectbox("Year", sorted(df["year"].unique()))
        in_gender = st.selectbox("Gender", sorted(df["gender"].unique()))

    if st.button("Predict CGPA"):
        row = pd.DataFrame([{
            "daily_screen_time_hrs": in_screen,
            "study_hours": in_study,
            "sleep_hours": in_sleep,
            "attendance_pct": in_attendance,
            "assignment_completion_pct": in_assignment,
            "platform_enc": le_platform.transform([in_platform])[0],
            "year_enc": le_year.transform([in_year])[0],
            "gender_enc": le_gender.transform([in_gender])[0],
        }])[FEATURES]
        pred_cgpa = rf_model.predict(row)[0]
        st.success(f"Predicted CGPA: **{pred_cgpa:.2f}** / 10")

        importances = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values()
        st.caption("Model: Random Forest Regressor — feature importances below")
        st.bar_chart(importances)

with tab4:
    st.subheader("Academic Risk Classifier")
    st.caption("Flags students in the bottom ~20% CGPA band based on behavioral inputs.")
    colA, colB, colC = st.columns(3)
    with colA:
        r_screen = st.slider("Daily screen time (hrs) ", 0.0, 14.0, 7.0, key="r1")
        r_study = st.slider("Study hours (hrs/day) ", 0.0, 9.0, 1.5, key="r2")
    with colB:
        r_sleep = st.slider("Sleep hours ", 3.0, 10.0, 6.0, key="r3")
        r_attendance = st.slider("Attendance % ", 40.0, 100.0, 70.0, key="r4")
    with colC:
        r_assignment = st.slider("Assignment completion % ", 10.0, 100.0, 50.0, key="r5")
        r_platform = st.selectbox("Platform preference ", sorted(df["platform_preference"].unique()), key="r6")
        r_year = st.selectbox("Year ", sorted(df["year"].unique()), key="r7")
        r_gender = st.selectbox("Gender ", sorted(df["gender"].unique()), key="r8")

    if st.button("Assess Risk"):
        row = pd.DataFrame([{
            "daily_screen_time_hrs": r_screen,
            "study_hours": r_study,
            "sleep_hours": r_sleep,
            "attendance_pct": r_attendance,
            "assignment_completion_pct": r_assignment,
            "platform_enc": le_platform.transform([r_platform])[0],
            "year_enc": le_year.transform([r_year])[0],
            "gender_enc": le_gender.transform([r_gender])[0],
        }])[FEATURES]
        proba = gb_model.predict_proba(row)[0][1]
        label = "⚠️ At Risk" if proba >= 0.5 else "✅ On Track"
        st.metric("Risk Probability", f"{proba*100:.1f}%", label)

st.divider()
st.caption(
    "Note: This dataset is simulated for demonstration purposes. "
    "Relationships shown illustrate analytics/ML workflow, not verified real-world findings."
)
