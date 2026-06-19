"""
03_genz_social_media_analytics
Cleaning + EDA + SQL + Analytics + ML

Run from the project root:
    python analysis.py
Produces chart PNGs in images/ and a trained model summary printed to console.
"""

import sqlite3
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              mean_absolute_error, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# ---------------------------------------------------------------------
# 1. LOAD + CLEAN
# ---------------------------------------------------------------------
df = pd.read_csv("data/student_social_media_data.csv")
df = df.drop_duplicates().dropna()
print("Shape:", df.shape)
print(df.describe(include="all").T[["mean", "std", "min", "max"]].head(10))

# ---------------------------------------------------------------------
# 2. SQL ANALYTICS LAYER (SQLite, in-memory)
# ---------------------------------------------------------------------
conn = sqlite3.connect(":memory:")
df.to_sql("students", conn, index=False, if_exists="replace")

q1 = pd.read_sql("""
    SELECT platform_preference AS platform, ROUND(AVG(cgpa),2) AS avg_cgpa
    FROM students GROUP BY platform_preference ORDER BY avg_cgpa DESC
""", conn)

q2 = pd.read_sql("""
    SELECT screen_time_group, ROUND(AVG(cgpa),2) AS avg_cgpa,
           ROUND(AVG(exam_score),1) AS avg_exam_score
    FROM students GROUP BY screen_time_group
""", conn)

q3 = pd.read_sql("""
    SELECT ROUND(study_hours) AS study_hours_rounded,
           ROUND(AVG(exam_score),1) AS avg_exam_score
    FROM students GROUP BY study_hours_rounded ORDER BY study_hours_rounded
""", conn)

print("\n--- Avg CGPA by platform ---\n", q1)
print("\n--- Avg CGPA/Exam by screen-time group ---\n", q2)

# ---------------------------------------------------------------------
# 3. EDA CHARTS
# ---------------------------------------------------------------------
# Chart 1: screen time distribution
plt.figure(figsize=(7, 4))
sns.histplot(df["daily_screen_time_hrs"], bins=25, kde=True, color="#6C5CE7")
plt.title("Daily Screen Time Distribution")
plt.xlabel("Hours/day")
plt.tight_layout()
plt.savefig("images/chart1_screen_time.png")
plt.close()

# Chart 2: platform vs CGPA
plt.figure(figsize=(7, 4))
sns.barplot(data=q1, x="platform", y="avg_cgpa", palette="viridis")
plt.title("Average CGPA by Platform Preference")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("images/chart2_platform_analysis.png")
plt.close()

# Chart 3: correlation heatmap
num_cols = ["daily_screen_time_hrs", "study_hours", "sleep_hours",
            "attendance_pct", "assignment_completion_pct", "exam_score", "cgpa"]
plt.figure(figsize=(7, 5))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/chart3_correlation_heatmap.png")
plt.close()

# Chart 4: study hours vs screen time, colored by CGPA
plt.figure(figsize=(7, 4))
sc = plt.scatter(df["daily_screen_time_hrs"], df["study_hours"],
                  c=df["cgpa"], cmap="plasma", alpha=0.6)
plt.colorbar(sc, label="CGPA")
plt.xlabel("Screen Time (hrs/day)")
plt.ylabel("Study Hours (hrs/day)")
plt.title("Study vs Screen Time (colored by CGPA)")
plt.tight_layout()
plt.savefig("images/chart4_study_vs_social.png")
plt.close()

print("\nSaved 4 charts to images/")

# ---------------------------------------------------------------------
# 4. ML LAYER
# ---------------------------------------------------------------------
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

# --- 4a. Regression: predict CGPA ---
X = df_ml[features]
y_reg = df_ml["cgpa"]
X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

lin = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42).fit(X_train, y_train)

for name, model in [("Linear Regression", lin), ("Random Forest", rf)]:
    pred = model.predict(X_test)
    print(f"\n[{name}] R2={r2_score(y_test, pred):.3f}  MAE={mean_absolute_error(y_test, pred):.3f}")

# --- 4b. Classification: predict at-risk students ---
y_clf = df_ml["at_risk"]
X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42).fit(X_train, y_train)
pred = gb.predict(X_test)
proba = gb.predict_proba(X_test)[:, 1]
print("\n[Gradient Boosting - At Risk Classifier]")
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print("ROC-AUC :", round(roc_auc_score(y_test, proba), 3))
print(classification_report(y_test, pred))

# Save processed file + feature importances for the dashboard
df.to_csv("data/processed_student_scores.csv", index=False)

importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\nTop CGPA predictors (Random Forest):\n", importances)
