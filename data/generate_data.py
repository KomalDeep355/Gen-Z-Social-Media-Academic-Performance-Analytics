"""
Generates a realistic simulated dataset for the
Gen Z Social Media & Academic Performance Analytics project.

Run: python generate_data.py
Output: student_social_media_data.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

platforms = ["Instagram", "YouTube", "TikTok", "Snapchat", "Twitter/X", "WhatsApp"]
genders = ["Male", "Female", "Other"]
years = ["1st Year", "2nd Year", "3rd Year", "4th Year"]

# --- Core behavioral variables ---
screen_time = np.clip(np.random.normal(5.5, 2.2, N), 0.5, 14)          # hrs/day
sleep_hours = np.clip(np.random.normal(7.0, 1.3, N) - 0.15 * (screen_time - 5.5), 3, 10)
study_hours = np.clip(np.random.normal(3.0, 1.5, N) - 0.18 * (screen_time - 5.5), 0, 9)
attendance = np.clip(np.random.normal(85, 10, N) - 1.2 * (screen_time - 5.5), 40, 100)
platform_pref = np.random.choice(platforms, N, p=[0.28, 0.22, 0.22, 0.12, 0.08, 0.08])
gender = np.random.choice(genders, N, p=[0.48, 0.49, 0.03])
year = np.random.choice(years, N)

# Platform "academic drag" weight (purely illustrative, not a real claim)
platform_weight = {
    "Instagram": 0.9, "YouTube": 0.4, "TikTok": 1.1,
    "Snapchat": 0.8, "Twitter/X": 0.6, "WhatsApp": 0.3,
}
platform_effect = np.array([platform_weight[p] for p in platform_pref])

# --- Outcome variables (simulated relationships, not real-world fact) ---
noise = np.random.normal(0, 0.4, N)
cgpa = (
    8.6
    - 0.10 * screen_time
    - 0.05 * platform_effect * screen_time / 5
    + 0.18 * study_hours
    + 0.05 * sleep_hours
    + 0.01 * (attendance - 85)
    + noise
)
cgpa = np.clip(cgpa, 4.0, 10.0).round(2)

exam_score = np.clip(
    55
    - 2.0 * screen_time
    + 4.5 * study_hours
    + 1.5 * sleep_hours
    + 0.3 * (attendance - 85)
    + np.random.normal(0, 6, N),
    20, 100
).round(1)

assignment_completion = np.clip(
    70 - 1.8 * screen_time + 3.0 * study_hours + np.random.normal(0, 8, N), 10, 100
).round(1)

df = pd.DataFrame({
    "student_id": [f"S{i:05d}" for i in range(1, N + 1)],
    "gender": gender,
    "year": year,
    "platform_preference": platform_pref,
    "daily_screen_time_hrs": screen_time.round(2),
    "study_hours": study_hours.round(2),
    "sleep_hours": sleep_hours.round(2),
    "attendance_pct": attendance.round(1),
    "assignment_completion_pct": assignment_completion,
    "exam_score": exam_score,
    "cgpa": cgpa,
})

# screen time buckets used throughout the analysis
bins = [0, 2, 4, 6, 8, 24]
labels = ["<2h", "2-4h", "4-6h", "6-8h", "8h+"]
df["screen_time_group"] = pd.cut(df["daily_screen_time_hrs"], bins=bins, labels=labels)

# simple academic-risk flag for the classification task (bottom ~20% of CGPA)
risk_threshold = df["cgpa"].quantile(0.20)
df["at_risk"] = (df["cgpa"] < risk_threshold).astype(int)

df.to_csv("student_social_media_data.csv", index=False)
print(f"Saved student_social_media_data.csv with {len(df)} rows")
print(df.head())
