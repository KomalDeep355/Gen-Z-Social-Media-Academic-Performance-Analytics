---
title: Gen Z Social Media & Academic Performance Analytics
emoji: 📱
colorFrom: purple
colorTo: pink
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# 📱 Gen Z Social Media & Academic Performance Analytics

Does Screen Time Really Hurt Your CGPA?

A complete behavioral analytics & student performance project exploring the
relationship between social media usage, screen time habits, and academic
outcomes in Gen Z students — built end-to-end as a data science project
(EDA → SQL → ML → Streamlit dashboard) and deployed on Hugging Face Spaces.

> **Note:** The dataset here is **simulated** (generated with realistic,
> directionally-plausible relationships) so the project is fully reproducible
> without needing to download anything from Kaggle. You can swap in a real
> dataset with the same column names and everything else just works.

## 🗂 Project Structure

```
.
├── app.py                          # Streamlit dashboard (HF Spaces entry point)
├── analysis.py                     # EDA + SQL + ML pipeline (script form of the notebook)
├── requirements.txt
├── data/
│   ├── generate_data.py            # Synthetic data generator
│   ├── student_social_media_data.csv
│   └── processed_student_scores.csv
└── images/
    ├── chart1_screen_time.png
    ├── chart2_platform_analysis.png
    ├── chart3_correlation_heatmap.png
    └── chart4_study_vs_social.png
```

## 🛠 Tech Stack
- **Analytics**: Python, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Streamlit charts
- **Database**: SQLite (in-memory SQL layer)
- **ML**: scikit-learn (Random Forest regression, Gradient Boosting classification)
- **Deployment**: Streamlit on Hugging Face Spaces

## 📊 What's Inside

- **EDA dashboard** — screen time distribution, platform vs CGPA, correlation
  heatmap, study-vs-screen-time scatter, all filterable by platform/year/screen time.
- **SQL explorer** — write and run your own SQL against the `students` table
  directly in the app.
- **CGPA predictor** — Random Forest regression estimating CGPA from screen
  time, study hours, sleep, attendance, assignment completion, platform, year, gender.
- **Academic risk classifier** — Gradient Boosting model flagging students in
  the bottom ~20% CGPA band, with a risk-probability output.

## 🚀 Run Locally

```bash
pip install -r requirements.txt
python data/generate_data.py      # regenerate the dataset if needed
python analysis.py                # run the full EDA/SQL/ML pipeline + save charts
streamlit run app.py              # launch the dashboard
```

## ☁️ Deploy on Hugging Face Spaces

1. Create a new Space → SDK: **Streamlit**.
2. Push this folder's contents (including `data/student_social_media_data.csv`)
   to the Space repo — the YAML block at the top of this README is the Spaces
   config, so no extra setup is needed.
3. Spaces will install `requirements.txt` and run `app.py` automatically.

```bash
# from this project folder
git init
git lfs install
huggingface-cli login
git remote add space https://huggingface.co/spaces/<your-username>/genz-social-academic-analytics
git add .
git commit -m "Initial deployment"
git push space main
```

## 🔍 Key Findings (on the simulated dataset)

- Study hours and daily screen time are the strongest predictors of CGPA.
- Higher screen-time buckets trend toward lower average CGPA and exam scores.
- Platform choice alone has a smaller effect than overall screen time and
  study habits — behavioral combination matters more than any single metric.

## 📈 Possible Extensions

- Swap in a real Kaggle dataset with matching column names.
- Add XGBoost/LightGBM models and hyperparameter tuning.
- Add a Power BI version of the dashboard for a non-technical audience.
- Add authentication + per-student tracking for longitudinal analysis.
