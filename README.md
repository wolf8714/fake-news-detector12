# Fake News Detector

Classifies news articles as real or fake using machine learning.

## Business Problem
Fake news spreads fast and causes harm. This app flags articles that look fake.

## Dataset
ISOT Fake and Real News dataset (Kaggle) — 44,898 articles (23,481 fake, 21,417 real).

## Approach
- Text cleaning: removed source tags (e.g. "Reuters") that caused data leakage.
- Features: TF-IDF (top 5,000 words).
- Model: Logistic Regression — Accuracy 0.979, F1 0.980.

## Known Limitation
Real and fake articles come from different sources, so accuracy partly reflects
writing-style differences between outlets, not pure truthfulness. This is a known
issue with this dataset and is stated openly.

## Live App
[Streamlit link — add after deploying]

## Demo Video
[Video link — add after recording]

## How to Run
1. `pip install -r requirements.txt`
2. `streamlit run streamlit_app.py`
