# 🎭 Mood Predictor from Smartphone Activity

This project uses simulated smartphone usage behavior to predict a user's mood using machine learning.

## 🚀 Features
- Simulates data like screen time, night usage, app usage
- Predicts mood categories: Happy, Sad, Neutral, Stressed, Excited
- Uses Random Forest Classifier
- Visualizes model performance with confusion matrix

## 📊 Example Data Points
- screen_time_min
- night_usage_min
- unlock_count
- notifications
- typing_speed
- app usage per category
- time of day of usage

## 🧠 Tech Stack
- Python, Pandas, NumPy
- Scikit-learn
- Matplotlib & Seaborn

## 📁 Run it
```bash
pip install -r requirements.txt
python mood_predictor.py
