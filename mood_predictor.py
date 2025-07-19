import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime, timedelta
import random

# 1️⃣ Simulate mood and usage data
def simulate_mood_data(n_days=60):
    moods = ['Happy', 'Sad', 'Neutral', 'Stressed', 'Excited']
    data = []

    for i in range(n_days):
        date = pd.Timestamp.today().normalize() - pd.Timedelta(days=i)
        screen_time = np.random.randint(60, 300)  # in minutes
        night_usage = np.random.randint(0, 90)
        unlocks = np.random.randint(10, 80)
        notifications = np.random.randint(20, 100)
        typing_speed = np.random.randint(100, 400)  # keystrokes per hour
        app_social = np.random.randint(10, 120)
        app_work = np.random.randint(5, 90)
        app_entertainment = np.random.randint(5, 90)
        time_of_day = random.choice(['Morning', 'Afternoon', 'Evening', 'Late Night'])

        # Rule-based mood simulation
        if screen_time > 240 or night_usage > 60:
            mood = 'Stressed'
        elif app_social > 80 and unlocks < 30:
            mood = 'Happy'
        elif notifications > 80 and typing_speed < 200:
            mood = 'Sad'
        elif app_work > 60 and app_entertainment < 20:
            mood = 'Stressed'
        else:
            mood = random.choice(moods)

        data.append({
            'date': date.date(),
            'screen_time_min': screen_time,
            'night_usage_min': night_usage,
            'unlock_count': unlocks,
            'notification_count': notifications,
            'typing_speed': typing_speed,
            'app_social_min': app_social,
            'app_work_min': app_work,
            'app_entertainment_min': app_entertainment,
            'usage_peak': time_of_day,
            'mood': mood
        })
    return pd.DataFrame(data)

# 2️⃣ Generate and save data
df = simulate_mood_data()
df.to_csv('mood_data.csv', index=False)
print("✅ Simulated mood data saved to mood_data.csv")
print(df.head())

# 3️⃣ Encode and prepare features
df_encoded = pd.get_dummies(df.drop(columns=['date']), columns=['usage_peak'])
X = df_encoded.drop('mood', axis=1)
y = df_encoded['mood']

# 4️⃣ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6️⃣ Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n📌 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
