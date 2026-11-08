"""
Student Performance Prediction System
Main model: Multiple Linear Regression
Extensions: Logistic Regression and CART (Pass/Fail classification)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

# Generate a reproducible sample dataset (replace with pd.read_csv for real data).
rng = np.random.default_rng(42)
n = 240
study = np.clip(rng.normal(3.5, 1.5, n), 0.5, 8)
attendance = np.clip(rng.normal(75, 15, n), 35, 100)
previous = np.clip(rng.normal(55, 20, n), 5, 100)
assignments = np.clip(rng.normal(70, 20, n), 0, 100)
participation = np.clip(rng.normal(3, 1.2, n), 1, 5)
noise = rng.normal(0, 8, n)
final_marks = np.clip(
    2.8*study + 0.12*attendance + 0.22*previous
    + 0.07*assignments + participation - 5 + noise, 0, 100
)
df = pd.DataFrame({
    "Study_Hours_Per_Day": study,
    "Attendance_Percentage": attendance,
    "Previous_Exam_Marks": previous,
    "Assignment_Completion_Percentage": assignments,
    "Class_Participation_Score": participation,
    "Final_Exam_Marks": final_marks
})
print("First five records:\\n", df.head())

# Multiple Linear Regression
features = ["Study_Hours_Per_Day", "Attendance_Percentage",
            "Previous_Exam_Marks", "Assignment_Completion_Percentage",
            "Class_Participation_Score"]
X, y = df[features], df["Final_Exam_Marks"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
reg_model = LinearRegression().fit(X_train, y_train)
y_pred = reg_model.predict(X_test)
print("\\nMULTIPLE LINEAR REGRESSION")
print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
print("R-squared:", round(r2_score(y_test, y_pred), 3))

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Final Exam Marks")
plt.ylabel("Predicted Final Exam Marks")
plt.title("Actual vs Predicted Student Marks")
plt.tight_layout()
plt.show()

# Logistic Regression: 40 or above = Pass (1), otherwise Fail (0)
labels = (df["Final_Exam_Marks"] >= 40).astype(int)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)
log_model = LogisticRegression(max_iter=1000).fit(Xc_train, yc_train)
log_pred = log_model.predict(Xc_test)
print("\\nLOGISTIC REGRESSION ACCURACY:",
      round(accuracy_score(yc_test, log_pred), 3))

# CART classification tree
cart_model = DecisionTreeClassifier(max_depth=3, random_state=42)
cart_model.fit(Xc_train, yc_train)
cart_pred = cart_model.predict(Xc_test)
print("CART ACCURACY:", round(accuracy_score(yc_test, cart_pred), 3))
print("\\nCART RULES:\\n", export_text(cart_model, feature_names=features))

# Predict for a new student
new_student = pd.DataFrame([{
    "Study_Hours_Per_Day": 4.0,
    "Attendance_Percentage": 85.0,
    "Previous_Exam_Marks": 70.0,
    "Assignment_Completion_Percentage": 80.0,
    "Class_Participation_Score": 4.0
}])
score = reg_model.predict(new_student)[0]
print("\\nPREDICTED FINAL MARKS:", round(score, 2))
print("PREDICTED RESULT:", "Pass" if score >= 40 else "Fail")
