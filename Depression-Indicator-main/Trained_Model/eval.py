import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import time

# Load preprocessed full dataset
df = pd.read_csv("Processed_Student_Depression_Dataset.csv")

# Define the top 15 features (used in training)
top_features = [
    'Academic Pressure',
    'Have you ever had suicidal thoughts ?_Yes',
    'Have you ever had suicidal thoughts ?_No',
    'Financial Stress',
    'CGPA',
    'Age',
    'Work/Study Hours',
    'Study Satisfaction',
    'Dietary Habits',
    'Sleep Duration',
    'Gender_Female',
    'Gender_Male',
    'Degree_Class 12',
    'Family History of Mental Illness_Yes',
    'Family History of Mental Illness_No'
]

# Split into features and labels
y = df["Depression"]
X = df[top_features]

# Split into training and test sets (same as training script)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load model
model = joblib.load("[FS]depression_prediction_model.pkl")

# Measure time to predict
start_time = time.time()
y_pred = model.predict(X_test)
end_time = time.time()
prediction_time = end_time - start_time
print(f"\nTime taken to predict on test set: {prediction_time:.4f} seconds")

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.clf()
plt.cla()
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Depressed', 'Depressed'], yticklabels=['Not Depressed', 'Depressed'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.show()

from sklearn.model_selection import StratifiedKFold, cross_val_score
# Measure time for cross-validation
start_cv = time.time()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
end_cv = time.time()
cv_time = end_cv - start_cv
print(f"Average F1 (5-fold CV): {scores.mean():.4f}")
print(f"Time taken for 5-fold cross-validation: {cv_time:.4f} seconds")

X_test[(y_test == 0) & (y_pred == 1)]  # False Positives
X_test[(y_test == 1) & (y_pred == 0)]  # False Negatives

##
## End of Eval
##

# get important features
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)

# graph 15 most important features
top_n = 15
top_features = importance_df.head(top_n)
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top Features - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

## Extra for machines with newly added python
library(reticulate)
use_python("C:\\Users\\djlan\\AppData\\Local\\Programs\\Python\\Python313\\python.exe", required = TRUE)
repl_python()
reticulate::py_install(c("pandas", "joblib", "scikit-learn", "matplotlib", "seaborn"))

os.chdir("C:\\Users\\djlan\\OneDrive\\Creative Cloud Files\\Desktop\\MILESTONE 2")
