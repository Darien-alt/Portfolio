import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Load preprocessed dataset
df = pd.read_csv("Processed_Student_Depression_Dataset.csv")

drop_cols1 = df.columns[29:59]   # Family History and Degree columns
drop_cols2 = df.columns[11:27]   # Gender and Profession columns
drop_cols = drop_cols1.union(drop_cols2)

df = df.drop(drop_cols, axis=1)

#imputer = SimpleImputer(strategy='mean')
# Drop rows with any NaN values
df = df.dropna()

# Split features and target
y = df["Depression"]
X = df.drop(columns=["Depression"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the SVM model
model = joblib.load("svm_model.pkl")

# Predict and evaluate
y_pred = model.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


plt.clf()
plt.cla()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Depressed', 'Depressed'],
            yticklabels=['Not Depressed', 'Depressed'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - SVM')
plt.show()

from sklearn.model_selection import StratifiedKFold, cross_val_score
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
print("Average F1:", scores.mean())
