import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the preprocessed dataset
df = pd.read_csv('Processed_Student_Depression_Dataset.csv')

# Define features (X) and target variable (y)
y = df['Depression']  # Assuming 'Depression' is the target column
X = df.drop(columns=['Depression'])

# Drop rows with any missing values
#df = df.dropna()

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## Feature Selection based on 15 most important features

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


## model with all features
model_full = RandomForestClassifier(n_estimators=100, random_state=42)
model_full.fit(X_train, y_train)
y_pred_full = model_full.predict(X_test)

accuracy_full = accuracy_score(y_test, y_pred_full)
print(f"\nFull Feature Model Accuracy: {accuracy_full:.4f}")
print("Classification Report - Full Model:")
print(classification_report(y_test, y_pred_full))


# Get feature importances of the model using full preprocessed dataset
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

## Feature Selection based on n most important features
# Set the threshold (top 15 features)
top_features = importance_df.head(15)['Feature'].values
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

# Retrain model with the selected features
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_selected, y_train)

# Evaluate the model
y_pred = model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# show features used
print("Top 15 features used:")
print(top_features)

# Save the trained model
joblib.dump(model, '[FS-dropna]depression_prediction_model.pkl')
print("Model saved as '[FS-dropna]depression_prediction_model.pkl'")
