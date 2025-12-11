import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
file_path = "C:\Users\ferna\OneDrive\Desktop\SE Loan Approval\dataset"
data = pd.read_csv(file_path)

# Clean column names (remove leading/trailing spaces)
data.columns = data.columns.str.strip()

# Encode categorical columns
label_encoder = LabelEncoder()
categorical_columns = ['education', 'self_employed', 'loan_status']
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Define features (X) and target (y)
X = data.drop(columns=['loan_id', 'loan_status'])  # Exclude ID and target column
y = data['loan_status']  # 1: Approved, 0: Rejected

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

# Display sample predictions
loan_predictions = ['Loan Approved' if prob >= 0.5 else 'Loan Denied' for prob in y_pred_prob]
print("\nSample Predictions with Probabilities:")
for i in range(min(10, len(y_pred_prob))):
    print(f"Predicted Probability: {y_pred_prob[i]:.2f}, Prediction: {loan_predictions[i]}")
