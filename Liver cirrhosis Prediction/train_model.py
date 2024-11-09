import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load the data
data = pd.read_csv('cirrhosis.csv')  # Use the correct path

# Select relevant features and target, handle missing values in 'Stage'
data = data[['Age', 'Bilirubin', 'Albumin', 'Ascites', 'Sex', 'Stage']]
data = data.dropna(subset=['Stage'])  # Drop rows where 'Stage' is NaN

# Convert categorical columns to numeric
data['Ascites'] = data['Ascites'].apply(lambda x: 1 if x == "Yes" else 0)
data['Sex'] = data['Sex'].apply(lambda x: 1 if x == "Male" else 0)

# Split data into features and target
X = data[['Age', 'Bilirubin', 'Albumin', 'Ascites', 'Sex']]
y = data['Stage']

# Check for NaN values in features or target again to ensure data integrity
if X.isna().sum().sum() > 0 or y.isna().sum() > 0:
    raise ValueError("Dataset contains NaN values after preprocessing.")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model trained and saved as 'model.pkl'")
