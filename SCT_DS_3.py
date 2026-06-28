import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import requests
from zipfile import ZipFile
from io import BytesIO

# Download dataset
url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

response = requests.get(url)
outer_zip = ZipFile(BytesIO(response.content))

# Open the inner zip
with outer_zip.open("bank-additional.zip") as inner_data:
    inner_zip = ZipFile(BytesIO(inner_data.read()))

    # Read CSV
    with inner_zip.open("bank-additional/bank-additional-full.csv") as f:
        df = pd.read_csv(f, sep=';')

# Convert categorical columns to numeric
df_encoded = pd.get_dummies(df, drop_first=True)

# Features and target
X = df_encoded.drop('y_yes', axis=1)
y = df_encoded['y_yes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))