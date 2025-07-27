# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load the dataset
df = pd.read_csv('training_datas.csv')

# --- CHANGE: Add the new features to the list ---
features = [
    'font_size', 
    'is_bold', 
    'word_count', 
    'size_diff_from_prev', 
    'starts_with_numbering',
    'y_position',       # <-- NEW
    'is_centered'       # <-- NEW
]
target = 'label'

# The rest of the script is unchanged
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

print("\nEvaluating model on test data...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, 'heading_classifier.joblib')
print("\nModel saved to heading_classifier.joblib")