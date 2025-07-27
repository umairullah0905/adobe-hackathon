# train_model.py (Adjusted for higher sensitivity to training data)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import sys

# Load the dataset
try:
    df = pd.read_csv('training_datas.csv')
except FileNotFoundError:
    sys.exit("Error: training_datas.csv not found. Please run create_data.py first.")

# The feature list remains the same, using our robust feature engineering
features = [
    'font_size', 
    'is_bold', 
    'word_count', 
    'size_diff_from_prev', 
    'starts_with_pattern',
    'y_position',
    'is_centered'
]
text_case_cols = [col for col in df.columns if col.startswith('text_case_')]
features.extend(text_case_cols)

target = 'label'

# Ensure all feature columns exist in the DataFrame
for col in features:
    if col not in df.columns:
        df[col] = 0

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training a more sensitive model...")

# --- MODIFICATION: Hyperparameters are relaxed to allow the model more complexity ---
# max_depth=None lets trees grow fully.
# min_samples_leaf=1 allows the model to create a "rule" for a single data point.
# n_estimators is slightly increased.
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

print("\n--- Model Training Complete ---")

print("\nEvaluating model on test data...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

# You can also check the accuracy on the training data itself to see how well it was memorized
print("\nEvaluating model on TRAINING data (to check for sensitivity)...")
y_train_pred = model.predict(X_train)
print(classification_report(y_train, y_train_pred, zero_division=0))


joblib.dump(model, 'heading_classifier.joblib')
print("\nModel saved to heading_classifier.joblib")