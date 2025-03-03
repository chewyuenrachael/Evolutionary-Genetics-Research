# Training a Machine Learning Model to Distinguish Introgression from ILS
# This script trains a classifier using the extracted features.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_model(feature_file, model_output):
    """
    Train a machine learning model.

    Args:
        feature_file (str): Path to the CSV file with extracted features.
        model_output (str): Path to save the trained model.
    """
    # Load features
    df = pd.read_csv(feature_file)

    # Prepare features and labels
    X = df.drop(columns=['window_start', 'window_end', 'label'])
    y = df['label']  # Assuming 'label' column indicates introgression (1) or ILS (0)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Initialize and train the model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(clf, model_output)
    print(f"Model saved to {model_output}")

def main():
    # Path to features extracted in the previous step
    feature_file = "features_labeled.csv"  # Make sure to label your features accordingly

    # Output path for the trained model
    model_output = "trained_model.joblib"

    # Train the model
    train_model(feature_file, model_output)

if __name__ == "__main__":
    main()
