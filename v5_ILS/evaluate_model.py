# Evaluating the Machine Learning Model
# This script evaluates the trained model on a new dataset or test set.

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib

def evaluate_model(feature_file, model_file):
    """
    Evaluate the trained machine learning model.

    Args:
        feature_file (str): Path to the CSV file with extracted features for evaluation.
        model_file (str): Path to the trained model file.
    """
    # Load features
    df = pd.read_csv(feature_file)

    # Prepare features and labels
    X = df.drop(columns=['window_start', 'window_end', 'label'])
    y_true = df['label']

    # Load the trained model
    clf = joblib.load(model_file)

    # Predict probabilities and classes
    y_pred_prob = clf.predict_proba(X)[:, 1]
    y_pred = clf.predict(X)

    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    plt.show()

def main():
    # Path to features for evaluation
    feature_file = "features_evaluation.csv"

    # Path to the trained model
    model_file = "trained_model.joblib"

    # Evaluate the model
    evaluate_model(feature_file, model_file)

if __name__ == "__main__":
    main()
