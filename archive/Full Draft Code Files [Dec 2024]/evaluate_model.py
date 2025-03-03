# Evaluating the Machine Learning Model
# This script evaluates the trained machine learning model on a new dataset or test set.

import pandas as pd  # For data manipulation and reading CSV files
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
# Metrics for model evaluation, including classification performance and ROC analysis
import matplotlib.pyplot as plt  # For plotting the ROC curve
import joblib  # For loading the trained machine learning model

def evaluate_model(feature_file, model_file):
    """
    Evaluate the trained machine learning model.

    Args:
        feature_file (str): Path to the CSV file with extracted features for evaluation.
        model_file (str): Path to the trained model file.
    """
    # Load the dataset containing extracted features and labels for evaluation
    df = pd.read_csv(feature_file)

    # Separate the features (X) and labels (y_true)
    # 'window_start' and 'window_end' are likely metadata columns, and 'label' is the target variable
    X = df.drop(columns=['window_start', 'window_end', 'label'])  # Features for prediction
    y_true = df['label']  # Ground truth labels for evaluation

    # Load the trained model from the specified file
    clf = joblib.load(model_file)

    # Predict the probabilities of the positive class for ROC analysis
    y_pred_prob = clf.predict_proba(X)[:, 1]
    # Predict the class labels for classification metrics
    y_pred = clf.predict(X)

    # Generate and display a detailed classification report
    # Includes precision, recall, f1-score, and support for each class
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # Generate and display the confusion matrix
    # Provides a summary of prediction results showing true positives, false positives, etc.
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Compute the ROC curve and AUC (Area Under the Curve) score
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')  # ROC curve with AUC in the legend
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')  # Diagonal reference line
    plt.xlabel('False Positive Rate')  # X-axis label
    plt.ylabel('True Positive Rate')  # Y-axis label
    plt.title('Receiver Operating Characteristic')  # Title of the plot
    plt.legend()  # Display the legend
    plt.show()  # Render the plot

def main():
    """
    Main function to evaluate the model using predefined file paths.
    """
    # Path to the CSV file containing features for evaluation
    feature_file = "features_evaluation.csv"

    # Path to the trained model file
    model_file = "trained_model.joblib"

    # Call the evaluation function with the specified feature and model files
    evaluate_model(feature_file, model_file)

if __name__ == "__main__":
    # Execute the main function when the script is run directly
    main()
