import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import shap

def train_introgression_classifier(df, output_dir="analysis_output"):
    """
    Train and evaluate machine learning models to distinguish between
    introgression and ILS based on genetic patterns.
    
    Enhancements:
    - Uses SMOTE for balancing classes.
    - Includes both Gradient Boosting and MLP classifiers in grid search.
    - Evaluates with ROC, Precision-Recall, calibration curves, and permutation importance.
    - Computes SHAP values for model interpretability.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing simulation results.
    output_dir : str
        Directory to save output figures.
        
    Returns:
    --------
    dict
        Dictionary containing trained models and evaluation metrics.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create target variable: True if simulation indicates a pulse introgression event;
    # False if scenario is "continuous_migration" (i.e. treated as ILS-like)
    df['has_introgression'] = ~df['scenario'].isin(['continuous_migration'])
    
    # Select features (make sure these columns exist in your dataframe)
    features = [
        'fst', 'd_stat', 'mean_internal_branch', 'std_internal_branch',
        'topology_concordance', 'private_alleles_A', 'private_alleles_B', 
        'shared_alleles'
    ]
    features = [f for f in features if f in df.columns]
    if len(features) < 2:
        raise ValueError("Not enough features in dataframe for ML classification")
    
    # Prepare data by dropping missing values and extracting target
    X_data = df[features].dropna()
    y_data = df.loc[X_data.index, 'has_introgression']
    if len(X_data) < 20:
        raise ValueError("Not enough complete samples for ML classification")
    
    # Split into training and test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.3, random_state=42, stratify=y_data
    )
    
    # Baseline Random Forest model (for feature importance evaluation)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_train_score = rf_model.score(X_train, y_train)
    rf_test_score = rf_model.score(X_test, y_test)
    
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance for Distinguishing Introgression vs ILS')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=300)
    plt.close()
    
    # Advanced pipeline: scaling + SMOTE + classifier (Grid Search over GBM and MLP)
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE()),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])
    
    param_grid = [
        {
            'classifier': [GradientBoostingClassifier(random_state=42)],
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
        },
        {
            'classifier': [MLPClassifier(random_state=42)],
            'classifier__hidden_layer_sizes': [(50, 25), (100,)],
            'classifier__early_stopping': [True]
        }
    ]
    
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)
    
    # ROC Curve for best model and for each individual feature
    plt.figure(figsize=(12, 10))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Best Model (AUC = {roc_auc:.2f})', linewidth=2)
    
    for feature in features:
        if feature in df.columns:
            # Invert features if necessary (e.g., topology_concordance might be reversed)
            feature_values = -df.loc[X_test.index, feature] if feature == 'topology_concordance' else df.loc[X_test.index, feature]
            fpr_feat, tpr_feat, _ = roc_curve(y_test, feature_values)
            feature_auc = auc(fpr_feat, tpr_feat)
            plt.plot(fpr_feat, tpr_feat, linestyle='--', label=f'{feature} (AUC = {feature_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Introgression Detection')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/roc_comparison.png", dpi=300)
    plt.close()
    
    # Confusion Matrix and Classification Report
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['ILS', 'Introgression'],
                yticklabels=['ILS', 'Introgression'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
    plt.close()
    
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    
    # SHAP values for interpretability


    X_test_processed = best_model['scaler'].transform(X_test)
    if isinstance(best_model['classifier'], GradientBoostingClassifier):
        explainer = shap.TreeExplainer(best_model['classifier'])
        shap_values = explainer.shap_values(X_test_processed)
    elif isinstance(best_model['classifier'], RandomForestClassifier):
        explainer = shap.TreeExplainer(best_model['classifier'])
        shap_values = explainer.shap_values(X_test_processed)
    elif isinstance(best_model['classifier'], MLPClassifier):
        # Option A: Use KernelExplainer
        explainer = shap.KernelExplainer(
            best_model['classifier'].predict_proba,
            X_train_processed[:100],  # small background subset
        )
        shap_values = explainer.shap_values(X_test_processed[:100])  # or entire X_test
    else:
        print("Skipping SHAP because the model is not tree-based.")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_processed, feature_names=features, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary.png", dpi=300)
    plt.close()
    
    # Decision boundaries for the top two features (from feature_importance)
    if len(features) >= 2:
        top_features = feature_importance['feature'].iloc[:2].tolist()
        plt.figure(figsize=(10, 8))
        X_plot = X_data[top_features].copy()
        scaler_local = StandardScaler()
        X_plot_scaled = scaler_local.fit_transform(X_plot)
        
        # Get the best pipeline and the classifier inside it
        best_classifier = best_model['classifier']
        
        # Only proceed if the best classifier is actually GradientBoostingClassifier
        if isinstance(best_classifier, GradientBoostingClassifier):
            # Extract the classifier's hyperparameters
            gb_params = best_classifier.get_params()
            
            # Create a brand-new GradientBoostingClassifier with those parameters
            simple_model = GradientBoostingClassifier(**gb_params)
            simple_model.fit(X_plot_scaled, y_data)
            
            # Plot decision boundaries
            h = 0.02
            x_min, x_max = X_plot_scaled[:, 0].min() - 1, X_plot_scaled[:, 0].max() + 1
            y_min, y_max = X_plot_scaled[:, 1].min() - 1, X_plot_scaled[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            Z = simple_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
            
            plt.contourf(xx, yy, Z, alpha=0.8, cmap='Blues')
            scatter = plt.scatter(
                X_plot_scaled[:, 0], X_plot_scaled[:, 1], 
                c=y_data, edgecolor='k', cmap='RdYlGn'
            )
            plt.legend(*scatter.legend_elements(), title="Class")
            plt.xlabel(top_features[0])
            plt.ylabel(top_features[1])
            plt.title(f'Decision Boundary Using {top_features[0]} and {top_features[1]}')
            plt.savefig(f"{output_dir}/decision_boundary.png", dpi=300)
            plt.close()
        else:
            # If the best model is not a GradientBoostingClassifier (e.g. MLP), 
            # skip plotting or handle it differently:
            print("Skipping decision boundary plot because the best classifier is not GBC.")
    
    # Permutation importance
    perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({
        'feature': features,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    
    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, 's-')
    plt.xlabel("Predicted probability")
    plt.ylabel("True probability")
    plt.title("Calibration Curve")
    plt.savefig(f"{output_dir}/calibration_curve.png", dpi=300)
    plt.close()
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(f"{output_dir}/pr_curve.png", dpi=300)
    plt.close()
    
    return {
        'rf_model': rf_model,
        'best_model': best_model,
        'feature_importance': feature_importance,
        'best_params': grid_search.best_params_,
        'classification_report': report_df,
        'permutation_importance': perm_df,
        'calibration_data': (prob_true, prob_pred)
    }

# Example usage:
# df = pd.read_csv('simulation_results.csv')
# results = train_introgression_classifier(df)
