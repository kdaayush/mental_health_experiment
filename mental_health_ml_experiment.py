import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Fix Windows cp1252 encoding for Unicode box-drawing characters
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
DATA_PATH = 'e:/bafdm research/data/college_student_iot_mental_health_dataset.csv'
OUTPUT_DIR = 'e:/bafdm research/output'
TARGET_COL = 'mental_health_risk'
CLASS_ORDER = ['Low', 'Moderate', 'High', 'Critical']

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)

print("=" * 80)
print("MACHINE LEARNING BASED MENTAL HEALTH RISK PREDICTION")
print("AMONG COLLEGE STUDENTS")
print("80:10:10 TRAIN / VALIDATION / TEST SPLIT EXPERIMENT")
print("=" * 80)


print("\n" + "=" * 80)
print("SECTION 1: DATASET INSPECTION")
print("=" * 80)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset Shape: {df.shape}")
print(f"Number of records: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

print("\n--- Column Names and Data Types ---")
for col in df.columns:
    print(f"  {col:<35} {str(df[col].dtype):<12}")

print("\n--- Missing Values ---")
missing = df.isnull().sum()
total_missing = missing.sum()
print(f"Total missing values: {total_missing}")
if total_missing > 0:
    print(missing[missing > 0])
else:
    print("No missing values in the dataset.")

print(f"\n--- Duplicate Rows ---")
n_duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {n_duplicates}")

print("\n--- Descriptive Statistics ---")
print(df.describe().round(2).to_string())

# Identify column types
id_cols = ['student_id', 'session_id']
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [c for c in categorical_cols if c != TARGET_COL]
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols = [c for c in numerical_cols if c not in id_cols]

print(f"\n--- Column Classification ---")
print(f"ID columns (to drop): {id_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")
print(f"Target column: {TARGET_COL}")

print(f"\n--- Categorical Feature Values ---")
for col in categorical_cols:
    print(f"  {col}: {df[col].unique().tolist()}")

print(f"\n--- Target Variable: {TARGET_COL} ---")
target_dist = df[TARGET_COL].value_counts()
print(target_dist)
print(f"\nTarget class proportions:")
print((target_dist / len(df) * 100).round(2).to_string())

print("\n" + "=" * 80)
print("SECTION 2: DATA PREPROCESSING")
print("=" * 80)

# Remove duplicates if any
if n_duplicates > 0:
    df = df.drop_duplicates()
    print(f"Removed {n_duplicates} duplicate rows. New shape: {df.shape}")
else:
    print("No duplicate rows to remove.")

# Drop ID columns (not predictive features)
df = df.drop(columns=id_cols)
print(f"Dropped ID columns: {id_cols}")
print(f"Shape after dropping IDs: {df.shape}")

# Separate features and target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Encode target variable
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(CLASS_ORDER)
y_encoded = label_encoder.transform(y)

print(f"\nTarget encoding: {dict(zip(CLASS_ORDER, label_encoder.transform(CLASS_ORDER)))}")
print(f"Features shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")

# Define preprocessing
print("\n--- Preprocessing Pipeline ---")
print(f"Categorical features: {categorical_cols} -> OrdinalEncoder")
print(f"Numerical features ({len(numerical_cols)} features) -> StandardScaler")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
    ],
    remainder='drop'
)

print("\n" + "=" * 80)
print("SECTION 3: TRAIN / VALIDATION / TEST SPLIT (80:10:10)")
print("=" * 80)

# Step 1: Split into 80% training and 20% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.20, random_state=RANDOM_STATE, stratify=y_encoded
)

# Step 2: Split the 20% temporary into 50/50 -> 10% validation, 10% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)

total_samples = len(y_encoded)
print(f"\nTotal dataset size: {total_samples}")
print(f"\nTraining set size:    {X_train.shape[0]}  ({X_train.shape[0]/total_samples*100:.1f}%)")
print(f"Validation set size:  {X_val.shape[0]}  ({X_val.shape[0]/total_samples*100:.1f}%)")
print(f"Test set size:        {X_test.shape[0]}  ({X_test.shape[0]/total_samples*100:.1f}%)")
print(f"Sum:                  {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]}")
print(f"\nrandom_state: {RANDOM_STATE}")
print(f"Stratified: Yes (based on {TARGET_COL})")

print(f"\nTraining set class distribution:")
train_dist = pd.Series(y_train).value_counts().sort_index()
for idx, count in train_dist.items():
    print(f"  {CLASS_ORDER[idx]}: {count} ({count/len(y_train)*100:.2f}%)")

print(f"\nValidation set class distribution:")
val_dist = pd.Series(y_val).value_counts().sort_index()
for idx, count in val_dist.items():
    print(f"  {CLASS_ORDER[idx]}: {count} ({count/len(y_val)*100:.2f}%)")

print(f"\nTest set class distribution:")
test_dist = pd.Series(y_test).value_counts().sort_index()
for idx, count in test_dist.items():
    print(f"  {CLASS_ORDER[idx]}: {count} ({count/len(y_test)*100:.2f}%)")

# Verify approximate proportions
print(f"\n--- Split Verification ---")
print(f"  Train: {X_train.shape[0]/total_samples*100:.2f}%  (target: 80%)")
print(f"  Val:   {X_val.shape[0]/total_samples*100:.2f}%  (target: 10%)")
print(f"  Test:  {X_test.shape[0]/total_samples*100:.2f}%  (target: 10%)")


print("\n" + "=" * 80)
print("SECTION 4: CLASS IMBALANCE HANDLING")
print("=" * 80)

print("\nClass distribution analysis:")
print(f"  Most common class: Critical ({target_dist['Critical']} samples, {target_dist['Critical']/len(df)*100:.1f}%)")
print(f"  Least common class: Low ({target_dist['Low']} samples, {target_dist['Low']/len(df)*100:.1f}%)")
print(f"  Imbalance ratio (max/min): {target_dist.max() / target_dist.min():.1f}:1")
print(f"\nThe dataset is SIGNIFICANTLY IMBALANCED.")
print(f"\nImbalance handling strategy:")
print(f"  1. SMOTE (Synthetic Minority Over-sampling Technique) on TRAINING data ONLY")
print(f"  2. class_weight='balanced' for models that support it")
print(f"  Note: Validation and test sets are NOT modified in any way.")

# Fit the preprocessor on training data ONLY
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

print(f"\nPreprocessed training set shape:   {X_train_processed.shape}")
print(f"Preprocessed validation set shape: {X_val_processed.shape}")
print(f"Preprocessed test set shape:       {X_test_processed.shape}")
print(f"\nNote: Preprocessor fit ONLY on training data. Transformations applied to val/test.")

# Apply SMOTE to training data only
smote = SMOTE(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

print(f"\nAfter SMOTE on training data:")
print(f"  Resampled training set shape: {X_train_resampled.shape}")
resampled_dist = pd.Series(y_train_resampled).value_counts().sort_index()
for idx, count in resampled_dist.items():
    print(f"  {CLASS_ORDER[idx]}: {count}")
print(f"\nSMOTE NOT applied to validation set (shape: {X_val_processed.shape})")
print(f"SMOTE NOT applied to test set (shape: {X_test_processed.shape})")

print("\n" + "=" * 80)
print("SECTION 5: MODEL TRAINING")
print("=" * 80)

models = {
    'Logistic Regression': LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=5000,
        solver='lbfgs',
        class_weight='balanced'
    ),
    'Decision Tree': DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    ),
    'SVM': SVC(
        random_state=RANDOM_STATE,
        kernel='rbf',
        class_weight='balanced',
        decision_function_shape='ovr'
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=7,
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        eval_metric='mlogloss',
        verbosity=0
    )
}

print("\nModel configurations:")
for name, model in models.items():
    print(f"\n  {name}:")
    params = model.get_params()
    important_params = {k: v for k, v in params.items()
                       if k in ['random_state', 'n_estimators', 'max_iter', 'kernel',
                               'class_weight', 'n_neighbors', 'solver', 'multi_class',
                               'eval_metric', 'n_jobs']}
    for k, v in important_params.items():
        print(f"    {k}: {v}")

# Train all models on SMOTE-resampled training data
trained_models = {}
val_predictions = {}
test_predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...", end=" ", flush=True)
    model.fit(X_train_resampled, y_train_resampled)
    trained_models[name] = model

    # Predict on validation set (for model comparison and selection)
    y_val_pred = model.predict(X_val_processed)
    val_predictions[name] = y_val_pred

    # Predict on test set (stored but NOT used until final evaluation)
    y_test_pred = model.predict(X_test_processed)
    test_predictions[name] = y_test_pred
    print("Done.")

print("\nAll models trained successfully.")



print("\n" + "=" * 80)
print("SECTION 6: VALIDATION SET EVALUATION (MODEL COMPARISON)")
print("=" * 80)

val_results = {}
val_classification_reports_text = ""

for name, y_val_pred in val_predictions.items():
    acc = accuracy_score(y_val, y_val_pred)
    macro_p = precision_score(y_val, y_val_pred, average='macro', zero_division=0)
    macro_r = recall_score(y_val, y_val_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
    weighted_p = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
    weighted_r = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)

    val_results[name] = {
        'Accuracy': acc,
        'Macro Precision': macro_p,
        'Macro Recall': macro_r,
        'Macro F1': macro_f1,
        'Weighted Precision': weighted_p,
        'Weighted Recall': weighted_r,
        'Weighted F1': weighted_f1,
    }

    report = classification_report(
        y_val, y_val_pred,
        target_names=CLASS_ORDER,
        zero_division=0
    )

    print(f"\n{'─' * 60}")
    print(f"  {name} (VALIDATION SET)")
    print(f"{'─' * 60}")
    print(f"  Accuracy:           {acc:.4f}")
    print(f"  Macro Precision:    {macro_p:.4f}")
    print(f"  Macro Recall:       {macro_r:.4f}")
    print(f"  Macro F1:           {macro_f1:.4f}")
    print(f"  Weighted Precision: {weighted_p:.4f}")
    print(f"  Weighted Recall:    {weighted_r:.4f}")
    print(f"  Weighted F1:        {weighted_f1:.4f}")
    print(f"\n  Classification Report (Validation):")
    print(report)

    val_classification_reports_text += f"\n{'=' * 60}\n"
    val_classification_reports_text += f"{name} — VALIDATION SET\n"
    val_classification_reports_text += f"{'=' * 60}\n"
    val_classification_reports_text += f"Accuracy: {acc:.4f}\n"
    val_classification_reports_text += f"Macro Precision: {macro_p:.4f}\n"
    val_classification_reports_text += f"Macro Recall: {macro_r:.4f}\n"
    val_classification_reports_text += f"Macro F1: {macro_f1:.4f}\n"
    val_classification_reports_text += f"Weighted Precision: {weighted_p:.4f}\n"
    val_classification_reports_text += f"Weighted Recall: {weighted_r:.4f}\n"
    val_classification_reports_text += f"Weighted F1: {weighted_f1:.4f}\n"
    val_classification_reports_text += f"\nClassification Report:\n{report}\n"


print("\n" + "=" * 80)
print("SECTION 7: VALIDATION CONFUSION MATRICES")
print("=" * 80)

fig_size = (8, 6)

for name, y_val_pred in val_predictions.items():
    cm = confusion_matrix(y_val, y_val_pred)

    fig, ax = plt.subplots(figsize=fig_size)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
                ax=ax, linewidths=0.5, linecolor='gray')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Validation Confusion Matrix - {name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Sanitize filename
    fname = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    save_path = os.path.join(OUTPUT_DIR, f'validation_confusion_matrix_{fname}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


print("\n" + "=" * 80)
print("SECTION 8: VALIDATION MODEL COMPARISON")
print("=" * 80)

val_comparison_df = pd.DataFrame(val_results).T
val_comparison_df.index.name = 'Model'
val_comparison_df = val_comparison_df.round(4)

print("\n--- Validation Model Comparison Table ---")
print(val_comparison_df.to_string())

# Save validation_model_comparison.csv
val_comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'validation_model_comparison.csv'))
print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'validation_model_comparison.csv')}")

# Create bar chart for Validation Macro F1
fig, ax = plt.subplots(figsize=(10, 6))
model_names = list(val_results.keys())
macro_f1_scores = [val_results[m]['Macro F1'] for m in model_names]

colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#00BCD4']
bars = ax.bar(model_names, macro_f1_scores, color=colors, edgecolor='black', linewidth=0.5)

# Add value labels on bars
for bar, score in zip(bars, macro_f1_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Macro F1-Score', fontsize=12, fontweight='bold')
ax.set_title('Validation Model Comparison - Macro F1-Score (80:10:10 Split)', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(macro_f1_scores) * 1.15)
ax.tick_params(axis='x', rotation=20)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'validation_model_comparison.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {os.path.join(OUTPUT_DIR, 'validation_model_comparison.png')}")


print("\n" + "=" * 80)
print("SECTION 9: FINAL MODEL SELECTION (BASED ON VALIDATION ONLY)")
print("=" * 80)

# Rank models by multiple criteria using VALIDATION results
print("\n--- Model Rankings (VALIDATION SET) ---")
ranking_metrics = ['Macro F1', 'Macro Recall', 'Weighted F1', 'Accuracy']
ranking_df = pd.DataFrame()

for metric in ranking_metrics:
    ranking_df[metric] = val_comparison_df[metric].rank(ascending=False)

ranking_df['Average Rank'] = ranking_df.mean(axis=1)
ranking_df = ranking_df.sort_values('Average Rank')

print(ranking_df.to_string())

# Select best model based on VALIDATION performance
best_model_name = ranking_df.index[0]
best_model = trained_models[best_model_name]

print(f"\n--- Selected Final Model: {best_model_name} ---")
print(f"\nSelection rationale:")
print(f"  The model was selected based on a composite ranking of VALIDATION metrics:")
print(f"  Macro F1, Macro Recall, Weighted F1, and Accuracy.")
print(f"  {best_model_name} achieved the best average rank ({ranking_df['Average Rank'].iloc[0]:.2f}).")
print(f"\n  Validation Performance of Selected Model:")
for metric, value in val_results[best_model_name].items():
    print(f"    {metric}: {value:.4f}")

print(f"\n  NOTE: The TEST SET was NOT used for model selection.")
print(f"  The test set will now be used ONCE for final unbiased evaluation.")


print("\n" + "=" * 80)
print("SECTION 10: FINAL TEST SET EVALUATION")
print("=" * 80)

# Use ONLY the selected best model on the test set
best_test_pred = test_predictions[best_model_name]

test_acc = accuracy_score(y_test, best_test_pred)
test_macro_p = precision_score(y_test, best_test_pred, average='macro', zero_division=0)
test_macro_r = recall_score(y_test, best_test_pred, average='macro', zero_division=0)
test_macro_f1 = f1_score(y_test, best_test_pred, average='macro', zero_division=0)
test_weighted_p = precision_score(y_test, best_test_pred, average='weighted', zero_division=0)
test_weighted_r = recall_score(y_test, best_test_pred, average='weighted', zero_division=0)
test_weighted_f1 = f1_score(y_test, best_test_pred, average='weighted', zero_division=0)

test_metrics = {
    'Accuracy': test_acc,
    'Macro Precision': test_macro_p,
    'Macro Recall': test_macro_r,
    'Macro F1': test_macro_f1,
    'Weighted Precision': test_weighted_p,
    'Weighted Recall': test_weighted_r,
    'Weighted F1': test_weighted_f1,
}

print(f"\n{'=' * 60}")
print(f"  FINAL TEST SET RESULTS — {best_model_name}")
print(f"{'=' * 60}")
print(f"  Accuracy:           {test_acc:.4f}")
print(f"  Macro Precision:    {test_macro_p:.4f}")
print(f"  Macro Recall:       {test_macro_r:.4f}")
print(f"  Macro F1:           {test_macro_f1:.4f}")
print(f"  Weighted Precision: {test_weighted_p:.4f}")
print(f"  Weighted Recall:    {test_weighted_r:.4f}")
print(f"  Weighted F1:        {test_weighted_f1:.4f}")

test_report = classification_report(
    y_test, best_test_pred,
    target_names=CLASS_ORDER,
    zero_division=0
)
print(f"\n  Final Test Classification Report:")
print(test_report)

# Final test confusion matrix
cm_test = confusion_matrix(y_test, best_test_pred)
print("  Final Test Confusion Matrix:")
print(cm_test)

# Save final test confusion matrix plot
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
            ax=ax, linewidths=1, linecolor='white',
            annot_kws={'size': 14, 'fontweight': 'bold'})
ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_title(f'Final Test Confusion Matrix - {best_model_name}\n(80:10:10 Split)', fontsize=16, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'final_test_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'final_test_confusion_matrix.png')}")

# Save final_test_classification_report.txt
test_report_path = os.path.join(OUTPUT_DIR, 'final_test_classification_report.txt')
with open(test_report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("FINAL TEST SET CLASSIFICATION REPORT\n")
    f.write(f"Selected Model: {best_model_name}\n")
    f.write("Split: 80% Train / 10% Validation / 10% Test\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Accuracy:           {test_acc:.4f}\n")
    f.write(f"Macro Precision:    {test_macro_p:.4f}\n")
    f.write(f"Macro Recall:       {test_macro_r:.4f}\n")
    f.write(f"Macro F1:           {test_macro_f1:.4f}\n")
    f.write(f"Weighted Precision: {test_weighted_p:.4f}\n")
    f.write(f"Weighted Recall:    {test_weighted_r:.4f}\n")
    f.write(f"Weighted F1:        {test_weighted_f1:.4f}\n")
    f.write(f"\nClassification Report:\n{test_report}\n")
    f.write(f"\nConfusion Matrix:\n{cm_test}\n")
print(f"Saved: {test_report_path}")

# Save final_test_results.csv
test_results_df = pd.DataFrame({
    'Metric': list(test_metrics.keys()),
    'Test Score': list(test_metrics.values())
})
test_results_df['Test Score'] = test_results_df['Test Score'].round(4)
test_results_df.to_csv(os.path.join(OUTPUT_DIR, 'final_test_results.csv'), index=False)
print(f"Saved: {os.path.join(OUTPUT_DIR, 'final_test_results.csv')}")


print("\n" + "=" * 80)
print("SECTION 11: FEATURE IMPORTANCE")
print("=" * 80)

feature_names = numerical_cols + categorical_cols

# Determine feature importance method
if hasattr(best_model, 'feature_importances_'):
    print(f"\nUsing native feature importance from {best_model_name}.")
    importances = best_model.feature_importances_
    importance_method = "Native (Gini Importance)" if 'Forest' in best_model_name or 'Tree' in best_model_name else "Native (Gain)"
else:
    print(f"\n{best_model_name} does not have native feature importance.")
    print("Using Permutation Importance instead.")
    print("Note: Permutation importance computed on VALIDATION set (NOT test set).")
    from sklearn.inspection import permutation_importance
    perm_result = permutation_importance(
        best_model, X_val_processed, y_val,
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    importances = perm_result.importances_mean
    importance_method = "Permutation Importance (on Validation Set)"

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print(f"\nFeature Importance Method: {importance_method}")
print(f"\n--- Top 20 Features ---")
print(importance_df.head(20).to_string(index=False))

# Save feature_importance.csv
importance_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)
print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'feature_importance.csv')}")

# Plot feature importance (top 20)
top_n = min(20, len(importance_df))
top_features = importance_df.head(top_n)

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(
    range(top_n),
    top_features['Importance'].values[::-1],
    color=plt.cm.viridis(np.linspace(0.2, 0.9, top_n)),
    edgecolor='black', linewidth=0.3
)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features['Feature'].values[::-1], fontsize=10)
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_title(f'Top {top_n} Feature Importance ({best_model_name})\nMethod: {importance_method}',
             fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {os.path.join(OUTPUT_DIR, 'feature_importance.png')}")

print("\n" + "=" * 80)
print("SECTION 12: SAVING ALL RESULTS")
print("=" * 80)

# Save validation classification reports
val_report_path = os.path.join(OUTPUT_DIR, 'validation_classification_reports.txt')
with open(val_report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("VALIDATION SET CLASSIFICATION REPORTS - ALL MODELS\n")
    f.write("Machine Learning Based Mental Health Risk Prediction\n")
    f.write("Among College Students\n")
    f.write("=" * 80 + "\n")
    f.write(f"\nDataset: College Student Mental Health IoT Dataset\n")
    f.write(f"Target: {TARGET_COL}\n")
    f.write(f"Classes: {CLASS_ORDER}\n")
    f.write(f"Split: 80% Train / 10% Validation / 10% Test\n")
    f.write(f"random_state: {RANDOM_STATE}\n")
    f.write(f"Imbalance Handling: SMOTE (training only) + class_weight='balanced'\n")
    f.write(val_classification_reports_text)
print(f"Saved: {val_report_path}")

# List all output files
print(f"\n--- Output Files ---")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(fpath)
    print(f"  {f:<55} {size:>10,} bytes")


print("\n" + "=" * 80)
print("SECTION 13: REPRODUCIBILITY")
print("=" * 80)

print(f"\n--- Reproducibility Parameters ---")
print(f"  random_state: {RANDOM_STATE}")
print(f"  Split: 80% Train / 10% Validation / 10% Test")
print(f"  Stratification: Yes (target variable)")
print(f"  SMOTE: Applied to training data only (NOT val/test)")
print(f"  Preprocessing: StandardScaler (numerical) + OrdinalEncoder (categorical)")
print(f"  Preprocessing fitted on: Training data ONLY")
print(f"  Pipeline: sklearn ColumnTransformer")
print(f"  ID columns dropped: {id_cols}")

print(f"\n--- Python Package Versions ---")
import sklearn
import xgboost
import imblearn
print(f"  scikit-learn: {sklearn.__version__}")
print(f"  xgboost: {xgboost.__version__}")
print(f"  imbalanced-learn: {imblearn.__version__}")
print(f"  pandas: {pd.__version__}")
print(f"  numpy: {np.__version__}")


print("\n" + "=" * 80)
print("SECTION 14: FINAL EXPERIMENTAL REPORT")
print("=" * 80)

print(f"""
┌─────────────────────────────────────────────────────────────────┐
│     FINAL EXPERIMENTAL REPORT — 80:10:10 SPLIT                 │
├─────────────────────────────────────────────────────────────────┤
│ 1. Dataset Size:           {df.shape[0]:>6} records, {df.shape[1]:>3} columns          │
│ 2. Number of Features:     {X.shape[1]:>6}                                  │
│ 3. Target Variable:        {TARGET_COL:<35}│
│ 4. Target Classes:         {', '.join(CLASS_ORDER):<35}│
│ 5. Training Set:           {X_train.shape[0]:>6} samples ({X_train.shape[0]/total_samples*100:.1f}%)                     │
│ 6. Validation Set:         {X_val.shape[0]:>6} samples ({X_val.shape[0]/total_samples*100:.1f}%)                     │
│ 7. Test Set:               {X_test.shape[0]:>6} samples ({X_test.shape[0]/total_samples*100:.1f}%)                     │
└─────────────────────────────────────────────────────────────────┘
""")

print("8. Target Class Distribution (Original):")
for cls in CLASS_ORDER:
    count = target_dist.get(cls, 0)
    pct = count / len(df) * 100
    print(f"   {cls:<12} {count:>6} ({pct:>5.1f}%)")

print(f"\n9. Class Distribution per Split:")
print(f"\n   Training Set:")
for idx, count in train_dist.items():
    print(f"     {CLASS_ORDER[idx]:<12} {count:>6} ({count/len(y_train)*100:.2f}%)")
print(f"\n   Validation Set:")
for idx, count in val_dist.items():
    print(f"     {CLASS_ORDER[idx]:<12} {count:>6} ({count/len(y_val)*100:.2f}%)")
print(f"\n   Test Set:")
for idx, count in test_dist.items():
    print(f"     {CLASS_ORDER[idx]:<12} {count:>6} ({count/len(y_test)*100:.2f}%)")

print(f"\n10. Preprocessing:")
print(f"    - Dropped ID columns: student_id, session_id")
print(f"    - StandardScaler applied to {len(numerical_cols)} numerical features")
print(f"    - OrdinalEncoder applied to {len(categorical_cols)} categorical features")
print(f"    - Preprocessing fitted on TRAINING DATA ONLY")
print(f"    - No missing values or duplicates to handle")

print(f"\n11. Class Imbalance Handling:")
print(f"    - SMOTE applied to training data only (NOT val/test)")
print(f"    - class_weight='balanced' used for LR, DT, RF, SVM")
print(f"    - Training set after SMOTE: {X_train_resampled.shape[0]} samples (balanced)")

print(f"\n12. Models Trained: {len(models)}")
for name in models:
    print(f"    - {name}")

print(f"\n13. VALIDATION Model Comparison Table:")
print("─" * 95)
header = f"{'Model':<25} {'Accuracy':>9} {'Macro P':>9} {'Macro R':>9} {'Macro F1':>9} {'Wt. F1':>9}"
print(header)
print("─" * 95)
for name in model_names:
    r = val_results[name]
    print(f"{name:<25} {r['Accuracy']:>9.4f} {r['Macro Precision']:>9.4f} {r['Macro Recall']:>9.4f} {r['Macro F1']:>9.4f} {r['Weighted F1']:>9.4f}")
print("─" * 95)

print(f"\n14. Selected Final Model: {best_model_name}")
print(f"    Selection based on: VALIDATION performance (composite ranking)")
print(f"    Average rank: {ranking_df['Average Rank'].iloc[0]:.2f}")

print(f"\n    Validation Performance:")
for metric, value in val_results[best_model_name].items():
    print(f"      {metric}: {value:.4f}")

print(f"\n15. FINAL TEST SET RESULTS ({best_model_name}):")
print("─" * 50)
for metric, value in test_metrics.items():
    print(f"    {metric:<25} {value:.4f}")
print("─" * 50)

print(f"\n16. Final Test Confusion Matrix Observations ({best_model_name}):")
for i, cls in enumerate(CLASS_ORDER):
    tp = cm_test[i, i]
    total = cm_test[i, :].sum()
    recall = tp / total if total > 0 else 0
    print(f"    {cls:<12}: {tp}/{total} correctly classified (Recall: {recall:.4f})")

print(f"\n17. Top 10 Important Features:")
for i, row in importance_df.head(10).iterrows():
    print(f"    {i+1:>2}. {row['Feature']:<35} {row['Importance']:.4f}")

print(f"\n18. Limitations:")
print(f"    - The dataset is significantly imbalanced (Critical: {target_dist['Critical']/len(df)*100:.1f}%, Low: {target_dist['Low']/len(df)*100:.1f}%)")
print(f"    - SMOTE was used to balance training data, which generates synthetic samples")
print(f"    - No hyperparameter tuning (e.g., GridSearchCV) was performed")
print(f"    - Results are from a single 80:10:10 split (no cross-validation)")
print(f"    - Validation set is relatively small (10%); metrics may have higher variance")
print(f"    - The dataset may contain synthetic/simulated data (IoT monitoring)")
print(f"    - Feature interactions and non-linear relationships may not be fully captured")
print(f"    - External validity is limited to the population represented in the dataset")

print("\n" + "=" * 80)
print("FINAL RESULTS — 80:10:10 SPLIT")
print("=" * 80)

print("\nA. VALIDATION MODEL COMPARISON:")
print("─" * 95)
print(header)
print("─" * 95)
for name in model_names:
    r = val_results[name]
    marker = " <<<" if name == best_model_name else ""
    print(f"{name:<25} {r['Accuracy']:>9.4f} {r['Macro Precision']:>9.4f} {r['Macro Recall']:>9.4f} {r['Macro F1']:>9.4f} {r['Weighted F1']:>9.4f}{marker}")
print("─" * 95)

print(f"\nB. SELECTED MODEL: {best_model_name}")
print(f"   (Selected based on validation performance — test set was NOT used for selection)")

print(f"\nC. FINAL TEST-SET METRICS ({best_model_name}):")
print("─" * 50)
for metric, value in test_metrics.items():
    print(f"   {metric:<25} {value:.4f}")
print("─" * 50)

print(f"\nD. TOP 10 IMPORTANT FEATURES:")
for i, row in importance_df.head(10).iterrows():
    print(f"   {i+1:>2}. {row['Feature']:<35} {row['Importance']:.4f}")

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("80:10:10 SPLIT — ALL RESULTS ARE FROM THE NEW EXPERIMENT")
print("=" * 80)
