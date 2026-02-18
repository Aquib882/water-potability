"""
Water Potability Prediction - Complete ML Pipeline
Author: Water Quality ML Project
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc, classification_report)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.model_selection import RandomizedSearchCV

PLOTS_DIR = "plots"
MODELS_DIR = "models"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("WATER POTABILITY PREDICTION - ML PIPELINE")
print("=" * 60)

df = pd.read_csv("water_potability.csv")
print(f"\n[1] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())

# ─────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────
print("\n[2] Exploratory Data Analysis")
print(df.describe().round(2))
print("\nMissing values:")
print(df.isnull().sum())
print(f"\nClass distribution:\n{df['Potability'].value_counts()}")

# Plot 1: Class distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['Potability'].value_counts().plot(kind='bar', ax=axes[0], color=['#2196F3','#F44336'], edgecolor='black')
axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Potability (0=Not Safe, 1=Safe)')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=0)
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height())}', (p.get_x()+p.get_width()/2, p.get_height()),
                     ha='center', va='bottom')

pct = df['Potability'].value_counts(normalize=True)*100
axes[1].pie(pct, labels=['Not Potable','Potable'], autopct='%1.1f%%',
            colors=['#2196F3','#4CAF50'], startangle=90)
axes[1].set_title('Class Balance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/1_class_distribution.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Feature distributions
features = [c for c in df.columns if c != 'Potability']
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, col in enumerate(features):
    df[col].hist(ax=axes[i], bins=30, color='#1976D2', alpha=0.7, edgecolor='white')
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/2_feature_distributions.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Correlation heatmap
fig, ax = plt.subplots(figsize=(11, 9))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/3_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 4: Boxplots by potability
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, col in enumerate(features):
    df.boxplot(column=col, by='Potability', ax=axes[i], 
               boxprops=dict(color='#1976D2'), whiskerprops=dict(color='#1976D2'),
               medianprops=dict(color='#F44336', linewidth=2))
    axes[i].set_title(col, fontsize=10, fontweight='bold')
    axes[i].set_xlabel('Potability')
plt.suptitle('Feature Distributions by Potability Class', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/4_boxplots_by_class.png", dpi=150, bbox_inches='tight')
plt.close()
print("  EDA plots saved.")

# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
print("\n[3] Preprocessing")

# Fill missing values with median (grouped by potability)
for col in features:
    df[col] = df[col].fillna(df.groupby('Potability')[col].transform('median'))
print(f"  Missing values after imputation: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# 4. HANDLE CLASS IMBALANCE (Oversampling minority)
# ─────────────────────────────────────────────
print("\n[4] Handling Class Imbalance via Oversampling")
df_majority = df[df.Potability == 0]
df_minority = df[df.Potability == 1]
df_minority_upsampled = resample(df_minority, replace=True,
                                  n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"  Balanced class distribution:\n{df_balanced['Potability'].value_counts()}")

# ─────────────────────────────────────────────
# 5. FEATURE ENGINEERING & SPLIT
# ─────────────────────────────────────────────
print("\n[5] Feature Engineering & Train/Test Split")

X = df_balanced[features]
y = df_balanced['Potability']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
print(f"  Train size: {X_train_sc.shape[0]}, Test size: {X_test_sc.shape[0]}")

# ─────────────────────────────────────────────
# 6. MODEL TRAINING WITH CROSS VALIDATION
# ─────────────────────────────────────────────
print("\n[6] Model Training & Cross Validation")

models = {
    "SVC": SVC(kernel='rbf', probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train_sc, y_train, cv=cv, scoring='f1', n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:<22} CV F1: {scores.mean():.4f} ± {scores.std():.4f}")

# Plot 5: CV comparison
fig, ax = plt.subplots(figsize=(10, 6))
data_plot = [cv_results[m] for m in models]
bp = ax.boxplot(data_plot, labels=list(models.keys()), patch_artist=True,
                medianprops=dict(color='black', linewidth=2))
colors = ['#1976D2', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title('5-Fold Cross Validation F1 Scores', fontsize=14, fontweight='bold')
ax.set_ylabel('F1 Score')
ax.set_xticklabels(list(models.keys()), rotation=15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/5_cv_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# 7. HYPERPARAMETER TUNING (Best models)
# ─────────────────────────────────────────────
print("\n[7] Hyperparameter Tuning")

# Tune Random Forest
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params, n_iter=20, cv=5, scoring='f1', random_state=42, n_jobs=-1)
rf_search.fit(X_train_sc, y_train)
best_rf = rf_search.best_estimator_
print(f"  Best RF params: {rf_search.best_params_}")

# Tune SVC
svc_params = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.1, 0.01]}
svc_search = RandomizedSearchCV(
    SVC(kernel='rbf', probability=True, random_state=42),
    svc_params, n_iter=10, cv=5, scoring='f1', random_state=42, n_jobs=-1)
svc_search.fit(X_train_sc, y_train)
best_svc = svc_search.best_estimator_
print(f"  Best SVC params: {svc_search.best_params_}")

# Tune GradientBoosting
gb_params = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1, 0.2],
             'max_depth': [3, 5, 7]}
gb_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_params, n_iter=10, cv=5, scoring='f1', random_state=42, n_jobs=-1)
gb_search.fit(X_train_sc, y_train)
best_gb = gb_search.best_estimator_
print(f"  Best GB params: {gb_search.best_params_}")

tuned_models = {
    "SVC (Tuned)": best_svc,
    "Random Forest (Tuned)": best_rf,
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting (Tuned)": best_gb
}

# Train all
for name, model in tuned_models.items():
    model.fit(X_train_sc, y_train)

# ─────────────────────────────────────────────
# 8. EVALUATION
# ─────────────────────────────────────────────
print("\n[8] Model Evaluation")

metrics_data = []
for name, model in tuned_models.items():
    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    metrics_data.append({'Model': name, 'Accuracy': acc, 'Precision': prec,
                         'Recall': rec, 'F1-Score': f1})
    print(f"\n  {name}:")
    print(f"    Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")

metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv(f"{MODELS_DIR}/model_metrics.csv", index=False)

# Plot 6: Metrics comparison
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(metrics_df))
w = 0.2
colors2 = ['#1976D2', '#4CAF50', '#FF9800', '#E91E63']
for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1-Score']):
    ax.bar(x + i*w, metrics_df[metric], w, label=metric, color=colors2[i], alpha=0.85)
ax.set_xticks(x + 1.5*w)
ax.set_xticklabels(metrics_df['Model'], rotation=20, ha='right')
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/6_model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 7: Confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, (name, model) in enumerate(tuned_models.items()):
    y_pred = model.predict(X_test_sc)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Not Potable','Potable'], yticklabels=['Not Potable','Potable'])
    axes[i].set_title(f'{name}', fontsize=11, fontweight='bold')
    axes[i].set_ylabel('Actual')
    axes[i].set_xlabel('Predicted')
axes[5].axis('off')
plt.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/7_confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 8: ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))
roc_colors = ['#1976D2', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']
for (name, model), color in zip(tuned_models.items(), roc_colors):
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={roc_auc:.3f})')
ax.plot([0,1],[0,1],'k--', lw=1.5, label='Random Classifier')
ax.set_xlim([0,1])
ax.set_ylim([0,1.02])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/8_roc_curves.png", dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# 9. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\n[9] Feature Importance")

# Random Forest feature importance
fi = best_rf.feature_importances_
fi_df = pd.DataFrame({'Feature': features, 'Importance': fi}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(fi_df['Feature'], fi_df['Importance'],
               color=plt.cm.RdYlGn(fi_df['Importance']/fi_df['Importance'].max()), edgecolor='black', linewidth=0.5)
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
for bar, val in zip(bars, fi_df['Importance']):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/9_feature_importance_rf.png", dpi=150, bbox_inches='tight')
plt.close()

# Gradient Boosting feature importance
fi_gb = best_gb.feature_importances_
fi_gb_df = pd.DataFrame({'Feature': features, 'Importance': fi_gb}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(fi_gb_df['Feature'], fi_gb_df['Importance'],
        color='#5C6BC0', edgecolor='black', linewidth=0.5, alpha=0.8)
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Feature Importance - Gradient Boosting', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/10_feature_importance_gb.png", dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# 10. SAVE BEST MODEL
# ─────────────────────────────────────────────
best_model_name = metrics_df.loc[metrics_df['F1-Score'].idxmax(), 'Model']
best_model = list(tuned_models.values())[metrics_df['F1-Score'].idxmax()]
joblib.dump(best_model, f"{MODELS_DIR}/best_model.pkl")
joblib.dump(best_rf, f"{MODELS_DIR}/random_forest.pkl")
print(f"\n[10] Best model: {best_model_name} — saved to models/")

# Save metrics summary
print("\n" + "="*60)
print("FINAL METRICS SUMMARY")
print("="*60)
print(metrics_df.to_string(index=False))
print(f"\nBest model: {best_model_name}")
print("\nAll plots saved in plots/ directory")
print("Pipeline complete!")

# Save feature importance data for PowerBI
fi_export = fi_df.sort_values('Importance', ascending=False)
fi_export.to_csv(f"{MODELS_DIR}/feature_importance.csv", index=False)
metrics_df.to_csv(f"{MODELS_DIR}/final_metrics.csv", index=False)

# Save predictions for PowerBI
y_pred_all = best_model.predict(X_test_sc)
y_prob_all = best_model.predict_proba(X_test_sc)[:, 1]
pred_df = X_test.copy()
pred_df['Actual'] = y_test.values
pred_df['Predicted'] = y_pred_all
pred_df['Probability_Potable'] = y_prob_all
pred_df.to_csv(f"{MODELS_DIR}/test_predictions.csv", index=False)

# Save full balanced dataset for PowerBI
df_balanced.to_csv(f"{MODELS_DIR}/balanced_dataset.csv", index=False)
print("CSV exports for PowerBI saved.")
