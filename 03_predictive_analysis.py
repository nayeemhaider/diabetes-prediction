"""
usage:
    python 03_predictive_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, GridSearchCV)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, precision_recall_curve,
                              average_precision_score, f1_score)
import xgboost as xgb
import shap
import json
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH      = "./dataset/diabetes.csv"
OUT_IMAGE      = "./results/03_res_predictive/predictive_plots.png"
OUT_TEXT       = "./results/03_res_predictive/predictive_results.txt"
OUT_MODEL      = "./results/03_res_predictive/best_model_xgb.json"

ZERO_COLS      = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
FEATURES       = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                  "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
RANDOM_STATE   = 42
 
NEG_COLOR, POS_COLOR = "#4C9BE8", "#E8655A"
BG, TEXT = "#FAFBFC", "#2C3E50"
ACCENT   = "#6C63FF"
 
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.facecolor": BG,
    "figure.facecolor": "white", "axes.edgecolor": "#D0D9E8",
    "axes.linewidth": 0.8, "grid.color": "#F0F4FA",
    "grid.linewidth": 0.7, "text.color": TEXT,
    "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
})
 
# ── Load & preprocess ─────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df[ZERO_COLS] = df[ZERO_COLS].replace(0, np.nan)
 
X_raw = df[FEATURES]
y     = df["Outcome"]
 
# Impute → scale
imputer = KNNImputer(n_neighbors=5)
X_imp   = pd.DataFrame(imputer.fit_transform(X_raw), columns=FEATURES)
 
scaler  = RobustScaler()
X_sc    = scaler.fit_transform(X_imp)
 
X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
 
# ── Model comparison (5-fold stratified CV) ───────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
 
candidates = {
    "Logistic Regression":   LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                                class_weight="balanced"),
    "Random Forest":         RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE,
                                                    class_weight="balanced"),
    "SVM (RBF)":             SVC(kernel="rbf", probability=True, class_weight="balanced",
                                 random_state=RANDOM_STATE),
    "KNN (k=7)":             KNeighborsClassifier(n_neighbors=7),
    "Gradient Boosting":     GradientBoostingClassifier(n_estimators=150,
                                                         random_state=RANDOM_STATE),
    "XGBoost":               xgb.XGBClassifier(n_estimators=150, random_state=RANDOM_STATE,
                                                eval_metric="logloss",
                                                scale_pos_weight=(y==0).sum()/(y==1).sum()),
}
 
cv_results = {}
print("Running 5-fold cross-validation …")
for name, model in candidates.items():
    scores = cross_val_score(model, X_sc, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:<25}  ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")
 
# ── Tune best model: XGBoost ──────────────────────────────────────────────────
print("\nTuning XGBoost with GridSearchCV …")
param_grid = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [3, 4, 6],
    "learning_rate":    [0.05, 0.10, 0.20],
    "subsample":        [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
}
base_xgb = xgb.XGBClassifier(
    random_state=RANDOM_STATE, eval_metric="logloss",
    scale_pos_weight=(y==0).sum()/(y==1).sum())
 
grid_search = GridSearchCV(base_xgb, param_grid, cv=cv,
                           scoring="roc_auc", n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"  Best params : {best_params}")
print(f"  Best CV AUC : {grid_search.best_score_:.4f}")
 
best_xgb = grid_search.best_estimator_
best_xgb.save_model(OUT_MODEL)
print(f"  Model saved → {OUT_MODEL}")
 
# ── Evaluate on held-out test set ─────────────────────────────────────────────
y_pred      = best_xgb.predict(X_test)
y_prob      = best_xgb.predict_proba(X_test)[:, 1]
roc_auc     = roc_auc_score(y_test, y_prob)
avg_prec    = average_precision_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)
prec, rec, thresh_pr = precision_recall_curve(y_test, y_prob)
cm          = confusion_matrix(y_test, y_pred)
report      = classification_report(y_test, y_pred,
                                    target_names=["No Diabetes", "Diabetes"])
 
# ── Threshold search (maximise F1) ───────────────────────────────────────────
thresh_range = np.linspace(0.1, 0.9, 80)
f1_scores    = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresh_range]
best_thresh  = thresh_range[np.argmax(f1_scores)]
y_pred_best  = (y_prob >= best_thresh).astype(int)
 
# ── SHAP values ───────────────────────────────────────────────────────────────
explainer  = shap.TreeExplainer(best_xgb)
shap_vals  = explainer.shap_values(X_test)
shap_df    = pd.DataFrame(np.abs(shap_vals), columns=FEATURES)
mean_shap  = shap_df.mean().sort_values(ascending=False)
 
# ── Text report ───────────────────────────────────────────────────────────────
lines = []
lines.append("=" * 70)
lines.append("PREDICTIVE ANALYSIS — Model Training & Evaluation")
lines.append("=" * 70)
 
lines.append("\n── 5-Fold CV ROC-AUC Comparison ──")
for name, sc in sorted(cv_results.items(), key=lambda x: -x[1].mean()):
    lines.append(f"  {name:<28} {sc.mean():.4f} ± {sc.std():.4f}")
 
lines.append(f"\n── XGBoost GridSearch Best Params ──")
for k, v in best_params.items():
    lines.append(f"  {k:<22}: {v}")
 
lines.append(f"\n── Test-Set Evaluation (threshold = 0.50) ──")
lines.append(f"  ROC-AUC           : {roc_auc:.4f}")
lines.append(f"  Avg Precision     : {avg_prec:.4f}")
lines.append(f"\n{report}")
 
lines.append(f"── Optimal Threshold Analysis ──")
lines.append(f"  Best threshold (max F1) : {best_thresh:.3f}")
lines.append(f"  F1 at best threshold    : {max(f1_scores):.4f}")
lines.append(f"\n{classification_report(y_test, y_pred_best, target_names=['No Diabetes','Diabetes'])}")
 
lines.append("── SHAP Mean |value| Feature Importance ──")
for feat, val in mean_shap.items():
    lines.append(f"  {feat:<30}: {val:.4f}")
 
summary_text = "\n".join(lines)
print("\n" + summary_text)
with open(OUT_TEXT, "w", encoding="utf-8") as f:
    f.write(summary_text)
 
# ── Visualisations ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 26), facecolor="white")
fig.suptitle("Predictive Analysis — XGBoost Classifier",
             fontsize=22, fontweight="bold", color=TEXT, y=0.99)
gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.42)
 
# [0,0-1] CV comparison
ax_cv = fig.add_subplot(gs[0, 0:2])
names = list(cv_results.keys())
means = [cv_results[n].mean() for n in names]
stds  = [cv_results[n].std()  for n in names]
sorted_idx = np.argsort(means)
bar_colors = [ACCENT if n == "XGBoost" else NEG_COLOR for n in [names[i] for i in sorted_idx]]
bars = ax_cv.barh([names[i] for i in sorted_idx], [means[i] for i in sorted_idx],
                  xerr=[stds[i] for i in sorted_idx],
                  color=bar_colors, edgecolor="white", height=0.6,
                  error_kw=dict(ecolor="#8899AA", capsize=3, linewidth=1))
for bar, m in zip(bars, [means[i] for i in sorted_idx]):
    ax_cv.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
               f"{m:.4f}", va="center", fontsize=8.5)
ax_cv.set_title("5-Fold CV ROC-AUC Comparison", fontsize=12, fontweight="bold")
ax_cv.set_xlabel("ROC-AUC", fontsize=10)
ax_cv.set_xlim(0.5, 1.0)
 
# [0,2-3] Confusion matrix
ax_cm = fig.add_subplot(gs[0, 2:4])
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
annot  = np.array([[f"{cm[r,c]}\n({cm_pct[r,c]*100:.1f}%)"
                    for c in range(2)] for r in range(2)])
sns.heatmap(cm_pct, annot=annot, fmt="", cmap="Blues",
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"],
            ax=ax_cm, linewidths=1, linecolor="white",
            cbar_kws={"shrink": 0.75},
            annot_kws={"size": 12, "weight": "bold"})
ax_cm.set_title(f"Confusion Matrix  (AUC={roc_auc:.3f})",
                fontsize=12, fontweight="bold")
ax_cm.set_xlabel("Predicted", fontsize=10)
ax_cm.set_ylabel("Actual", fontsize=10)
 
# [1,0-1] ROC curve
ax_roc = fig.add_subplot(gs[1, 0:2])
ax_roc.plot(fpr, tpr, color=POS_COLOR, linewidth=2.5,
            label=f"XGBoost AUC = {roc_auc:.3f}")
ax_roc.plot([0,1],[0,1], color="grey", linestyle="--", linewidth=1, label="Random")
ax_roc.fill_between(fpr, tpr, alpha=0.12, color=POS_COLOR)
ax_roc.set_xlabel("False Positive Rate", fontsize=10)
ax_roc.set_ylabel("True Positive Rate", fontsize=10)
ax_roc.set_title("ROC Curve", fontsize=12, fontweight="bold")
ax_roc.legend(fontsize=10)
 
# [1,2-3] Precision-Recall curve
ax_pr = fig.add_subplot(gs[1, 2:4])
baseline = y_test.mean()
ax_pr.plot(rec, prec, color=NEG_COLOR, linewidth=2.5,
           label=f"XGBoost AP = {avg_prec:.3f}")
ax_pr.axhline(baseline, color="grey", linestyle="--",
              linewidth=1, label=f"Baseline ({baseline:.2f})")
ax_pr.fill_between(rec, prec, alpha=0.12, color=NEG_COLOR)
ax_pr.set_xlabel("Recall", fontsize=10)
ax_pr.set_ylabel("Precision", fontsize=10)
ax_pr.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
ax_pr.legend(fontsize=10)
 
# [2,0-1] SHAP feature importance
ax_shap = fig.add_subplot(gs[2, 0:2])
mean_shap_sorted = mean_shap.sort_values()
colors_shap = [ACCENT if f == mean_shap.idxmax() else NEG_COLOR
               for f in mean_shap_sorted.index]
bars_s = ax_shap.barh(mean_shap_sorted.index, mean_shap_sorted.values,
                      color=colors_shap, edgecolor="white", height=0.6)
for bar, v in zip(bars_s, mean_shap_sorted.values):
    ax_shap.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                 f"{v:.4f}", va="center", fontsize=8.5)
ax_shap.set_title("SHAP Mean |value| Feature Importance",
                  fontsize=12, fontweight="bold")
ax_shap.set_xlabel("Mean |SHAP value|", fontsize=10)
 
# [2,2-3] SHAP scatter (Glucose — top feature)
ax_sg = fig.add_subplot(gs[2, 2:4])
top_feat = mean_shap.idxmax()
top_idx  = FEATURES.index(top_feat)
sc = ax_sg.scatter(X_test[:, top_idx], shap_vals[:, top_idx],
                   c=X_test[:, top_idx], cmap="RdYlBu_r",
                   alpha=0.55, s=22, edgecolors="none")
plt.colorbar(sc, ax=ax_sg, label=f"Scaled {top_feat}")
ax_sg.axhline(0, color="grey", linestyle="--", linewidth=0.8)
ax_sg.set_xlabel(f"Scaled {top_feat}", fontsize=10)
ax_sg.set_ylabel("SHAP value", fontsize=10)
ax_sg.set_title(f"SHAP Scatter: {top_feat}", fontsize=12, fontweight="bold")
 
# [3,0-3] Threshold F1 analysis
ax_thr = fig.add_subplot(gs[3, 0:2])
ax_thr.plot(thresh_range, f1_scores, color=ACCENT, linewidth=2.5)
ax_thr.axvline(best_thresh, color=POS_COLOR, linestyle="--",
               linewidth=2, label=f"Best threshold = {best_thresh:.3f}")
ax_thr.axvline(0.5, color="grey", linestyle=":", linewidth=1, label="Default (0.5)")
ax_thr.set_xlabel("Decision Threshold", fontsize=10)
ax_thr.set_ylabel("F1 Score", fontsize=10)
ax_thr.set_title("F1 Score vs Decision Threshold", fontsize=12, fontweight="bold")
ax_thr.legend(fontsize=10)
 
# [3,2-3] Feature importance from XGBoost (gain)
ax_gain = fig.add_subplot(gs[3, 2:4])
gain_dict = best_xgb.get_booster().get_score(importance_type="gain")
gain_df   = pd.Series(gain_dict).sort_values()
bar_col_g = [ACCENT if f == gain_df.idxmax() else POS_COLOR for f in gain_df.index]
ax_gain.barh(gain_df.index, gain_df.values, color=bar_col_g,
             edgecolor="white", height=0.6)
ax_gain.set_title("XGBoost Feature Importance (Gain)",
                  fontsize=12, fontweight="bold")
ax_gain.set_xlabel("Gain", fontsize=10)
 
plt.savefig(OUT_IMAGE, dpi=155, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nPlot saved  → {OUT_IMAGE}")
print(f"Results txt → {OUT_TEXT}")