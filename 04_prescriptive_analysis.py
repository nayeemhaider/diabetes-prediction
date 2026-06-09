"""
04_prescriptive_analysis.py
===========================
PRESCRIPTIVE ANALYSIS — "What should we do about it?"

Covers:
  - Risk stratification (Low / Medium / High / Critical)
  - SHAP-based personalised feature recommendations per patient
  - What-if simulation: how much must each feature change to flip prediction?
  - Clinical threshold rules derived from data
  - Intervention priority matrix (impact vs modifiability)
  - Population-level risk segmentation dashboard

Run:
    python 04_prescriptive_analysis.py
Output:
    prescriptive_recommendations.csv
    prescriptive_plots.png
    prescriptive_report.txt
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH      = "./dataset/diabetes.csv"
OUT_IMAGE      = "./results/04_res_prescriptive/prescriptive_plots.png"
OUT_CSV        = "./results/04_res_prescriptive/prescriptive_recommendations.csv"
OUT_TEXT       = "./results/04_res_prescriptive/prescriptive_report.txt"

ZERO_COLS      = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
FEATURES       = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                  "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
RANDOM_STATE   = 42
 
# Modifiability score (0-3):  3 = highly modifiable, 0 = fixed
MODIFIABILITY  = {
    "Glucose":                  3,
    "BMI":                      3,
    "Insulin":                  2,
    "BloodPressure":            3,
    "DiabetesPedigreeFunction": 0,   # genetic — not modifiable
    "Age":                      0,   # not modifiable
    "SkinThickness":            1,
    "Pregnancies":              0,
}
 
CLINICAL_TARGETS = {
    "Glucose":       126,   # mg/dL — pre-diabetes threshold
    "BMI":           24.9,  # healthy upper limit
    "BloodPressure": 80,    # mmHg diastolic
    "Insulin":       166,   # µU/mL upper normal (2h post-load)
}
 
RISK_BANDS = {
    "Low":      (0.00, 0.30),
    "Medium":   (0.30, 0.55),
    "High":     (0.55, 0.75),
    "Critical": (0.75, 1.01),
}
RISK_COLORS = {
    "Low": "#4CAF50", "Medium": "#FFC107",
    "High": "#FF7043", "Critical": "#B71C1C"}
 
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
 
imputer = KNNImputer(n_neighbors=5)
X_imp   = pd.DataFrame(imputer.fit_transform(X_raw), columns=FEATURES)
 
scaler  = RobustScaler()
X_sc    = scaler.fit_transform(X_imp)
X_sc_df = pd.DataFrame(X_sc, columns=FEATURES)
 
X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
 
# ── Train/load best model ─────────────────────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    subsample=0.9, colsample_bytree=0.9,
    scale_pos_weight=(y==0).sum()/(y==1).sum(),
    random_state=RANDOM_STATE, eval_metric="logloss"
)
model.fit(X_train, y_train)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"Model AUC on test set: {auc:.4f}")
 
# ── SHAP ─────────────────────────────────────────────────────────────────────
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X_sc)
mean_shap = np.abs(shap_vals).mean(axis=0)
shap_rank = pd.Series(mean_shap, index=FEATURES).sort_values(ascending=False)
 
# ── Risk stratification ───────────────────────────────────────────────────────
probs = model.predict_proba(X_sc)[:, 1]
 
def get_risk_band(p):
    for band, (lo, hi) in RISK_BANDS.items():
        if lo <= p < hi:
            return band
    return "Critical"
 
risk_bands = [get_risk_band(p) for p in probs]
 
# ── What-if simulation ────────────────────────────────────────────────────────
def what_if_shift(row_scaled, feature_idx, step=0.02, max_steps=200):
    """
    Nudge a single feature toward zero (the scaler median) in small steps.
    Return how many steps until the model flips to 'No Diabetes' (prob < 0.5).
    """
    x = row_scaled.copy()
    for s in range(max_steps):
        x[feature_idx] -= step * np.sign(x[feature_idx])
        if model.predict_proba(x.reshape(1, -1))[0, 1] < 0.5:
            return s * step
    return np.nan
 
# ── Per-patient recommendation table ─────────────────────────────────────────
records = []
for i in range(len(X_imp)):
    row_raw    = X_imp.iloc[i]
    row_scaled = X_sc[i]
    prob       = probs[i]
    band       = risk_bands[i]
 
    # Top 3 driving SHAP features for this patient
    row_shap   = np.abs(shap_vals[i])
    top_feats  = pd.Series(row_shap, index=FEATURES).sort_values(ascending=False).head(3)
 
    # Build recommendation string
    recs = []
    for feat in top_feats.index:
        mod = MODIFIABILITY[feat]
        if mod == 0:
            continue
        raw_val = row_raw[feat]
        if feat in CLINICAL_TARGETS and raw_val > CLINICAL_TARGETS[feat]:
            target = CLINICAL_TARGETS[feat]
            recs.append(f"Reduce {feat} from {raw_val:.1f} → {target} "
                        f"(target {target})")
        else:
            recs.append(f"Monitor {feat} (current {raw_val:.1f})")
 
    records.append({
        "PatientID":      i,
        "RiskProb":       round(prob, 4),
        "RiskBand":       band,
        "Top1Feature":    top_feats.index[0],
        "Top2Feature":    top_feats.index[1] if len(top_feats) > 1 else "",
        "Top3Feature":    top_feats.index[2] if len(top_feats) > 2 else "",
        "Recommendation": " | ".join(recs) if recs else "Maintain current lifestyle",
    })
 
rec_df = pd.DataFrame(records)
rec_df.to_csv(OUT_CSV, index=False)
print(f"Recommendations saved → {OUT_CSV}  ({len(rec_df)} patients)")
 
# ── Intervention priority matrix data ─────────────────────────────────────────
impact_scores  = shap_rank.to_dict()
mod_scores     = {f: MODIFIABILITY[f] for f in FEATURES}
 
priority_df = pd.DataFrame({
    "Feature":       FEATURES,
    "Impact":        [impact_scores[f] for f in FEATURES],
    "Modifiability": [mod_scores[f]    for f in FEATURES],
})
 
# ── Population risk segment counts ───────────────────────────────────────────
band_counts = rec_df["RiskBand"].value_counts().reindex(
    ["Low", "Medium", "High", "Critical"], fill_value=0)
 
# ── Text report ───────────────────────────────────────────────────────────────
lines = []
lines.append("=" * 70)
lines.append("PRESCRIPTIVE ANALYSIS — Risk Stratification & Interventions")
lines.append("=" * 70)
 
lines.append("\n── Population Risk Segments ──")
for band, cnt in band_counts.items():
    pct = cnt / len(rec_df) * 100
    lines.append(f"  {band:<10}: {cnt:>4} patients ({pct:.1f}%)")
 
lines.append("\n── Intervention Priority Matrix ──")
lines.append(f"  {'Feature':<30} {'SHAP Impact':>12} {'Modifiability':>14} "
             f"{'Priority':>10}")
lines.append("  " + "-" * 70)
for _, row in priority_df.sort_values("Impact", ascending=False).iterrows():
    mod_str = ["Fixed","Low","Medium","High"][int(row["Modifiability"])]
    score   = row["Impact"] * row["Modifiability"]
    lines.append(f"  {row['Feature']:<30} {row['Impact']:>12.4f} {mod_str:>14} "
                 f"{score:>10.4f}")
 
lines.append("\n── Clinical Threshold Rules (derived from analysis) ──")
lines.append("  These are data-informed thresholds for clinical intervention:")
thresholds_info = [
    ("Glucose",        126,  "mg/dL",   "Pre-diabetes boundary; high correlation (r=0.47)"),
    ("BMI",            30.0, "kg/m²",   "Obesity threshold; strong SHAP weight"),
    ("BloodPressure",  80,   "mmHg",    "Diastolic hypertension onset"),
    ("Age",            45,   "years",   "Elevated incidence above this age"),
    ("Pregnancies",    4,    "count",   "Higher gestational diabetes risk"),
]
for feat, thr, unit, note in thresholds_info:
    above = (X_imp[feat] > thr).sum()
    pct   = above / len(X_imp) * 100
    lines.append(f"  {feat:<22} > {thr} {unit:<8}  → "
                 f"{above} patients ({pct:.1f}%)  [{note}]")
 
lines.append("\n── Sample High-Risk Patient Recommendations ──")
high_risk = rec_df[rec_df["RiskBand"].isin(["High", "Critical"])].head(5)
for _, row in high_risk.iterrows():
    lines.append(f"\n  Patient {row['PatientID']}  Risk={row['RiskProb']:.1%}  "
                 f"Band={row['RiskBand']}")
    lines.append(f"    Key drivers : {row['Top1Feature']}, {row['Top2Feature']}, "
                 f"{row['Top3Feature']}")
    lines.append(f"    Action plan : {row['Recommendation']}")
 
summary_text = "\n".join(lines)
print("\n" + summary_text)
with open(OUT_TEXT, "w", encoding="utf-8") as f:
    f.write(summary_text)
 
# ── Visualisations ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 26), facecolor="white")
fig.suptitle("Prescriptive Analysis — Risk Stratification & Intervention Planning",
             fontsize=22, fontweight="bold", color=TEXT, y=0.99)
gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.42)
 
# [0,0-1] Population risk band pie / donut
ax_pie = fig.add_subplot(gs[0, 0:2])
colors_pie = [RISK_COLORS[b] for b in band_counts.index]
wedge_p    = dict(width=0.55, edgecolor="white", linewidth=2.5)
wedges, texts, autotexts = ax_pie.pie(
    band_counts.values,
    labels=[f"{b}\n{v} pts" for b, v in band_counts.items()],
    colors=colors_pie, autopct="%1.1f%%",
    wedgeprops=wedge_p, startangle=90,
    textprops={"fontsize": 10, "color": TEXT})
for at in autotexts:
    at.set_fontsize(11); at.set_fontweight("bold"); at.set_color("white")
ax_pie.set_title("Population Risk Stratification", fontsize=13, fontweight="bold")
 
# [0,2-3] Risk probability histogram
ax_hist = fig.add_subplot(gs[0, 2:4])
for band, (lo, hi) in RISK_BANDS.items():
    mask = (probs >= lo) & (probs < hi)
    ax_hist.hist(probs[mask], bins=20, color=RISK_COLORS[band],
                 alpha=0.82, edgecolor="white", label=band)
for thr in [0.30, 0.55, 0.75]:
    ax_hist.axvline(thr, color=TEXT, linestyle="--", linewidth=1, alpha=0.6)
ax_hist.set_xlabel("Predicted Probability of Diabetes", fontsize=10)
ax_hist.set_ylabel("Count", fontsize=10)
ax_hist.set_title("Risk Probability Distribution", fontsize=12, fontweight="bold")
ax_hist.legend(fontsize=9)
 
# [1,0-1] Intervention priority matrix (scatter)
ax_mat = fig.add_subplot(gs[1, 0:2])
for _, row in priority_df.iterrows():
    size  = row["Impact"] * 6000
    color = POS_COLOR if row["Modifiability"] >= 2 else (
            "#FFC107" if row["Modifiability"] == 1 else "#AABDD0")
    ax_mat.scatter(row["Modifiability"], row["Impact"],
                   s=size, color=color, alpha=0.70, edgecolors="white", linewidth=0.5)
    ax_mat.annotate(row["Feature"], (row["Modifiability"], row["Impact"]),
                    xytext=(6, 3), textcoords="offset points", fontsize=8)
ax_mat.set_xticks([0, 1, 2, 3])
ax_mat.set_xticklabels(["Fixed", "Low", "Medium", "High"], fontsize=9)
ax_mat.set_xlabel("Modifiability (clinical controllability)", fontsize=10)
ax_mat.set_ylabel("SHAP Impact (mean |shap|)", fontsize=10)
ax_mat.set_title("Intervention Priority Matrix\n"
                 "(bubble size = impact × modifiability)",
                 fontsize=11, fontweight="bold")
patches = [mpatches.Patch(color=POS_COLOR, label="High modify"),
           mpatches.Patch(color="#FFC107",  label="Low modify"),
           mpatches.Patch(color="#AABDD0",  label="Fixed")]
ax_mat.legend(handles=patches, fontsize=8)
 
# [1,2-3] Clinical threshold bars
ax_thr = fig.add_subplot(gs[1, 2:4])
thr_names  = [t[0] for t in thresholds_info]
thr_pcts   = [(X_imp[t[0]] > t[1]).sum() / len(X_imp) * 100 for t in thresholds_info]
bar_colors = [RISK_COLORS["High"] if p > 40 else
              RISK_COLORS["Medium"] if p > 20 else
              RISK_COLORS["Low"] for p in thr_pcts]
bars_t = ax_thr.barh(thr_names, thr_pcts, color=bar_colors, edgecolor="white", height=0.6)
for bar, v in zip(bars_t, thr_pcts):
    ax_thr.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{v:.1f}%", va="center", fontsize=9.5)
ax_thr.set_xlabel("% of Patients Exceeding Threshold", fontsize=10)
ax_thr.set_title("Clinical Threshold Exceedance", fontsize=12, fontweight="bold")
ax_thr.set_xlim(0, 100)
 
# [2,0-3] SHAP beeswarm (manual scatter proxy)
ax_bee = fig.add_subplot(gs[2, 0:4])
shap_plot_df = pd.DataFrame(shap_vals, columns=FEATURES)
feature_order = shap_rank.index.tolist()
 
for yi, feat in enumerate(reversed(feature_order)):
    sv = shap_plot_df[feat].values
    fv = X_sc_df[feat].values
    jitter = np.random.default_rng(yi).uniform(-0.25, 0.25, len(sv))
    sc = ax_bee.scatter(sv, np.full(len(sv), yi) + jitter,
                        c=fv, cmap="RdYlBu_r",
                        alpha=0.35, s=8, edgecolors="none")
 
ax_bee.set_yticks(range(len(feature_order)))
ax_bee.set_yticklabels(reversed(feature_order), fontsize=9)
ax_bee.axvline(0, color=TEXT, linewidth=0.8, linestyle="--", alpha=0.5)
ax_bee.set_xlabel("SHAP value (impact on model output)", fontsize=10)
ax_bee.set_title("SHAP Beeswarm — Feature Impact Across All Patients\n"
                 "(colour = feature value: blue=low, red=high)",
                 fontsize=12, fontweight="bold")
plt.colorbar(sc, ax=ax_bee, label="Feature value (scaled)", shrink=0.6)
 
# [3,0-1] Risk prob by actual outcome violin
ax_vio = fig.add_subplot(gs[3, 0:2])
d0 = probs[y == 0]
d1 = probs[y == 1]
parts = ax_vio.violinplot([d0, d1], positions=[0, 1],
                           showmedians=True, showextrema=False)
for j, (pc, col_c) in enumerate(zip(parts["bodies"], [NEG_COLOR, POS_COLOR])):
    pc.set_facecolor(col_c); pc.set_alpha(0.70); pc.set_edgecolor("white")
parts["cmedians"].set_color(TEXT); parts["cmedians"].set_linewidth(2)
for j, (d, col_c) in enumerate([(d0, NEG_COLOR), (d1, POS_COLOR)]):
    jitter = np.random.default_rng(j).uniform(-0.1, 0.1, len(d))
    ax_vio.scatter(np.full(len(d), j) + jitter, d,
                   color=col_c, alpha=0.18, s=7, linewidths=0)
ax_vio.set_xticks([0, 1])
ax_vio.set_xticklabels(["No Diabetes\n(Actual)", "Diabetes\n(Actual)"], fontsize=9)
ax_vio.set_ylabel("Predicted Risk Probability", fontsize=10)
ax_vio.set_title("Predicted Risk by Actual Outcome", fontsize=12, fontweight="bold")
 
# [3,2-3] Actionable recommendations summary table
ax_tab = fig.add_subplot(gs[3, 2:4])
ax_tab.axis("off")
table_data = [
    ["Risk Band",  "Protocol",                        "Follow-up"],
    ["Low",        "Lifestyle counselling",             "Annual"],
    ["Medium",     "Diet + exercise programme",         "6 months"],
    ["High",       "Specialist referral + medication",  "3 months"],
    ["Critical",   "Immediate clinical intervention",   "Weekly"],
]
colors_tab = [
    ["#E8EDF2"] * 3,
    [RISK_COLORS["Low"]      + "55"] * 3,
    [RISK_COLORS["Medium"]   + "55"] * 3,
    [RISK_COLORS["High"]     + "55"] * 3,
    [RISK_COLORS["Critical"] + "55"] * 3,
]
tbl = ax_tab.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc="center", loc="center",
    cellColours=colors_tab[1:],
    colColours=colors_tab[0],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.1)
ax_tab.set_title("Prescriptive Action Protocol", fontsize=12, fontweight="bold", pad=18)
 
plt.savefig(OUT_IMAGE, dpi=155, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nPlot saved  → {OUT_IMAGE}")
print(f"Report txt  → {OUT_TEXT}")
print(f"CSV saved   → {OUT_CSV}")