"""
01_descriptive_analysis.py
==========================
DESCRIPTIVE ANALYSIS — "What happened / What does the data look like?"

Covers:
  - Summary statistics (mean, median, std, quartiles, skewness, kurtosis)
  - Missing-value audit (zeros as NaN)
  - Class distribution
  - Per-feature distribution plots (histogram + KDE + boxplot)
  - Correlation heatmap
  - Outlier summary table

Run:
    python 01_descriptive_analysis.py
Output:
    descriptive_summary.txt
    descriptive_plots.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "./dataset/diabetes.csv"
OUT_IMAGE   = "./results/01_res_descriptive/descriptive_plots.png"
OUT_TEXT    = "./results/01_res_descriptive/descriptive_summary.txt"

ZERO_COLS   = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
FEATURES    = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
               "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
 
NEG_COLOR, POS_COLOR = "#4C9BE8", "#E8655A"
BG, TEXT    = "#FAFBFC", "#2C3E50"
 
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.facecolor": BG,
    "figure.facecolor": "white", "axes.edgecolor": "#D0D9E8",
    "axes.linewidth": 0.8, "grid.color": "#F0F4FA",
    "grid.linewidth": 0.7, "text.color": TEXT,
    "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
})
 
# ── Load & preprocess zeros ───────────────────────────────────────────────────
raw_df = pd.read_csv(DATA_PATH)
df     = raw_df.copy()
df[ZERO_COLS] = df[ZERO_COLS].replace(0, np.nan)   # treat zeros as missing
 
# ── 1. Text Summary ───────────────────────────────────────────────────────────
lines = []
lines.append("=" * 70)
lines.append("DESCRIPTIVE ANALYSIS — Diabetes Dataset")
lines.append("=" * 70)
 
lines.append(f"\nDataset shape : {df.shape[0]} rows × {df.shape[1]} columns")
lines.append(f"Total cells   : {df.size}")
lines.append(f"Duplicate rows: {df.duplicated().sum()}")
 
lines.append("\n── Missing Values (after zero → NaN replacement) ──")
miss = df.isnull().sum()
miss_pct = (miss / len(df) * 100).round(2)
miss_df = pd.DataFrame({"Count": miss, "Pct (%)": miss_pct})
lines.append(miss_df[miss_df["Count"] > 0].to_string())
 
lines.append("\n── Class Distribution ──")
vc = df["Outcome"].value_counts()
lines.append(f"  Non-Diabetic (0): {vc[0]}  ({vc[0]/len(df)*100:.1f}%)")
lines.append(f"  Diabetic     (1): {vc[1]}  ({vc[1]/len(df)*100:.1f}%)")
 
lines.append("\n── Summary Statistics ──")
desc = df[FEATURES].describe().T
desc["skewness"] = df[FEATURES].skew().round(3)
desc["kurtosis"] = df[FEATURES].kurt().round(3)
lines.append(desc.round(3).to_string())
 
lines.append("\n── Outlier Count (IQR × 1.5 rule) ──")
for col in FEATURES:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr     = q3 - q1
    n_out   = ((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).sum()
    lines.append(f"  {col:<30}: {n_out} outliers")
 
summary_text = "\n".join(lines)
print(summary_text)
 
with open(OUT_TEXT, "w", encoding="utf-8") as f:
    f.write(summary_text)
 
# ── 2. Visualisations ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 26), facecolor="white")
fig.suptitle("Descriptive Analysis — Diabetes Dataset",
             fontsize=22, fontweight="bold", color=TEXT, y=0.99)
 
gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.55, wspace=0.40)
 
# Row 0: Class pie + correlation heatmap
# ── Pie ──────────────────────────────────────────────────────────────────────
ax_pie = fig.add_subplot(gs[0, 0:2])
counts  = df["Outcome"].value_counts()
wedge   = dict(width=0.55, edgecolor="white", linewidth=2.5)
wedges, texts, autotexts = ax_pie.pie(
    counts,
    labels=["Non-Diabetic (500)", "Diabetic (268)"],
    autopct="%1.1f%%",
    colors=[NEG_COLOR, POS_COLOR],
    wedgeprops=wedge,
    startangle=90,
    textprops={"fontsize": 11, "color": TEXT},
)
for at in autotexts:
    at.set_fontsize(12); at.set_fontweight("bold"); at.set_color("white")
ax_pie.set_title("Class Distribution", fontsize=13, fontweight="bold", pad=10)
 
# ── Correlation heatmap ───────────────────────────────────────────────────────
ax_heat = fig.add_subplot(gs[0, 2:4])
corr    = df[FEATURES + ["Outcome"]].corr()
mask    = np.triu(np.ones_like(corr, dtype=bool))
cmap    = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            annot=True, fmt=".2f", linewidths=0.5,
            linecolor="#E0E8F0", annot_kws={"size": 8},
            ax=ax_heat, cbar_kws={"shrink": 0.8})
ax_heat.set_title("Correlation Matrix", fontsize=13, fontweight="bold")
ax_heat.tick_params(axis="x", rotation=35, labelsize=8)
ax_heat.tick_params(axis="y", labelsize=8)
 
# Rows 1–4: Per-feature histogram+KDE+boxplot (2 axes per feature)
for i, feat in enumerate(FEATURES):
    row = 1 + (i // 4)
    col = i % 4
    ax  = fig.add_subplot(gs[row, col])
 
    d0 = df[df["Outcome"] == 0][feat].dropna()
    d1 = df[df["Outcome"] == 1][feat].dropna()
 
    # Histogram
    ax.hist(d0, bins=20, color=NEG_COLOR, alpha=0.60,
            label="No Diabetes", edgecolor="white", density=True)
    ax.hist(d1, bins=20, color=POS_COLOR, alpha=0.60,
            label="Diabetes", edgecolor="white", density=True)
 
    # KDE
    from scipy.stats import gaussian_kde
    for data, color in [(d0, NEG_COLOR), (d1, POS_COLOR)]:
        if len(data) > 5:
            kde  = gaussian_kde(data, bw_method=0.3)
            xs   = np.linspace(data.min(), data.max(), 200)
            ax.plot(xs, kde(xs), color=color, linewidth=2)
 
    # Mean lines
    ax.axvline(d0.mean(), color=NEG_COLOR, linestyle="--",
               linewidth=1.5, alpha=0.85)
    ax.axvline(d1.mean(), color=POS_COLOR, linestyle="--",
               linewidth=1.5, alpha=0.85)
 
    # Stats annotation
    ax.text(0.98, 0.96,
            f"μ₀={d0.mean():.1f}\nμ₁={d1.mean():.1f}\nσ={df[feat].std():.1f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color=TEXT,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="#D0D9E8"))
 
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.set_ylabel("Density", fontsize=8)
    if i == 0:
        ax.legend(fontsize=7.5, loc="upper right")
 
plt.savefig(OUT_IMAGE, dpi=155, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nPlot saved  → {OUT_IMAGE}")
print(f"Summary txt → {OUT_TEXT}")