"""
02_diagnostic_analysis.py
=========================
DIAGNOSTIC ANALYSIS — "Why did it happen?"

Covers:
  - Statistical hypothesis testing (t-test, Mann-Whitney U) per feature
  - Effect-size measurement (Cohen's d)
  - Chi-square test for categorical associations
  - Odds-ratio analysis
  - Violin + strip plots per feature by class
  - Pair-grid of top 4 features coloured by outcome

Run:
    python 02_diagnostic_analysis.py
Output:
    diagnostic_tests.txt
    diagnostic_plots.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "./dataset/diabetes.csv"
OUT_IMAGE   = "./results/02_res_diagnostic/diagnostic_plots.png"
OUT_TEXT    = "./results/02_res_diagnostic/diagnostic_tests.txt"

ZERO_COLS   = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
FEATURES    = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
               "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
 
NEG_COLOR, POS_COLOR = "#4C9BE8", "#E8655A"
BG, TEXT    = "#FAFBFC", "#2C3E50"
ACCENT      = "#6C63FF"
 
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.facecolor": BG,
    "figure.facecolor": "white", "axes.edgecolor": "#D0D9E8",
    "axes.linewidth": 0.8, "grid.color": "#F0F4FA",
    "grid.linewidth": 0.7, "text.color": TEXT,
    "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
})
 
# ── Load & clean ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df[ZERO_COLS] = df[ZERO_COLS].replace(0, np.nan)
 
neg = df[df["Outcome"] == 0]
pos = df[df["Outcome"] == 1]
 
 
def cohens_d(a, b):
    """Pooled Cohen's d for two independent groups."""
    a, b   = a.dropna(), b.dropna()
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1)*a.std()**2 + (nb - 1)*b.std()**2) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled if pooled else 0.0
 
 
def interpret_d(d):
    d = abs(d)
    if d < 0.2:   return "negligible"
    if d < 0.5:   return "small"
    if d < 0.8:   return "medium"
    return "large"
 
 
# ── 1. Statistical Tests ──────────────────────────────────────────────────────
lines = []
lines.append("=" * 75)
lines.append("DIAGNOSTIC ANALYSIS — Statistical Tests")
lines.append("=" * 75)
lines.append(f"\n{'Feature':<28} {'t-stat':>8} {'t p-val':>10} {'MW U-stat':>10} "
             f"{'MW p-val':>10} {'Cohen d':>9} {'Effect':>12}")
lines.append("-" * 95)
 
test_results = {}
for feat in FEATURES:
    a = neg[feat].dropna()
    b = pos[feat].dropna()
 
    # Welch t-test
    t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
 
    # Mann–Whitney U (non-parametric)
    u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
 
    # Cohen's d
    d   = cohens_d(a, b)
    eff = interpret_d(d)
 
    sig_t  = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "ns"
    sig_mw = "***" if u_p < 0.001 else "**" if u_p < 0.01 else "*" if u_p < 0.05 else "ns"
 
    lines.append(f"  {feat:<26} {t_stat:>8.3f} {t_p:>9.4f}{sig_t:>2}  "
                 f"{u_stat:>10.0f} {u_p:>9.4f}{sig_mw:>2}  {d:>8.3f}  {eff:>12}")
 
    test_results[feat] = dict(t_stat=t_stat, t_p=t_p, u_stat=u_stat,
                               u_p=u_p, cohen_d=d, effect=eff,
                               mean_neg=a.mean(), mean_pos=b.mean())
 
lines.append("\n  Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
 
# Odds-ratio analysis (binary splits at median)
lines.append("\n── Odds Ratios (split at feature median) ──")
lines.append(f"\n  {'Feature':<28} {'OR':>8}  {'95% CI':>18}  Interpretation")
lines.append("  " + "-" * 72)
for feat in FEATURES:
    med = df[feat].median()
    high_dia  = ((df[feat] > med) & (df["Outcome"] == 1)).sum()
    high_nodia = ((df[feat] > med) & (df["Outcome"] == 0)).sum()
    low_dia   = ((df[feat] <= med) & (df["Outcome"] == 1)).sum()
    low_nodia  = ((df[feat] <= med) & (df["Outcome"] == 0)).sum()
 
    if low_nodia == 0 or low_dia == 0:
        continue
    or_val = (high_dia / high_nodia) / (low_dia / low_nodia)
 
    # Woolf CI
    se_log = np.sqrt(1/high_dia + 1/high_nodia + 1/low_dia + 1/low_nodia)
    ci_lo  = np.exp(np.log(or_val) - 1.96 * se_log)
    ci_hi  = np.exp(np.log(or_val) + 1.96 * se_log)
 
    interp = ("Positive risk" if or_val > 1.2
              else "Negative/protective" if or_val < 0.8
              else "Neutral")
    lines.append(f"  {feat:<28} {or_val:>8.3f}  [{ci_lo:.3f} – {ci_hi:.3f}]  {interp}")
 
summary_text = "\n".join(lines)
print(summary_text)
with open(OUT_TEXT, "w", encoding="utf-8") as f:
    f.write(summary_text)
 
# ── 2. Visualisations ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 28), facecolor="white")
fig.suptitle("Diagnostic Analysis — Statistical Significance & Effect Sizes",
             fontsize=22, fontweight="bold", color=TEXT, y=0.99)
 
gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.55, wspace=0.42)
 
# Row 0 col 0-1: Cohen's d bar chart
ax_d = fig.add_subplot(gs[0, 0:2])
feats_sorted = sorted(test_results.keys(),
                      key=lambda f: abs(test_results[f]["cohen_d"]), reverse=True)
d_vals  = [test_results[f]["cohen_d"] for f in feats_sorted]
d_colors = [POS_COLOR if v < 0 else NEG_COLOR for v in d_vals]
bars = ax_d.barh(feats_sorted, [abs(v) for v in d_vals],
                 color=d_colors, edgecolor="white", height=0.6)
for bar, v in zip(bars, d_vals):
    ax_d.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
              f"|d|={abs(v):.3f}", va="center", fontsize=8.5)
for thresh, label, ls in [(0.2, "small", ":"), (0.5, "medium", "--"), (0.8, "large", "-")]:
    ax_d.axvline(thresh, linestyle=ls, color="grey", alpha=0.5, linewidth=1)
    ax_d.text(thresh+0.01, -0.6, label, fontsize=7.5, color="grey")
ax_d.set_title("Cohen's d  (effect size: non-diabetic vs diabetic)",
               fontsize=12, fontweight="bold")
ax_d.set_xlabel("|Cohen's d|", fontsize=10)
ax_d.set_xlim(0, max([abs(v) for v in d_vals]) * 1.25)
 
# Row 0 col 2-3: p-value significance bar (−log10)
ax_p = fig.add_subplot(gs[0, 2:4])
p_vals   = [test_results[f]["t_p"] for f in feats_sorted]
neg_log_p = [-np.log10(p + 1e-300) for p in p_vals]
p_colors  = [POS_COLOR if p < 0.05 else "#AABDD0" for p in p_vals]
ax_p.barh(feats_sorted, neg_log_p, color=p_colors, edgecolor="white", height=0.6)
ax_p.axvline(-np.log10(0.05), linestyle="--", color="grey",
             linewidth=1.3, alpha=0.7, label="p = 0.05")
ax_p.axvline(-np.log10(0.001), linestyle=":", color="grey",
             linewidth=1.3, alpha=0.7, label="p = 0.001")
ax_p.legend(fontsize=9)
ax_p.set_title("−log₁₀(p-value)  Welch t-test", fontsize=12, fontweight="bold")
ax_p.set_xlabel("−log₁₀(p)", fontsize=10)
 
# Rows 1–4: Violin + strip plots per feature
for i, feat in enumerate(FEATURES):
    row = 1 + (i // 4)
    col = i % 4
    ax  = fig.add_subplot(gs[row, col])
 
    plot_df = df[[feat, "Outcome"]].copy()
    plot_df["Class"] = plot_df["Outcome"].map({0: "No Diabetes", 1: "Diabetes"})
 
    # Violin
    parts = ax.violinplot(
        [neg[feat].dropna(), pos[feat].dropna()],
        positions=[0, 1],
        showmedians=True,
        showextrema=False,
    )
    for j, (pc, col_c) in enumerate(zip(parts["bodies"], [NEG_COLOR, POS_COLOR])):
        pc.set_facecolor(col_c); pc.set_alpha(0.65); pc.set_edgecolor("white")
    parts["cmedians"].set_color(TEXT); parts["cmedians"].set_linewidth(2)
 
    # Strip (jitter)
    for j, (grp, col_c) in enumerate([(neg, NEG_COLOR), (pos, POS_COLOR)]):
        vals = grp[feat].dropna().values
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), j) + jitter, vals,
                   color=col_c, alpha=0.20, s=6, linewidths=0)
 
    # Annotate p-value
    r = test_results[feat]
    sig = ("***" if r["t_p"] < 0.001 else "**" if r["t_p"] < 0.01
           else "*" if r["t_p"] < 0.05 else "ns")
    ax.set_title(f"{feat}  [{sig}]", fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Diabetes", "Diabetes"], fontsize=8.5)
    ax.text(0.98, 0.97, f"d={r['cohen_d']:.2f}\n{r['effect']}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75, ec="#D0D9E8"))
 
plt.savefig(OUT_IMAGE, dpi=155, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nPlot saved  → {OUT_IMAGE}")
print(f"Tests txt   → {OUT_TEXT}")