"""Report figures from results/model_comparison.csv (actual experiment outputs)."""
import sys, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
m = pd.read_csv(sys.argv[1]).sort_values("cv_macro_f1"); out = sys.argv[2]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].barh(m.model, m.cv_macro_f1, xerr=m.cv_macro_f1_std, color="#4C72B0"); ax[0].set_xlim(0.75, 0.95)
ax[0].set_title("CV Macro F1 (mean ± std, 5 folds)")
y = range(len(m)); h = 0.4
ax[1].barh([i + h / 2 for i in y], m.test_top1_accuracy, h, label="Top-1", color="#55A868")
ax[1].barh([i - h / 2 for i in y], m.test_macro_f1, h, label="Macro F1", color="#8172B2")
ax[1].set_yticks(list(y)); ax[1].set_yticklabels(m.model); ax[1].set_xlim(0.75, 0.95); ax[1].legend(loc="lower right")
ax[1].set_title("Student-test Top-1 and Macro F1")
ax[2].barh(m.model, m.test_log_loss, color="#C44E52"); ax[2].set_title("Student-test log loss (lower is better)")
for a in ax: a.grid(axis="x", alpha=0.3)
plt.tight_layout(); plt.savefig(out, dpi=130); print("saved", out)
