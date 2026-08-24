"""JUDGe figures over the full five-judge ladder.

Judges are ordered by reference-free balanced accuracy on the seen arm, which is the
capability axis the paper argues on. No chart titles or in-axes conclusions: the caption
carries the reading.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
INK, BLUE, RED, GREY = "#1a1a1a", "#2166ac", "#b2182b", "#7f7f7f"

J = ["Instella\n3B", "Llama 3.1\n8B", "Qwen2.5\n7B", "Qwen2.5\n14B", "Qwen2.5\n32B"]
x = range(len(J))
given     = [-0.054, -0.074, -0.036, -0.023, -0.062]
given_lo  = [-0.083, -0.116, -0.068, -0.057, -0.103]
given_hi  = [-0.026, -0.032, -0.008,  0.006, -0.021]
with_     = [-0.081, -0.243, -0.200, -0.184, -0.191]
with_lo   = [-0.109, -0.281, -0.248, -0.231, -0.239]
with_hi   = [-0.054, -0.201, -0.149, -0.133, -0.140]

fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 2.2))

a.axhline(0, color=GREY, linewidth=0.7, zorder=0)
off = 0.10
a.errorbar([i - off for i in x], given,
           yerr=[[g - l for g, l in zip(given, given_lo)], [h - g for g, h in zip(given, given_hi)]],
           fmt="o", color=BLUE, markersize=4, capsize=2.4, elinewidth=0.75,
           linestyle="-", linewidth=1.0, label="Reference given")
a.errorbar([i + off for i in x], with_,
           yerr=[[w - l for w, l in zip(with_, with_lo)], [h - w for w, h in zip(with_, with_hi)]],
           fmt="s", color=RED, markersize=4, capsize=2.4, elinewidth=0.75,
           linestyle="--", linewidth=1.0, label="Reference withheld")
a.set_xticks(list(x)); a.set_xticklabels(J)
a.set_ylabel("Balanced accuracy,\nseen minus unseen")
a.set_ylim(-0.30, 0.05); a.yaxis.set_major_locator(MultipleLocator(0.1))
a.legend(loc="lower left", handlelength=1.7)
a.text(-0.03, 1.06, "(a)", transform=a.transAxes, fontweight="bold", fontsize=9)

# Decomposition: the gap is specificity, and withholding the reference triples it.
D = ["Llama 3.1 8B", "Qwen2.5 32B"]
xi = range(len(D))
sens_w   = [ 0.011, -0.004]; sens_lo = [-0.005, -0.024]; sens_hi = [ 0.034,  0.016]
spec_g   = [-0.151, -0.124]; spg_lo  = [-0.236, -0.207]; spg_hi  = [-0.068, -0.042]
spec_w   = [-0.496, -0.378]; spw_lo  = [-0.573, -0.472]; spw_hi  = [-0.415, -0.276]
b.axhline(0, color=GREY, linewidth=0.7, zorder=0)
o = 0.16
b.errorbar([i - o for i in xi], sens_w,
           yerr=[[s - l for s, l in zip(sens_w, sens_lo)], [h - s for s, h in zip(sens_w, sens_hi)]],
           fmt="^", color=INK, markersize=4, capsize=2.4, elinewidth=0.75,
           linestyle="none", label="Sensitivity, withheld")
b.errorbar(list(xi), spec_g,
           yerr=[[s - l for s, l in zip(spec_g, spg_lo)], [h - s for s, h in zip(spec_g, spg_hi)]],
           fmt="o", color=BLUE, markersize=4, capsize=2.4, elinewidth=0.75,
           linestyle="none", label="Specificity, given")
b.errorbar([i + o for i in xi], spec_w,
           yerr=[[s - l for s, l in zip(spec_w, spw_lo)], [h - s for s, h in zip(spec_w, spw_hi)]],
           fmt="s", color=RED, markersize=4, capsize=2.4, elinewidth=0.75,
           linestyle="none", label="Specificity, withheld")
b.set_xticks(list(xi)); b.set_xticklabels(D)
b.set_ylabel("Component gap,\nseen minus unseen")
b.set_ylim(-0.65, 0.12); b.yaxis.set_major_locator(MultipleLocator(0.2))
b.set_xlim(-0.5, 1.5)
b.legend(loc="lower left", handlelength=1.4, labelspacing=0.25)
b.text(-0.03, 1.06, "(b)", transform=b.transAxes, fontweight="bold", fontsize=9)

fig.subplots_adjust(wspace=0.42)
for ext in ("pdf", "png"):
    fig.savefig(f"judge_fig1_dissociation.{ext}")
print("wrote judge_fig1_dissociation")
