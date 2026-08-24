import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)
INK, BLUE, RED, GREY = "#1a1a1a", "#2166ac", "#b2182b", "#7f7f7f"

d = json.load(open("fig_data_trajectory.json"))
labels = ["Stage 1", "Instella 3B", "SFT", "Instruct"]
short = ["Stage 1", "Inst 3B", "SFT", "Instruct"]
x = range(len(labels))

fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.35))
(a, b, c), (dd, e, f) = axes

# (a) trajectory
for arm, col, mk, ls, nm in (
    ("train", BLUE, "o", "-", "In corpus"),
    ("test", RED, "s", "--", "Verified absent"),
):
    y = [d[f"{arm}|{label}"]["rate"] for label in labels]
    lo = [y[i] - d[f"{arm}|{label}"]["lo"] for i, label in enumerate(labels)]
    hi = [d[f"{arm}|{label}"]["hi"] - y[i] for i, label in enumerate(labels)]
    a.errorbar(
        x,
        y,
        yerr=[lo, hi],
        color=col,
        marker=mk,
        linestyle=ls,
        markersize=3.2,
        linewidth=0.95,
        capsize=2,
        elinewidth=0.65,
        label=nm,
    )
a.axvline(0.5, color=GREY, linewidth=0.7, linestyle=":", zorder=0)
a.set_xticks(list(x))
a.set_xticklabels(short, rotation=30, ha="right")
a.set_ylabel("Deletion recall (%)")
a.set_ylim(0, 6.2)
a.yaxis.set_major_locator(MultipleLocator(2))
a.legend(loc="upper left", handlelength=1.4)
a.text(-0.04, 1.09, "(a)", transform=a.transAxes, fontweight="bold", fontsize=8)

# (b) DiD
bn = ["Instella 3B", "SFT", "Instruct"]
est, blo, bhi = [-0.26, 0.25, 0.94], [-1.66, -1.14, -0.46], [1.14, 1.65, 2.41]
yp = range(len(bn))
b.axvline(0, color=INK, linewidth=0.8, zorder=0)
b.errorbar(
    est,
    yp,
    xerr=[
        [estimate - lower for estimate, lower in zip(est, blo)],
        [h - e for e, h in zip(est, bhi)],
    ],
    fmt="D",
    color=INK,
    markersize=2.9,
    capsize=2,
    elinewidth=0.65,
    linestyle="none",
)
b.set_yticks(list(yp))
b.set_yticklabels(bn)
b.set_ylim(-0.6, 2.6)
b.invert_yaxis()
b.set_xlabel("Diff. in differences (pp)")
b.set_xlim(-2.7, 3.0)
b.xaxis.set_major_locator(MultipleLocator(2))
b.text(-0.04, 1.09, "(b)", transform=b.transAxes, fontweight="bold", fontsize=8)

# (c) containment gradient inside the train arm
bins = ["0.0\n0.1", "0.1\n0.3", "0.3\n0.5", "0.5\n0.8", "0.8\n1.0"]
rate = [3.69, 1.53, 0.87, 4.34, 4.12]
clo = [1.86, 0.31, 0.00, 2.53, 1.00]
chi = [5.87, 3.07, 1.96, 6.39, 8.11]
xi = range(len(bins))
c.errorbar(
    xi,
    rate,
    yerr=[
        [point - lower for point, lower in zip(rate, clo)],
        [h - r for r, h in zip(rate, chi)],
    ],
    fmt="o",
    color=BLUE,
    markersize=3.2,
    capsize=2,
    elinewidth=0.65,
    linestyle="none",
)
c.set_xticks(list(xi))
c.set_xticklabels(bins)
c.set_xlabel("Containment bin")
c.set_ylabel("Deletion recall (%)")
c.set_ylim(0, 9)
c.yaxis.set_major_locator(MultipleLocator(3))
c.text(-0.04, 1.09, "(c)", transform=c.transAxes, fontweight="bold", fontsize=8)

# (d) verbatim vs dose
dose = [0, 1, 4, 16, 64]
di = range(len(dose))
dd.plot(di, [13.3, 15.3, 18.0, 49.1, 94.6], color=INK, marker="o", markersize=3.2, linewidth=0.95)
dd.set_xticks(list(di))
dd.set_xticklabels([f"{v}x" for v in dose])
dd.set_xlabel("Injected repetitions")
dd.set_ylabel("Verbatim reprod. (%)")
dd.set_ylim(0, 104)
dd.yaxis.set_major_locator(MultipleLocator(25))
dd.text(-0.04, 1.09, "(d)", transform=dd.transAxes, fontweight="bold", fontsize=8)

# (e) procedure vs answer
ans, alo, ahi = (
    [0, 0.06, 0.09, -0.22, 1.52],
    [0, -3.06, -2.42, -3.66, -1.93],
    [0, 3.20, 2.70, 3.07, 5.08],
)
pro, plo, phi = (
    [0, 0.48, 0.32, 2.10, 4.76],
    [0, -2.47, -2.33, -1.27, 1.30],
    [0, 3.29, 3.01, 5.33, 8.23],
)
off = 0.12
e.axhline(0, color=GREY, linewidth=0.7, zorder=0)
e.errorbar(
    [i - off for i in di],
    pro,
    yerr=[
        [point - lower for point, lower in zip(pro, plo)],
        [h - p for p, h in zip(pro, phi)],
    ],
    fmt="o",
    color=BLUE,
    markersize=3.2,
    capsize=2,
    elinewidth=0.65,
    linestyle="none",
    label="Procedure",
)
e.errorbar(
    [i + off for i in di],
    ans,
    yerr=[
        [point - lower for point, lower in zip(ans, alo)],
        [h - v for v, h in zip(ans, ahi)],
    ],
    fmt="s",
    color=RED,
    markersize=3.2,
    capsize=2,
    elinewidth=0.65,
    linestyle="none",
    label="Answer",
)
e.set_xticks(list(di))
e.set_xticklabels([f"{v}x" for v in dose])
e.set_xlabel("Injected repetitions")
e.set_ylabel("Inj. minus held out (pp)")
e.set_ylim(-4.8, 9.4)
e.yaxis.set_major_locator(MultipleLocator(4))
e.legend(loc="upper left", handlelength=1.2)
e.text(-0.04, 1.09, "(e)", transform=e.transAxes, fontweight="bold", fontsize=8)

# (f) crossed 2x2: does the inserted number matter once length and topicality are fixed?
names = ["off-topic\nno num.", "domain\nno num.", "off-topic\n+number", "domain\n+number"]
s2   = [-1.47, -1.76, -2.44, -1.98]
s2lo = [-3.29, -3.52, -4.31, -3.80]
s2hi = [+0.34, +0.00, -0.68, -0.17]
ins   = [-1.76, -1.81, -1.81, -2.61]
inlo  = [-3.46, -3.52, -3.52, -4.31]
inhi  = [-0.06, -0.11, -0.11, -0.85]
pi = range(len(names)); off = 0.13
f.axhline(0, color=GREY, linewidth=0.7, zorder=0)
f.errorbar([i-off for i in pi], s2,
           yerr=[[v-l for v,l in zip(s2,s2lo)],[h-v for v,h in zip(s2,s2hi)]],
           fmt="o", color=BLUE, markersize=3.2, capsize=2, elinewidth=0.65,
           linestyle="none", label="Instella 3B")
f.errorbar([i+off for i in pi], ins,
           yerr=[[v-l for v,l in zip(ins,inlo)],[h-v for v,h in zip(ins,inhi)]],
           fmt="s", color=RED, markersize=3.2, capsize=2, elinewidth=0.65,
           linestyle="none", label="Instruct")
f.set_xticks(list(pi)); f.set_xticklabels(names, fontsize=5.6)
f.set_ylabel("Accuracy vs original (pp)"); f.set_ylim(-5.6, 1.8)
f.yaxis.set_major_locator(MultipleLocator(2))
f.legend(loc="lower left", handlelength=1.2, fontsize=6.4)
f.text(-0.04, 1.09, "(f)", transform=f.transAxes, fontweight="bold", fontsize=8)

fig.subplots_adjust(wspace=0.52, hspace=0.78)
for ext in ("pdf", "png"):
    fig.savefig(f"mathai_fig_combined.{ext}")
print("wrote mathai_fig_combined (6 panels)")
