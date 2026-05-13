"""
Figure 1 for Rodriguez (2026): eta vs g_bar/a_0

Produces a publication-quality plot of the escape-to-circular velocity
ratio across the SPARC RAR dataset.

USAGE: python3 rar_figure.py
       python3 rar_figure.py path/to/rar_galaxies.txt

Outputs: fig_eta_crossover.pdf
Requires: numpy, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

def parse_rar_file(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line[0].isalpha() or line.startswith('=') or line.startswith('---'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    vals = [float(parts[i]) for i in range(4)]
                    if -14 < vals[0] < -6 and -14 < vals[2] < -6:
                        data.append(vals)
                except (ValueError, IndexError):
                    continue
    return np.array(data)

# ============================================================
# ============================================================
# LOAD
# ============================================================
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
filepath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, 'rar_galaxies.txt')

if not os.path.exists(filepath):
    print(f"ERROR: File not found: {filepath}")
    sys.exit(1)

data = parse_rar_file(filepath)
gbar_log = data[:, 0]
gobs_log = data[:, 2]

a0 = 1.2e-10
log_a0 = np.log10(a0)

gbar = 10**gbar_log
gobs = 10**gobs_log
eta = np.sqrt(2.0 * gbar / gobs)
gbar_over_a0 = 10**(gbar_log - log_a0)

# ============================================================
# ============================================================
# BIN
# ============================================================
# ============================================================
edges = np.arange(-12.5, -8.0, 0.25)
centers_ga0, medians, stds, counts = [], [], [], []

for i in range(len(edges) - 1):
    lo, hi = edges[i], edges[i + 1]
    mask = (gbar_log >= lo) & (gbar_log < hi)
    n = mask.sum()
    if n < 3:
        continue
    mid = (lo + hi) / 2
    ga0 = 10**(mid - log_a0)
    centers_ga0.append(ga0)
    medians.append(np.median(eta[mask]))
    stds.append(np.std(eta[mask]))
    counts.append(n)

centers_ga0 = np.array(centers_ga0)
medians = np.array(medians)
stds = np.array(stds)
counts = np.array(counts)

# Standard error on median ~ 1.253 * std / sqrt(N)
median_errs = 1.253 * stds / np.sqrt(counts)

LN2_SQ = np.log(2)**2

# ============================================================
# ============================================================
# PLOT
# ============================================================
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5))

# Individual points (faint)
ax.scatter(gbar_over_a0, eta, s=0.4, c='0.75', alpha=0.3, rasterized=True,
           zorder=1, linewidths=0)

# Binned medians with error bars
ax.errorbar(centers_ga0, medians, yerr=median_errs, fmt='o', ms=5,
            color='k', ecolor='0.4', elinewidth=0.8, capsize=2,
            zorder=3, label='Binned median')

# Reference lines
ax.axhline(np.sqrt(2), color='0.4', ls='--', lw=0.8, zorder=2)
ax.axhline(1.0, color='0.4', ls='-', lw=0.8, zorder=2)

# Labels for reference lines
ax.text(0.007, np.sqrt(2) + 0.03, r'$\eta = \sqrt{2}$',
        fontsize=9, color='0.3', va='bottom')
ax.text(0.007, 1.0 + 0.03, r'$\eta = 1$',
        fontsize=9, color='0.3', va='bottom')

# Mark the (ln 2)^2 crossing
ax.axvline(LN2_SQ, color='0.5', ls=':', lw=0.8, zorder=2)
ax.annotate(r'$(\ln 2)^2$', xy=(LN2_SQ, 1.0), xytext=(LN2_SQ * 3, 0.55),
            fontsize=9, color='0.3',
            arrowprops=dict(arrowstyle='->', color='0.5', lw=0.8))

ax.set_xscale('log')
ax.set_xlabel(r'$g_{\mathrm{bar}}\,/\,a_0$', fontsize=11)
ax.set_ylabel(r'$\eta = v_{\mathrm{esc}}\,/\,v_{\mathrm{circ}}$', fontsize=11)
ax.set_xlim(0.004, 60)
ax.set_ylim(0.15, 1.75)
ax.tick_params(which='both', direction='in', top=True, right=True)

outfile = os.path.join(script_dir, 'fig_eta_crossover.pdf')
fig.tight_layout()
fig.savefig(outfile, dpi=300, bbox_inches='tight')
print(f"Saved: {outfile}")

# Also save png for quick inspection
outpng = os.path.join(script_dir, 'fig_eta_crossover.png')
fig.savefig(outpng, dpi=150, bbox_inches='tight')
print(f"Saved: {outpng}")
