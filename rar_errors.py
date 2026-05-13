"""
Bootstrap + Monte Carlo Error Analysis for the Escape-to-Circular Velocity Ratio in SPARC Galaxies
Rodriguez 2026

Bootstrap: resamples the 2693 data points with replacement.
  Answers: "If we had a different sample of galaxies, would we get the same crossover?"

Monte Carlo: perturbs each data point within its measurement error bars.
  Answers: "Given the uncertainties on each measurement, how uncertain is the crossover?"

USAGE: python3 rar_errors.py
       python3 rar_errors.py path/to/rar_galaxies.txt
"""
import numpy as np
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

def find_crossover(gbar_log, gobs_log, bin_edges):
    """Compute binned virial ratio and find where it crosses 1.0"""
    gbar = 10**gbar_log
    gobs = 10**gobs_log
    v_ratio = np.sqrt(2.0 * gbar / gobs)
    
    centers, medians = [], []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (gbar_log >= lo) & (gbar_log < hi)
        if mask.sum() < 3:
            continue
        centers.append((lo + hi) / 2)
        medians.append(np.median(v_ratio[mask]))
    
    centers = np.array(centers)
    medians = np.array(medians)
    
    # Find crossover by linear interpolation
    for i in range(len(medians) - 1):
        if (medians[i] - 1.0) * (medians[i+1] - 1.0) < 0:
            frac = (1.0 - medians[i]) / (medians[i+1] - medians[i])
            cross_log = centers[i] + frac * (centers[i+1] - centers[i])
            a0 = 1.2e-10
            return 10**(cross_log - np.log10(a0))  # return g_bar/a0
    
    return np.nan

# ============================================================
# ============================================================
# LOAD DATA
# ============================================================
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
filepath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, 'rar_galaxies.txt')

if not os.path.exists(filepath):
    print(f"ERROR: File not found: {filepath}")
    sys.exit(1)

print(f"Reading: {filepath}")
data = parse_rar_file(filepath)
N = len(data)
print(f"Parsed {N} data points\n")

gbar_log = data[:, 0]
gbar_err = data[:, 1]
gobs_log = data[:, 2]
gobs_err = data[:, 3]

bin_edges = np.arange(-12.5, -8.0, 0.25)

# ============================================================
# ============================================================
# CENTRAL VALUE
# ============================================================
# ============================================================
central = find_crossover(gbar_log, gobs_log, bin_edges)
theory = np.log(2)**2
print(f"Central value: g_bar/a0 = {central:.4f}")
print(f"Theory (ln2)^2:         = {theory:.6f}")
print(f"Match: {abs(central - theory)/theory * 100:.1f}%")

# ============================================================
# ============================================================
# BOOTSTRAP
# ============================================================
# ============================================================
N_BOOT = 10000
print(f"\n{'='*60}")
print(f"BOOTSTRAP: {N_BOOT} resamples")
print(f"{'='*60}")

rng = np.random.default_rng(42)
boot_crossovers = []

for i in range(N_BOOT):
    # Resample with replacement
    idx = rng.integers(0, N, size=N)
    gb = gbar_log[idx]
    go = gobs_log[idx]
    cr = find_crossover(gb, go, bin_edges)
    if not np.isnan(cr):
        boot_crossovers.append(cr)

boot_crossovers = np.array(boot_crossovers)
boot_valid = len(boot_crossovers)

print(f"  Valid resamples: {boot_valid}/{N_BOOT}")
print(f"  Mean:   {np.mean(boot_crossovers):.4f}")
print(f"  Median: {np.median(boot_crossovers):.4f}")
print(f"  Std:    {np.std(boot_crossovers):.4f}")
print(f"  2.5%:   {np.percentile(boot_crossovers, 2.5):.4f}")
print(f"  97.5%:  {np.percentile(boot_crossovers, 97.5):.4f}")
print(f"  95% CI: [{np.percentile(boot_crossovers, 2.5):.4f}, {np.percentile(boot_crossovers, 97.5):.4f}]")

# Does the theoretical value fall within the 95% CI?
in_ci = np.percentile(boot_crossovers, 2.5) <= theory <= np.percentile(boot_crossovers, 97.5)
print(f"\n  (ln2)^2 = {theory:.4f} inside 95% CI? {'YES' if in_ci else 'NO'}")

# ============================================================
# ============================================================
# MONTE CARLO
# ============================================================
# ============================================================
N_MC = 10000
print(f"\n{'='*60}")
print(f"MONTE CARLO: {N_MC} perturbations within error bars")
print(f"{'='*60}")

mc_crossovers = []

for i in range(N_MC):
    # Perturb each point within its Gaussian error
    gb = gbar_log + rng.normal(0, gbar_err)
    go = gobs_log + rng.normal(0, gobs_err)
    cr = find_crossover(gb, go, bin_edges)
    if not np.isnan(cr):
        mc_crossovers.append(cr)

mc_crossovers = np.array(mc_crossovers)
mc_valid = len(mc_crossovers)

print(f"  Valid perturbations: {mc_valid}/{N_MC}")
print(f"  Mean:   {np.mean(mc_crossovers):.4f}")
print(f"  Median: {np.median(mc_crossovers):.4f}")
print(f"  Std:    {np.std(mc_crossovers):.4f}")
print(f"  2.5%:   {np.percentile(mc_crossovers, 2.5):.4f}")
print(f"  97.5%:  {np.percentile(mc_crossovers, 97.5):.4f}")
print(f"  95% CI: [{np.percentile(mc_crossovers, 2.5):.4f}, {np.percentile(mc_crossovers, 97.5):.4f}]")

in_ci_mc = np.percentile(mc_crossovers, 2.5) <= theory <= np.percentile(mc_crossovers, 97.5)
print(f"\n  (ln2)^2 = {theory:.4f} inside 95% CI? {'YES' if in_ci_mc else 'NO'}")

# ============================================================
# ============================================================
# COMBINED SUMMARY
# ============================================================
# ============================================================
# Total uncertainty: combine bootstrap and MC in quadrature
total_std = np.sqrt(np.std(boot_crossovers)**2 + np.std(mc_crossovers)**2)

print(f"\n{'='*60}")
print(f"COMBINED RESULTS")
print(f"{'='*60}")
print(f"  Central value:           {central:.4f}")
print(f"  Bootstrap uncertainty:   ±{np.std(boot_crossovers):.4f} (sampling)")
print(f"  Monte Carlo uncertainty: ±{np.std(mc_crossovers):.4f} (measurement)")
print(f"  Combined uncertainty:    ±{total_std:.4f} (quadrature)")
print(f"")
print(f"  Result: g_bar/a0 = {central:.3f} ± {total_std:.3f}")
print(f"  Theory: g_bar/a0 = {theory:.4f}")
print(f"")
print(f"  Deviation: {abs(central - theory)/total_std:.1f} sigma")
print(f"")

if abs(central - theory) / total_std < 2:
    print(f"  CONSISTENT with (ln2)^2 within {abs(central-theory)/total_std:.1f}σ")
else:
    print(f"  TENSION with (ln2)^2 at {abs(central-theory)/total_std:.1f}σ")

# ============================================================
# ============================================================
# FOR THE PAPER
# ============================================================
# ============================================================
print(f"\n{'='*60}")
print(f"FOR THE PAPER")
print(f"{'='*60}")
print(f"  Recommended citation of the result:")
print(f"")
print(f"  g_bar/a0 |_{{eta=1}} = {central:.3f} ± {total_std:.3f} (stat+meas)")
print(f"")
print(f"  compared to the analytic prediction")
print(f"")
print(f"  g_bar/a0 |_{{eta=1}} = (ln 2)^2 = {theory:.4f}")
print(f"")
print(f"  consistent to {abs(central-theory)/total_std:.1f}σ")

# Save
outfile = os.path.join(script_dir, 'rar_error_analysis.txt')
with open(outfile, 'w') as f:
    f.write(f"# RAR Error Analysis\n")
    f.write(f"central = {central:.6f}\n")
    f.write(f"theory = {theory:.6f}\n")
    f.write(f"boot_mean = {np.mean(boot_crossovers):.6f}\n")
    f.write(f"boot_std = {np.std(boot_crossovers):.6f}\n")
    f.write(f"mc_mean = {np.mean(mc_crossovers):.6f}\n")
    f.write(f"mc_std = {np.std(mc_crossovers):.6f}\n")
    f.write(f"combined_std = {total_std:.6f}\n")
    f.write(f"sigma_deviation = {abs(central-theory)/total_std:.2f}\n")
print(f"\nSaved: {outfile}")
