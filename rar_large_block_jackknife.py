"""
rar_large_block_jackknife.py
==============================
Large-block chunk jackknife to test sensitivity to galaxy-level covariance.
Uses the same RAR data file and crossover computation as the rest of the analytic pipeline.

Usage:
    python rar_large_block_jackknife.py
"""

import numpy as np

# ── Configuration (matches existing pipeline) ──
DATA_FILE = 'rar_galaxies.txt'  
A0 = 1.2e-10  # m/s^2
BIN_WIDTH = 0.25
BIN_START = -12.5
BIN_END = -8.0

# ── Load data ──
data = []
with open(DATA_FILE) as f:
    lines = f.readlines()
# Skip header lines (adjust if needed)
for line in lines:
    line = line.strip()
    if not line:
        continue
    parts = line.split()
    if len(parts) >= 4:
        try:
            lg_bar = float(parts[0])
            lg_obs = float(parts[2])
            # Quick check it's a data line (log values should be negative)
            if lg_bar < 0 and lg_obs < 0:
                data.append((lg_bar, lg_obs))
        except ValueError:
            continue

data = np.array(data)
log_gbar = data[:, 0]
log_gobs = data[:, 1]
gbar = 10**log_gbar
gobs = 10**log_gobs
eta = np.sqrt(2 * gbar / gobs)
N = len(eta)

print(f"Loaded {N} data points")

# ── Crossover function──
def compute_crossover(lg, et):
    edges = np.arange(BIN_START, BIN_END + BIN_WIDTH, BIN_WIDTH)
    centers, medians = [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (lg >= lo) & (lg < hi)
        if mask.sum() < 3:
            continue
        centers.append(0.5 * (lo + hi))
        medians.append(np.median(et[mask]))
    for j in range(len(medians) - 1):
        if medians[j] < 1.0 and medians[j + 1] >= 1.0:
            frac = (1.0 - medians[j]) / (medians[j + 1] - medians[j])
            log_cross = centers[j] + frac * (centers[j + 1] - centers[j])
            return 10**log_cross / A0
    return None

cross_full = compute_crossover(log_gbar, eta)
ln2sq = np.log(2)**2
print(f"Full-sample crossover: {cross_full:.4f}")
print(f"(ln 2)^2 = {ln2sq:.4f}")

# ── Chunk jackknife at multiple block sizes ──
print(f"\n{'Block size':>10s} {'N_blocks':>8s} {'Mean':>8s} {'σ_jack':>8s} {'Range':>20s}")
print("-" * 60)

for block_size in [17, 50, 100]:
    # Partition into contiguous blocks
    n_blocks = N // block_size
    if n_blocks < 3:
        continue

    crossovers = []
    for k in range(n_blocks):
        start = k * block_size
        end = start + block_size
        mask = np.ones(N, dtype=bool)
        mask[start:end] = False
        c = compute_crossover(log_gbar[mask], eta[mask])
        if c is not None:
            crossovers.append(c)

    crossovers = np.array(crossovers)
    n = len(crossovers)
    jmean = np.mean(crossovers)
    jvar = (n - 1) / n * np.sum((crossovers - jmean)**2)
    jstd = np.sqrt(jvar)

    print(f"{block_size:10d} {n:8d} {jmean:8.4f} {jstd:8.4f} "
          f"[{crossovers.min():.4f}, {crossovers.max():.4f}]")

# ──153-block version for comparison ──
block_size_153 = N // 153
crossovers_153 = []
for k in range(153):
    start = k * block_size_153
    end = start + block_size_153
    mask = np.ones(N, dtype=bool)
    mask[start:end] = False
    c = compute_crossover(log_gbar[mask], eta[mask])
    if c is not None:
        crossovers_153.append(c)

crossovers_153 = np.array(crossovers_153)
n153 = len(crossovers_153)
jmean_153 = np.mean(crossovers_153)
jvar_153 = (n153 - 1) / n153 * np.sum((crossovers_153 - jmean_153)**2)
jstd_153 = np.sqrt(jvar_153)

print(f"\n153-block (original): mean={jmean_153:.4f}, σ={jstd_153:.4f}")

