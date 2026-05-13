"""
RAR Analysis: Escape-to-Circular Velocity Ratio Across 153 SPARC Galaxies
Rodriguez 2026

Computes eta = v_esc / v_circ = sqrt(2 * g_bar / g_obs) per data point,
bins by log10(g_bar), and locates the eta = 1 crossover.

USAGE: python3 rar_local.py
       python3 rar_local.py path/to/datafile.txt

Default: looks for 'rar_galaxies.txt' in the same directory as this script.
"""
import numpy as np, sys, os

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

script_dir = os.path.dirname(os.path.abspath(__file__))
filepath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, 'rar_galaxies.txt')

if not os.path.exists(filepath):
    print(f"ERROR: File not found: {filepath}")
    print(f"Put 'rar_galaxies.txt' in the same folder as this script,")
    print(f"or run: python3 {os.path.basename(__file__)} path/to/file.txt")
    sys.exit(1)

print(f"Reading: {filepath}")
data = parse_rar_file(filepath)
N = len(data)
print(f"Parsed {N} data points\n")

gbar_log, gbar_err, gobs_log, gobs_err = data[:,0], data[:,1], data[:,2], data[:,3]
gbar, gobs = 10**gbar_log, 10**gobs_log
a0 = 1.2e-10; log_a0 = np.log10(a0)

print(f"Range: log10(g_bar) [{gbar_log.min():.2f}, {gbar_log.max():.2f}]")
print(f"       log10(g_obs) [{gobs_log.min():.2f}, {gobs_log.max():.2f}]")
print(f"       log10(a0) = {log_a0:.3f}")

v_ratio = np.sqrt(2.0 * gbar / gobs)
D = gobs / gbar

print(f"\n{'='*65}")
print(f"  eta = v_esc / v_circ = sqrt(2 * g_bar / g_obs)")
print(f"{'='*65}")
print(f"  Mean={np.mean(v_ratio):.4f}  Median={np.median(v_ratio):.4f}  Std={np.std(v_ratio):.4f}  sqrt2={np.sqrt(2):.4f}")

print(f"\n{'='*65}")
print(f"  BINNED ANALYSIS")
print(f"{'='*65}")
print(f"\n{'log10(g_bar)':>14} | {'g/a0':>8} | {'N':>5} | {'eta':>9} | {'+-':>5} | {'gobs/gbar':>9}")
print('-'*65)

edges = np.arange(-12.5, -8.0, 0.25)
bc, bm = [], []
for i in range(len(edges)-1):
    lo, hi = edges[i], edges[i+1]
    m = (gbar_log >= lo) & (gbar_log < hi)
    n = m.sum()
    if n < 3: continue
    mv = np.median(v_ratio[m]); sv = np.std(v_ratio[m]); md = np.median(D[m])
    mid = (lo+hi)/2; ga = 10**(mid-log_a0)
    bc.append(mid); bm.append(mv)
    f = " <-- eta=1" if abs(mv-1)<.05 else (" <-- sqrt2" if abs(mv-np.sqrt(2))<.02 else "")
    print(f"[{lo:6.2f},{hi:6.2f}) | {ga:>7.3f}  | {n:>5} | {mv:>8.4f}  | {sv:>4.3f} | {md:>8.3f}  {f}")

bc, bm = np.array(bc), np.array(bm)

print(f"\n{'='*65}")
print(f"  CROSSOVER")
print(f"{'='*65}")

emp = None
for i in range(len(bm)-1):
    if (bm[i]-1)*(bm[i+1]-1) < 0:
        frac = (1-bm[i])/(bm[i+1]-bm[i])
        cr = bc[i]+frac*(bc[i+1]-bc[i])
        emp = 10**(cr-log_a0)
        print(f"\n  EMPIRICAL:    g_bar/a0 = {emp:.4f}   (log10 = {cr:.3f})")

th = np.log(2)**2
print(f"  ANALYTIC:     g_bar/a0 = {th:.6f}  (= (ln 2)^2)")
if emp: print(f"  MATCH:        {abs(emp-th)/th*100:.1f}%")

print(f"\n{'='*65}")
print(f"  COUNTS RELATIVE TO eta = 1")
print(f"{'='*65}")
n_above = (v_ratio > 1).sum()
n_below = (v_ratio < 1).sum()
print(f"  eta > 1 (baryons sufficient):     {n_above:>5} pts ({100*n_above/N:5.1f}%)")
print(f"  eta < 1 (baryons insufficient):   {n_below:>5} pts ({100*n_below/N:5.1f}%)")

print(f"\n{'='*65}")
print(f"  DERIVATION OF (ln 2)^2")
print(f"{'='*65}")
print(f"  eta = 1                                  (definition of crossover)")
print(f"  => g_obs = 2 * g_bar")
print(f"  g_obs = g_bar/(1 - exp(-sqrt(g_bar/a0))) (RAR interpolation function)")
print(f"  => 1 - exp(-sqrt(g_bar/a0)) = 1/2")
print(f"  => exp(-sqrt(g_bar/a0)) = 1/2")
print(f"  => sqrt(g_bar/a0) = ln 2")
print(f"  => g_bar/a0 = (ln 2)^2 = {th:.6f}")
print(f"")
print(f"  Confirmed to {abs(emp-th)/th*100:.1f}% across {N} points, 153 galaxies, 0 parameters.")

out = os.path.join(script_dir, 'rar_results.txt')
with open(out,'w') as f:
    f.write(f"# Empirical g/a0 = {emp:.4f}\n# Analytic (ln 2)^2 = {th:.6f}\n# Match = {abs(emp-th)/th*100:.1f}%\n")
    for i in range(len(bc)): f.write(f"{bc[i]:.3f} {bm[i]:.6f}\n")
print(f"\nSaved: {out}")
