# galaxy_escape_circular

Reproduction code for Rodriguez (2026), "The Escape-to-Circular Velocity Ratio in SPARC Galaxies", submitted for review May 2026.

##

Given the public SPARC radial acceleration relation dataset (Lelli et al. 2017), this code computes the ratio of Newtonian escape velocity to observed circular velocity,

```
eta = v_esc / v_circ = sqrt(2 * g_bar / g_obs)
```

across all 2,693 individual measurements in 153 galaxies, and reproduces the numerical results, table, figure, and robustness checks in the paper.

## Quickstart

```
git clone https://github.com/amr28693/galaxy_escape_circular
cd galaxy_escape_circular
pip install -r requirements.txt
python rar_local.py
python rar_errors.py
python rar_robustness.py
python rar_figure.py
python rar_large_block_jackknife.py
```

Each script is independent. Each prints its full output to stdout and writes a results file. NumPy is the only dependency for the analysis scripts; Matplotlib is required only for the figure.

## Headline results

- Empirical eta=1 crossing: `g_bar/a_0 = 0.477 +/- 0.036` (combined bootstrap and Monte Carlo uncertainty)
- Analytic value from RAR interpolation function: `(ln 2)^2 = 0.4805`
- Match: 0.7% (0.1 sigma)
- High-acceleration bin (g_bar/a_0 ~ 11): median eta = 1.414 (= sqrt(2) to four sig figs)
- Robust across bin widths 0.15-0.40 dex, bin offsets +/-0.10 dex, and 153-chunk jackknife
- Among RAR interpolation functions, the McGaugh exponential is consistent with the empirical crossing at 0.1 sigma; the simple form (1+sqrt(1+4/y))/2 at 0.6 sigma; the standard MOND form x/sqrt(1+x^2) is excluded at 5.2 sigma

## Files

- `rar_local.py` -- central analysis: bins the data, computes the ratio, finds the eta=1 crossing, reports counts above and below the threshold. Produces `rar_results.txt`.
- `rar_errors.py` -- bootstrap (10,000 resamples) and Monte Carlo (10,000 perturbations within errors). Produces `rar_error_analysis.txt`.
- `rar_robustness.py` -- sensitivity analysis: bin width, bin offset, chunk jackknife, alternative interpolation functions. Produces `rar_robustness_results.json`.
- `rar_figure.py` -- generates Fig. 1 of the paper. Produces `fig_eta_crossover.pdf` and `fig_eta_crossover.png`.
- `rar_galaxies.txt` -- SPARC RAR data file from Lelli et al. (2017). Place this in the working directory.
- `terminal_output.txt` -- example terminal output after successful completion of the *Quickstart* pipeline
- `rar_large_block_jackknife.py` -- large-block chunk jackknife (block sizes 17, 50, 100) testing sensitivity to galaxy-level covariance in distance, inclination, and mass-to-light ratio.

## Data

`rar_galaxies.txt` is the SPARC Radial Acceleration Relation "All Data" file, downloaded from [https://astroweb.case.edu/SPARC/RAR.mrt](https://astroweb.case.edu/SPARC/RAR.mrt) (Lelli et al. 2017; McGaugh et al. 2016) and renamed from `RAR.mrt` to `rar_galaxies.txt`. The file is otherwise unmodified. Four whitespace-separated columns: `log10(g_bar)`, `err_log10(g_bar)`, `log10(g_obs)`, `err_log10(g_obs)`, all in log10 of acceleration in m/s^2. 2,693 data points across 153 late-type galaxies.

## Dependencies

- NumPy
- Matplotlib (figure only)

## Citation

If you use this code, please cite:

> Rodriguez, A. M. (2026). The Escape-to-Circular Velocity Ratio in SPARC Galaxies. [Journal TBD].

## License

MIT.
