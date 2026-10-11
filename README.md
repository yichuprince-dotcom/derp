# DERP Code Collection

This directory contains code collected from the existing implementations for the paper Directed extended-range percolation (DERP) by Liu, W., Zeng, Y., Liu, X. and Bianconi, G., Physical Review E (2026)

If you use the included codes please cite

[1] Liu, W., Zeng, Y., Liu, X. and Bianconi, G., 2026. Directed extended-range percolation. Physical Review E (2026) arXiv preprint arXiv:2605.22646.

[2] Cirigliano, Lorenzo, Claudio Castellano, and Ginestra Bianconi. "General theory for extended-range percolation on simple and multiplex networks." Physical Review E 110, no. 3 (2024): 034302.

## Naming Conventions and File Descriptions

ER denotes Poisson random networks, and SF denotes networks with power-law degree distributions. UC denotes the uncorrelated model, MC denotes the maximally correlated model, and `realnet` denotes real-world networks. The MC designation describes the intended model; it does not imply that all historical simulation implementations have been verified to enforce exact equality between each node’s in-degree and out-degree.

| File | Description | Current Default Parameters |
|---|---|---|
| `er_uc_simulation_theory.c` | Simulation and theoretical calculations for uncorrelated ER networks | N=2000, c=4.0, R=3, 100 runs |
| `er_mc_simulation_theory.c` | Simulation and theoretical calculations for maximally correlated ER networks | N=1000, c=1.5, R=2/3/4, 100 runs |
| `sf_uc_simulation_theory.c` | Simulation and theoretical calculations for uncorrelated SF networks | N=20000, γ=2.2/2.5/2.8, R=2/3/4, 100 runs |
| `sf_mc_simulation_theory.c` | Simulation and theoretical calculations for maximally correlated SF networks | N=20000, γ=2.2/2.5/2.8, R=2/3/4, 100 runs |
| `realnet_simulation_er_theory.c` | Simulation on the Gnutella04 network and an ER theoretical approximation with the same mean degree | R=2, 100 runs |
| `realnet_decoupled_theory.c` | A decoupled theoretical approximation using the marginal in-degree and out-degree distributions of the real network | R=2 |
| `er_mc_critical_exponents.py` | Critical threshold, order parameters, and log-log fitting for maximally correlated ER networks | c=2.5, R=2, k_max=100; fits the first 20 points |
| `sf_uc_critical_exponents.py` | γ scan and extraction of maximum effective exponents for uncorrelated SF networks | γ=2.2–3.5, numerical step size 0.1, R=2, k_c=10000 |
| `sf_mc_critical_exponents.py` | Local effective exponent curves for maximally correlated SF networks | γ=2.5, R=3, k_c=10000 |

## Network Data

| File | Description |
|---|---|
| `gnutella04_edges_original.txt` | Original directed edge list with the original node IDs |
| `gnutella04_edges_reindexed.txt` | Default computational input, with node IDs remapped consecutively from 0 to 10875 |

Both files contain two integer columns without a header, representing the source and target of each directed edge. The network contains 39,994 edges and 10,876 distinct nodes. Renaming did not change any bytes in the data files. The largest original node ID is 10878, with gaps in the numbering; the programs read the reindexed version by default.

## Compilation and Execution

Each C file contains its own `main` function and must be compiled separately. A C11-compatible GCC compiler and the standard math library are required. The following examples use Windows PowerShell; first navigate to this directory:

```powershell
gcc -O2 -std=c11 er_uc_simulation_theory.c -o er_uc_simulation_theory.exe -lm
.\er_uc_simulation_theory.exe
```

Example for the real network:

```powershell
gcc -O2 -std=c11 realnet_decoupled_theory.c -o realnet_decoupled_theory.exe -lm
.\realnet_decoupled_theory.exe
```

The real-network programs look for `gnutella04_edges_reindexed.txt` in the current working directory. Run them from this directory, or place the input file in the directory from which they are executed. Each program writes its results to the current working directory. Repeating a run with the same parameters overwrites results with the same filenames.

The Python scripts require Python 3, NumPy, SciPy, and Matplotlib:

```powershell
python -m pip install -r requirements.txt
python er_mc_critical_exponents.py
python sf_uc_critical_exponents.py
python sf_mc_critical_exponents.py
```

Both SF scripts start computing when imported and should therefore be run as standalone programs. The γ scans and the C simulations with 100 default runs may take considerable time. Run settings can be adjusted through the parameter definitions at the beginning of each file or within `main`.

`sf_mc_simulation_theory.c` also accepts a random seed as its first command-line argument, for example, `.\sf_mc_simulation_theory.exe 12345`. Its default degree cutoff is `floor(sqrt(N))`, which can be configured through `DEGREE_CUTOFF`.

## Output Files

Existing output filenames have been retained for compatibility with the original data-processing workflow.

| Program | Main Outputs |
|---|---|
| ER UC C | `ER_SIM_c*_R*.txt`, `ER_THEORY_c*_R*.txt` |
| ER MC C | `ER_Pos_SIM_c*_R*.txt`, `ER_Pos_THEORY_c*_R*.txt` |
| SF UC C | `SF_SIM_G*_R*_kmin*.txt`, `SF_THEORY_G*_R*_kmin*.txt`, `comparison_pk_G*_R*_kmin*.txt` |
| SF MC C | `SF_Pos_SIM/THEORY/SOLVER/CONFIG_G*_R*_kmin*.txt` (four separate file types), `comparison_pk_SF_Pos_G*_R*_kmin*.txt` |
| Real-network simulation and ER approximation | `RealNet_SIM_R2.txt`, `RealNet_ER_THEORY_R2.txt` |
| Real-network decoupled theory | `RealNet_Decoupled_THEORY.txt` |
| ER MC Python | `er_pos_critical_data_c2.5_R2.txt`, `er_pos_critical_indices_c2.5_R2.png/.pdf` |
| SF UC Python | `results_gamma_scan/gamma_exponent_results.txt` and `gamma_vs_exponents_extended.png` in the same directory |
| SF MC Python | `mcpl_final_exponents_gamma_2.5_R_3.png`; no numerical text output is currently generated |

Data headers such as `# Columns`, hexadecimal color values, and console messages within source-code strings are part of the program’s runtime content. C directives such as `#include` and `#define` are preprocessor instructions. All of these have been retained.

## Provenance and Scope of Verification

The renamed real-network decoupled-theory program was executed successfully. It read 10,876 nodes and 39,994 edges and produced 101 rows and 8 columns of finite numerical values. The reindexed data correspond to the original data edge by edge, preserving edge order and direction.

The code retains numerical and reproducibility limitations identified in earlier checks. These algorithms were not modified during the cleanup:

- In the SF UC critical-exponent script, the parameters passed to `gammaincc` produce NaN values for γ≥3. The script also selects the maximum effective exponent rather than using a verified plateau-selection procedure.
- The SF MC critical-exponent script defaults to R=3, whereas Fig. 4 uses R=2. The reference exponent formulas contain a zero denominator at γ=3, and the squared-marginal expressions for the intersection order parameters in the correlated case still require verification.
- The intersection quantities returned by the ER MC critical-exponent script are not multiplied by p or 1−p. Their interpretation must therefore be checked against the definitions in the paper.
- All three critical-exponent scripts numerically solve theoretical equations; they do not generate finite random graphs and perform Monte Carlo simulations.
- Relay-distance initialization in the older C simulations, and degree changes caused by discarding invalid edges during graph construction in the older ER MC implementation, remain subject to further scientific validation.
- The ER theory and decoupled theory used for the real network are both approximations. The real-network simulation evaluates points only at intervals of 0.05, so other entries in the historical output may be NaN. The internal arrays in the decoupled-theory implementation require R to lie between 1 and 19.

These notes distinguish file organization from the validation of scientific computations.
