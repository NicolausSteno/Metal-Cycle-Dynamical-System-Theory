# Metal-Cycle-Dynamical-System-Theory

Code for the analytical solution and sensitivity analysis of the ODE model of
anthropogenic metal cycles, with Erlang product-lifetime distributions. It
accompanies **Chapter 4, "Dynamics of metal circularity: stability,
oscillations and resource conservation in the global copper cycle"**, of the
PhD thesis of Nicola Gambaro (Imperial College London), and its appendices.

The chapter treats the copper cycle as a linear system of ODEs,
ds/dt = A s + b, in which a single transfer matrix A gives both the
industrial-ecology indicators (technological lifetime, expected number of
uses) and the transient dynamics (convergence rate, damping, oscillation,
sensitivity). A second-order accelerator (SOA) in the mining sector
reproduces multi-decadal inventory cycles. The use phase is represented by a
fixed exponential, a mixed exponential or a chain of Erlang compartments,
fitted to the product lifetimes of Glöser et al. (2013).

## Contents

| File | What it does |
|---|---|
| `NumPy_Analytical_ODE_Solver.ipynb` | Python. Cell 1 solves the basic eight-stock system analytically by eigendecomposition and plots s₁–s₈. Cell 2 builds the coefficient matrix of the SOA system and computes its eigenvalues; cells 3–4 print the eigenvalues and eigenvectors. Cell 5 builds the SOA model with an exponential, mixed-exponential or Erlang in-use stock, solves it by matrix exponential and draws the lifetime-comparison figures. |
| `sens_analysis.nb` | Mathematica 13.3. Dynamic sensitivity analysis of the basic and SOA systems: preliminaries, sensitivity analysis and the SOA system, with the tables reported in the chapter and its appendices. |
| `distributions.py` | Python. Reconstructs the aggregate and sectoral lifetime distributions of Glöser et al. (2013), fits the fixed-exponential, mixed-exponential and Erlang (n = 5, 33) representations, and writes the two distribution figures and the fit errors. |
| `subsector_weights_lifetimes.csv` | Sectoral lifetimes τ and σ (Glöser et al. 2013, SI Table S4) with the inflow weights used for the sectoral distribution |
| `reconstructed_use1985_shares.csv` | Broad-sector shares of copper use in 1985, digitised from Glöser et al. (2013), Figure 3 |
| `fit_errors.csv` | Integrated absolute error and total-variation error of each approximation (the chapter's table of fit errors) |

## Figures

| File | In the thesis | Produced by |
|---|---|---|
| `SOA_solution_lifetime_comparison.png` | Chapter 4: analytical solutions of the SOA model under three lifetime distributions | Notebook, cell 5 (saved there as `SOA_solution_lifetime_comparison_gloser_consistent.png`) |
| `aggregate_distribution.png` | Chapter 4: lifetime distributions, panel (a), aggregate | `distributions.py` (written as `gloser_distribution_outputs/aggregate_distribution_journal_ready.png`) |
| `sectoral_distribution.png` | Chapter 4: lifetime distributions, panel (b), sectoral | `distributions.py` (written as `gloser_distribution_outputs/sectoral_distribution_journal_ready.png`) |
| `SOA_solution_lifetime_comparison_with_mixed_exponential.png` | Not used; supplementary version that adds the sectoral mixed exponential | Notebook, cell 5 |

The three figures used by the thesis are identical to the copies in the thesis
source.

## Requirements and use

- **Python 3.12** with NumPy, SciPy, pandas and matplotlib. The code was last
  run with Python 3.12.7, NumPy 2.4.1, SciPy 1.17.0, pandas 2.3.3 and
  matplotlib 3.10.8. `distributions.py` uses `np.trapezoid` and falls back to
  `np.trapz` on older NumPy versions.
- **Wolfram Mathematica 13.3** (or later) for `sens_analysis.nb`.

`distributions.py` writes its outputs to `gloser_distribution_outputs/` (and a
zip of that folder) under the current working directory, so run it from a
scratch directory:

```sh
mkdir -p /tmp/lifetimes && cd /tmp/lifetimes
python /path/to/Metal-Cycle-Dynamical-System-Theory/distributions.py
```

Open the notebook with Jupyter and run the cells in order; cell 5 saves its
figures to the working directory.

## Data

The lifetime parameters and use shares are taken or digitised from Glöser, S.,
Soulier, M. and Tercero Espinoza, L. A. (2013), Dynamic analysis of global
copper flows: global stocks, postconsumer material flows, recycling
indicators, and uncertainty evaluation, *Environmental Science & Technology*
47(12), 6564–6572, <https://doi.org/10.1021/es400069b>, and its Supporting
Information. Please cite the original source when reusing them.

## Licence

Code: PolyForm Noncommercial License 1.0.0. Figures, results and
documentation: CC BY-NC 4.0, the licence of the thesis. Research, teaching and
other non-commercial use is free; commercial use needs a separate licence from
the author. Third-party data stay under their owners' terms. See
[`LICENSE`](LICENSE). Releases before this licence was added (v1 and v2 on
Zenodo) carried no licence.

## Citation

Gambaro, N. (2026). NicolausSteno/Metal-Cycle-Dynamical-System-Theory: Code for
analytical solution and sensitivity analysis of ODE model for anthropogenic
metal cycles: Erlang product lifetime distributions (Version v2). Zenodo.
<https://doi.org/10.5281/zenodo.23188406>

Please also cite the thesis chapter.
