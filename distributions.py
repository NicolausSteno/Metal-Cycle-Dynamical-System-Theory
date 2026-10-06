"""
Create, fit, and plot lifetime/discard-age distributions for the Glöser comparison.

Outputs:
- aggregate_distribution_journal_ready.png / .pdf
- sectoral_distribution_journal_ready.png / .pdf
- reconstructed_use1985_shares.csv
- subsector_weights_lifetimes.csv
- fit_errors.csv
- summary.json

Assumptions:
- Fixed exponential uses alpha_9 + alpha_18 = 0.02 + 0.01 = 0.03.
- Aggregate Glöser distribution uses tau = 25 years and sigma = 0.175*tau.
- Sectoral Glöser distribution uses Table S4 lifetimes and reconstructed Figure 3 "Use 1985" shares.
- Broad Figure 3 shares are split equally across Table S4 subsectors within each broad sector.
"""

from pathlib import Path
import json
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gammaln


# ---------------------------------------------------------------------
# Output folder
# ---------------------------------------------------------------------

outdir = Path("gloser_distribution_outputs")
outdir.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Reconstructed Figure 3, "Use 1985" broad-sector shares
# ---------------------------------------------------------------------
# Approximate digitisation of the stacked bar in Glöser et al. Figure 3.
# Order: Consumer & Electronics, Transport, Industrial, Infrastructure,
# Building & Construction.

broad_names = [
    "Consumer & Electronics",
    "Transport",
    "Industrial",
    "Infrastructure",
    "Building & Construction",
]

# Approximate segment heights from Figure 3, "Use 1985".
pixel_heights = np.array([31, 11, 15, 15, 35], dtype=float)
broad_weights = pixel_heights / pixel_heights.sum()
broad_weight_map = dict(zip(broad_names, broad_weights))


# ---------------------------------------------------------------------
# Glöser et al. SI Table S4 sectoral lifetimes
# ---------------------------------------------------------------------

subsectors = [
    ("Building & Construction", "Plumbing", 40.0),
    ("Building & Construction", "Building Plant", 40.0),
    ("Building & Construction", "Architecture", 50.0),
    ("Building & Construction", "Communications", 30.0),
    ("Building & Construction", "Electrical Power", 40.0),

    ("Infrastructure", "Telecommunications", 30.0),
    ("Infrastructure", "Power Utility", 30.0),

    ("Industrial", "Electrical Industrial", 15.0),
    ("Industrial", "Non Electrical Industrial", 20.0),

    ("Transport", "Electrical Automotive", 12.0),
    ("Transport", "Non Electrical Automotive", 15.0),
    ("Transport", "Other Transport", 25.0),

    ("Consumer & Electronics", "Consumer", 8.0),
    ("Consumer & Electronics", "Cooling", 10.0),
    ("Consumer & Electronics", "Electronic", 5.0),
    ("Consumer & Electronics", "Diverse", 10.0),
]

df = pd.DataFrame(subsectors, columns=["broad_sector", "subsector", "tau"])

# Split each reconstructed broad-sector share equally across its subsectors.
subsector_counts = df.groupby("broad_sector").size().to_dict()
df["broad_weight"] = df["broad_sector"].map(broad_weight_map)
df["weight"] = df.apply(
    lambda r: r["broad_weight"] / subsector_counts[r["broad_sector"]],
    axis=1,
)

# Glöser SI Figure S17 "moderate" standard deviation case.
sigma_factor = 0.175
df["sigma"] = sigma_factor * df["tau"]


# ---------------------------------------------------------------------
# Distribution functions
# ---------------------------------------------------------------------

def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Gaussian lifetime density."""
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


def exp_pdf(x: np.ndarray, rate: float) -> np.ndarray:
    """Exponential lifetime density."""
    return rate * np.exp(-rate * x)


def erlang_pdf(x: np.ndarray, m: int, k: float) -> np.ndarray:
    """
    Erlang density with integer shape m and rate k.

    g(a) = k^m a^(m-1) exp(-ka) / (m-1)!
    """
    x_safe = np.maximum(x, 1e-300)
    return np.exp(
        m * np.log(k)
        + (m - 1) * np.log(x_safe)
        - k * x_safe
        - gammaln(m)
    )


def integrated_absolute_error(
    approx: np.ndarray,
    target: np.ndarray,
    x: np.ndarray,
) -> float:
    """
    Integrated absolute error between two density curves.

    IAE = integral |approx(a) - target(a)| da

    This is dimensionless. IAE/2 is the total variation distance.
    """
    # np.trapz was removed in NumPy 2.x; np.trapezoid is its replacement.
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(np.abs(approx - target), x))


# ---------------------------------------------------------------------
# Common assumptions
# ---------------------------------------------------------------------

# Evaluation grid. The IAE is defined over [0, inf); the grid runs to 500 years
# so the slow exponential tails are integrated in full (spacing 0.025 yr, as
# before). The figures still show only 0-60 years via xlim.
x = np.linspace(0, 500, 20001)

alpha_9 = 0.02
alpha_18 = 0.01
alpha_total = alpha_9 + alpha_18

# Fixed exponential residence-time density.
fixed_exponential = exp_pdf(x, alpha_total)

# Aggregate Glöser reference distribution.
tau_agg = 25.0
sigma_agg = sigma_factor * tau_agg
gloser_aggregate = normal_pdf(x, tau_agg, sigma_agg)

# Erlang m=5: intermediate approximation.
m_mid = 5
k_mid_agg = m_mid / tau_agg
erlang_mid_aggregate = erlang_pdf(x, m_mid, k_mid_agg)

# Erlang m=33: best approximation by matching CV = sigma/tau = 0.175.
# For Erlang, CV = 1/sqrt(n), so n ~ (tau/sigma)^2 = 1/0.175^2 ~ 32.65.
m_best = int(round((tau_agg / sigma_agg) ** 2))
k_best_agg = m_best / tau_agg
erlang_best_aggregate = erlang_pdf(x, m_best, k_best_agg)


# ---------------------------------------------------------------------
# Sectoral distributions
# ---------------------------------------------------------------------

gloser_sectoral = np.zeros_like(x)
mixed_exponential_sectoral = np.zeros_like(x)
erlang_mid_sectoral = np.zeros_like(x)
erlang_best_sectoral = np.zeros_like(x)

for _, row in df.iterrows():
    w_i = float(row["weight"])
    tau_i = float(row["tau"])
    sigma_i = float(row["sigma"])

    # Glöser sectoral Gaussian mixture.
    gloser_sectoral += w_i * normal_pdf(x, tau_i, sigma_i)

    # Sectoral mixed exponential:
    # g(l) = sum_z w_z mu_z exp(-mu_z l),
    # mu_z = alpha_{9,z} + alpha_{18,z} = 1/tau_z (total exit rate of sector z).
    mixed_exponential_sectoral += w_i * exp_pdf(x, 1.0 / tau_i)

    # Sectoral Erlang m=5:
    # g(l) = sum_z w_z Erlang(l; n_z=5, kappa_z=5/tau_z).
    erlang_mid_sectoral += w_i * erlang_pdf(x, m_mid, m_mid / tau_i)

    # Sectoral Erlang m=33:
    # g(l) = sum_z w_z Erlang(l; n_z=33, kappa_z=33/tau_z).
    erlang_best_sectoral += w_i * erlang_pdf(x, m_best, m_best / tau_i)


# Optional check: brute-force best sectoral Erlang m by IAE.
sectoral_erlang_scan = []
for m in range(1, 81):
    y = np.zeros_like(x)
    for _, row in df.iterrows():
        w_i = float(row["weight"])
        tau_i = float(row["tau"])
        y += w_i * erlang_pdf(x, m, m / tau_i)

    err = integrated_absolute_error(y, gloser_sectoral, x)
    sectoral_erlang_scan.append((m, err))

m_best_sectoral_by_iae, best_sectoral_iae = min(
    sectoral_erlang_scan,
    key=lambda t: t[1],
)


# ---------------------------------------------------------------------
# Save input tables and errors
# ---------------------------------------------------------------------

broad_share_table = pd.DataFrame({
    "broad_sector": broad_names,
    "reconstructed_share_use1985": broad_weights,
    "digitised_pixel_height": pixel_heights,
})
broad_share_table.to_csv(outdir / "reconstructed_use1985_shares.csv", index=False)

df[["broad_sector", "subsector", "tau", "sigma", "weight"]].to_csv(
    outdir / "subsector_weights_lifetimes.csv",
    index=False,
)

errors = pd.DataFrame([
    {
        "case": "aggregate",
        "approximation": "Fixed exponential",
        "IAE": integrated_absolute_error(
            fixed_exponential,
            gloser_aggregate,
            x,
        ),
        "total_variation_percent": 100.0
        * integrated_absolute_error(fixed_exponential, gloser_aggregate, x) / 2.0,
    },
    {
        "case": "aggregate",
        "approximation": "Erlang m=5",
        "IAE": integrated_absolute_error(
            erlang_mid_aggregate,
            gloser_aggregate,
            x,
        ),
        "total_variation_percent": 100.0
        * integrated_absolute_error(erlang_mid_aggregate, gloser_aggregate, x) / 2.0,
    },
    {
        "case": "aggregate",
        "approximation": f"Erlang m={m_best}",
        "IAE": integrated_absolute_error(
            erlang_best_aggregate,
            gloser_aggregate,
            x,
        ),
        "total_variation_percent": 100.0
        * integrated_absolute_error(erlang_best_aggregate, gloser_aggregate, x) / 2.0,
    },
    {
        "case": "sectoral",
        "approximation": "Fixed exponential",
        "IAE": integrated_absolute_error(
            fixed_exponential,
            gloser_sectoral,
            x,
        ),
        "total_variation_percent": 100.0
        * integrated_absolute_error(fixed_exponential, gloser_sectoral, x) / 2.0,
    },
    {
        "case": "sectoral",
        "approximation": "Mixed exponential",
        "IAE": integrated_absolute_error(
            mixed_exponential_sectoral,
            gloser_sectoral,
            x,
        ),
        "total_variation_percent": 100.0
        * integrated_absolute_error(
            mixed_exponential_sectoral,
            gloser_sectoral,
            x,
        ) / 2.0,
    },
    {
        "case": "sectoral",
        "approximation": "Erlang m=5",
        "IAE": integrated_absolute_error(
            erlang_mid_sectoral,
            gloser_sectoral,
            x,
        ),
        "total_variation_percent": 100.0
        * integrated_absolute_error(erlang_mid_sectoral, gloser_sectoral, x) / 2.0,
    },
    {
        "case": "sectoral",
        "approximation": f"Erlang m={m_best}",
        "IAE": integrated_absolute_error(
            erlang_best_sectoral,
            gloser_sectoral,
            x,
        ),
        "total_variation_percent": 100.0
        * integrated_absolute_error(erlang_best_sectoral, gloser_sectoral, x) / 2.0,
    },
])

errors.to_csv(outdir / "fit_errors.csv", index=False)

summary = {
    "fixed_exponential": {
        "alpha_9": alpha_9,
        "alpha_18": alpha_18,
        "alpha_9_plus_alpha_18": alpha_total,
        "mean_residence_time_years": 1.0 / alpha_total,
    },
    "aggregate_Gloser": {
        "tau": tau_agg,
        "sigma": sigma_agg,
    },
    "aggregate_Erlang_mid": {
        "m": m_mid,
        "k": k_mid_agg,
    },
    "aggregate_Erlang_best": {
        "m": m_best,
        "k": k_best_agg,
    },
    "sectoral_Erlang_mid": {
        "formula": "sum_z w_z * Erlang(l; n_z=5, kappa_z=5/tau_z)",
    },
    "sectoral_Erlang_best": {
        "formula": f"sum_z w_z * Erlang(l; n_z={m_best}, kappa_z={m_best}/tau_z)",
    },
    "sectoral_best_m_by_IAE_scan": {
        "m": int(m_best_sectoral_by_iae),
        "IAE": float(best_sectoral_iae),
        "note": (
            "The plotted best curve uses m=33, from matching the Glöser "
            "moderate CV. The IAE scan is included only as a diagnostic."
        ),
    },
    "reconstructed_broad_shares_use1985": {
        name: float(w) for name, w in zip(broad_names, broad_weights)
    },
}

(outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------
# Final plots
# ---------------------------------------------------------------------
# Colourblind-safe, publication-friendly palette.

COL = {
    "gloser": "#000000",   # black
    "fixed": "#E69F00",    # orange
    "mixexp": "#009E73",   # bluish green
    "erl_m5": "#D55E00",   # vermilion
    "erl_m33": "#0072B2",  # blue
}

legend_kwargs = dict(
    frameon=True,
    facecolor="white",
    framealpha=1.0,
    edgecolor="0.8",
    fontsize=9,
)

# Aggregate figure.
plt.figure(figsize=(8.2, 5.2))

plt.plot(
    x,
    gloser_aggregate,
    color=COL["gloser"],
    linewidth=2.6,
    linestyle="-",
    label=r"Glöser distribution, $\tau=25.0$, $\sigma=4.375$",
)

plt.plot(
    x,
    fixed_exponential,
    color=COL["fixed"],
    linewidth=2.0,
    linestyle="-",
    label=r"Fixed exponential, $\alpha_9+\alpha_{18}=0.03$",
)

plt.plot(
    x,
    erlang_mid_aggregate,
    color=COL["erl_m5"],
    linewidth=2.0,
    linestyle="-.",
    label=rf"Erlang, $n={m_mid}$, $\kappa={k_mid_agg:.2f}$",
)

plt.plot(
    x,
    erlang_best_aggregate,
    color=COL["erl_m33"],
    linewidth=2.3,
    linestyle="--",
    label=rf"Erlang, $n={m_best}$, $\kappa={k_best_agg:.2f}$",
)

plt.xlim(0, 60)
plt.ylim(bottom=0)
plt.xlabel(r"Lifetime / discard age, $\ell$ (years)")
plt.ylabel("Probability density")
plt.title("Aggregate distribution")
plt.legend(**legend_kwargs)
plt.tight_layout()

plt.savefig(outdir / "aggregate_distribution.png", dpi=400, bbox_inches="tight")
plt.close()


# Sectoral figure.
plt.figure(figsize=(9.2, 5.8))

plt.plot(
    x,
    gloser_sectoral,
    color=COL["gloser"],
    linewidth=2.6,
    linestyle="-",
    label="Glöser sectoral distribution",
)

plt.plot(
    x,
    fixed_exponential,
    color=COL["fixed"],
    linewidth=2.0,
    linestyle="-",
    label=r"Fixed exponential, $\alpha_9+\alpha_{18}=0.03$",
)

plt.plot(
    x,
    mixed_exponential_sectoral,
    color=COL["mixexp"],
    linewidth=2.0,
    linestyle=":",
    label=r"Mixed exponential, $w_z$, $\mu_z=1/\tau_z$",
)

plt.plot(
    x,
    erlang_mid_sectoral,
    color=COL["erl_m5"],
    linewidth=2.0,
    linestyle="-.",
    label=rf"Erlang, $n_z={m_mid}$, $\kappa_z=n_z/\tau_z$",
)

plt.plot(
    x,
    erlang_best_sectoral,
    color=COL["erl_m33"],
    linewidth=2.3,
    linestyle="--",
    label=rf"Erlang, $n_z={m_best}$, $\kappa_z=n_z/\tau_z$",
)

plt.xlim(0, 60)
plt.ylim(bottom=0)
plt.xlabel(r"Lifetime / discard age, $\ell$ (years)")
plt.ylabel("Probability density")
plt.title("Sectoral distribution")
plt.legend(
    frameon=True,
    facecolor="white",
    framealpha=1.0,
    edgecolor="0.8",
    fontsize=8.8,
)
plt.tight_layout()

plt.savefig(outdir / "sectoral_distribution.png", dpi=400, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------
# Zip outputs
# ---------------------------------------------------------------------

zip_path = Path("gloser_distribution_outputs.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in outdir.iterdir():
        z.write(p, arcname=p.name)

print(f"Saved outputs to: {outdir.resolve()}")
print(f"Zipped outputs to: {zip_path.resolve()}")
print(errors)
