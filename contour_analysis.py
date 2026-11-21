#!/usr/bin/env python3
"""
Contour analysis of symmetric vs asymmetric policies from dataSet.xlsx.

Expected Excel columns (column-name whitespace is ignored):
    mu1, mu2, alpha, runs,
    Symmetric sub opt, Asymmetric sub opt

For every alpha, this script creates:
  1) A three-panel contour figure in (mu1, mu2):
       symmetric suboptimal count, asymmetric suboptimal count,
       and (asymmetric - symmetric) suboptimal-count difference.
  2) The same three panels in (mu1/mu2, mu2) coordinates.

Interpretation of the difference panel:
    positive: asymmetric has MORE suboptimal outcomes;
              symmetric performs better on this measure.
    negative: asymmetric has FEWER suboptimal outcomes;
              asymmetric performs better on this measure.

The data points are irregular/scattered rather than a complete rectangular
grid, so Matplotlib's triangular contouring is used. The contours interpolate
between observed parameter pairs; they do not create new observations.
"""

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def safe_name(value):
    """Convert an alpha value to a filename-safe string."""
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", f"{value:g}")


def add_scattered_contour(ax, x, y, z, title, xlabel, ylabel,
                          cmap="viridis", diverging=False):
    """
    Draw contours for scattered 2-D observations.
    Falls back to a scatter plot if there are too few non-collinear points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[valid], y[valid], z[valid]

    # Average duplicate coordinates, if any.
    points = pd.DataFrame({"x": x, "y": y, "z": z})
    points = points.groupby(["x", "y"], as_index=False)["z"].mean()
    x, y, z = points["x"].to_numpy(), points["y"].to_numpy(), points["z"].to_numpy()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if len(x) < 3 or np.linalg.matrix_rank(
        np.column_stack([x - x.mean(), y - y.mean()])
    ) < 2:
        sc = ax.scatter(x, y, c=z, cmap=cmap, s=65, edgecolors="black")
        for xi, yi, zi in zip(x, y, z):
            ax.annotate(f"{zi:.0f}", (xi, yi), xytext=(4, 4),
                        textcoords="offset points", fontsize=7)
        plt.colorbar(sc, ax=ax, label="Observed value")
        ax.text(0.03, 0.97, "Too few non-collinear points for contours",
                transform=ax.transAxes, va="top", fontsize=8)
        return

    triangulation = mtri.Triangulation(x, y)

    # Mask very long triangles. This reduces misleading interpolation across
    # large gaps in an irregular parameter design.
    triangles = triangulation.triangles
    coords = np.column_stack([x, y])
    edges = np.concatenate([
        np.linalg.norm(coords[triangles[:, 0]] - coords[triangles[:, 1]], axis=1),
        np.linalg.norm(coords[triangles[:, 1]] - coords[triangles[:, 2]], axis=1),
        np.linalg.norm(coords[triangles[:, 2]] - coords[triangles[:, 0]], axis=1),
    ])
    typical_edge = np.median(edges) if len(edges) else np.inf
    max_edge = 2.5 * typical_edge
    tri_edges = np.stack([
        np.linalg.norm(coords[triangles[:, 0]] - coords[triangles[:, 1]], axis=1),
        np.linalg.norm(coords[triangles[:, 1]] - coords[triangles[:, 2]], axis=1),
        np.linalg.norm(coords[triangles[:, 2]] - coords[triangles[:, 0]], axis=1),
    ], axis=1)
    triangulation.set_mask(np.any(tri_edges > max_edge, axis=1))

    # if diverging:
    #     bound = np.nanmax(np.abs(z))
    #     if not np.isfinite(bound) or bound == 0:
    #         bound = 1.0
    #     levels = np.linspace(-bound, bound, 17)
    #     cf = ax.tricontourf(triangulation, z, levels=levels,
    #                         cmap="RdBu_r", extend="both")
    #     ax.tricontour(triangulation, z, levels=[0], colors="black",
    #                   linewidths=1.4)
    # else:
    #     zmin, zmax = np.nanmin(z), np.nanmax(z)
    #     if np.isclose(zmin, zmax):
    #         levels = np.linspace(zmin - 0.5, zmax + 0.5, 8)
    #     else:
    #         levels = np.linspace(zmin, zmax, 14)
    #     cf = ax.tricontourf(triangulation, z, levels=levels,
    #                         cmap=cmap, extend="both")
    #     ax.tricontour(triangulation, z, levels=levels[::2],
    #                   colors="black", linewidths=0.45, alpha=0.6)

    # Explicit contour levels
    levels = np.arange(1000, 10001, 1000)

    # Include the exact value we want to investigate
    levels = np.sort(np.unique(np.append(levels, 7564)))

    # Draw filled contours
    cf = ax.tricontourf(
        triangulation, z,
        levels=levels,
        cmap="viridis",
        extend="both"
    )

    # Draw contour lines
    cs = ax.tricontour(
        triangulation, z,
        levels=levels,
        colors="black",
        linewidths=0.8
    )

    # Label every contour with its actual value
    ax.clabel(
        cs,
        inline=True,
        fontsize=9,
        fmt=lambda value: f"{value:,.0f}"
    )

    # Show the actual observed values at each point
    for xi, yi, zi in zip(x, y, z):
        ax.annotate(
            f"{zi:,.0f}",
            (xi, yi),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.15",
                fc="white",
                alpha=0.75
            )
        )

    # plt.colorbar(cf, ax=ax, label="Suboptimal count")

    ax.scatter(x, y, c="black", s=13, alpha=0.65, label="Observed parameter pair")
    ax.legend(loc="best", fontsize=7)
    plt.colorbar(cf, ax=ax, label="Suboptimal outcomes (count)")


def make_figure(data, alpha, output_dir, coordinates="mu"):
    """Create one three-panel comparison figure for a fixed alpha."""
    if coordinates == "mu":
        x = data["mu1"].to_numpy()
        y = data["mu2"].to_numpy()
        xlabel, ylabel = r"$\mu_1$", r"$\mu_2$"
        prefix = "mu1_mu2"
    else:
        x = (data["mu1"] / data["mu2"]).to_numpy()
        y = data["mu2"].to_numpy()
        xlabel, ylabel = r"$\mu_1/\mu_2$", r"$\mu_2$"
        prefix = "ratio_mu2"

    sym = data["sym"].to_numpy(dtype=float)
    asym = data["asym"].to_numpy(dtype=float)
    difference = asym - sym

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    add_scattered_contour(
        axes[0], x, y, sym,
        "Symmetric policy: suboptimal outcomes", xlabel, ylabel
    )
    add_scattered_contour(
        axes[1], x, y, asym,
        "Asymmetric policy: suboptimal outcomes", xlabel, ylabel
    )
    add_scattered_contour(
        axes[2], x, y, difference,
        "Difference: asymmetric − symmetric", xlabel, ylabel,
        diverging=True
    )

    fig.suptitle(
        rf"Symmetric vs asymmetric policy, $\alpha={alpha:g}$",
        fontsize=15
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"{prefix}_alpha_{safe_name(alpha)}.png"
    fig.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return filename


def main():
    parser = argparse.ArgumentParser(
        description="Create symmetric/asymmetric policy contour plots."
    )
    parser.add_argument(
        "excel_file", nargs="?", default="dataSet.xlsx",
        help="Input Excel file (default: dataSet.xlsx)"
    )
    parser.add_argument(
        "--output", default="contour_plots",
        help="Directory for generated PNG figures (default: contour_plots)"
    )
    parser.add_argument(
        "--rate", action="store_true",
        help="Plot suboptimal fraction (count/runs) instead of raw counts."
    )
    args = parser.parse_args()

    excel_path = Path(args.excel_file)
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Cannot find {excel_path}. Pass the Excel file path as an argument."
        )

    df = pd.read_excel(excel_path)
    # Strip accidental leading/trailing spaces from Excel column headings.
    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "mu1", "mu2", "alpha", "runs",
        "Symmetric sub opt", "Asymmetric sub opt"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required Excel columns: {missing}")

    df = df[required].copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Raw suboptimal counts do not require the 'runs' field. In the supplied
    # workbook, runs is populated only for some rows, so do not discard the
    # other observations unless the user explicitly requests rates.
    core_columns = [
        "mu1", "mu2", "alpha",
        "Symmetric sub opt", "Asymmetric sub opt"
    ]
    df = df.dropna(subset=core_columns)

    if args.rate:
        df = df.dropna(subset=["runs"])
        if (df["runs"] <= 0).any():
            raise ValueError("All 'runs' values must be positive when using --rate.")
        df["sym"] = df["Symmetric sub opt"] / df["runs"]
        df["asym"] = df["Asymmetric sub opt"] / df["runs"]
        unit = "fraction"
    else:
        df["sym"] = df["Symmetric sub opt"]
        df["asym"] = df["Asymmetric sub opt"]
        unit = "count"

    output_dir = Path(args.output)
    alphas = sorted(df["alpha"].unique())

    print(f"Loaded {len(df)} rows.")
    print(f"Found {len(alphas)} alpha values and "
          f"{df[['mu1', 'mu2']].drop_duplicates().shape[0]} distinct (mu1, mu2) pairs.")
    print(f"Plotting suboptimal outcomes as {unit}.")

    generated = []
    for alpha in alphas:
        subset = df[np.isclose(df["alpha"], alpha)].copy()
        # One row per parameter pair is expected for each alpha.
        if subset.duplicated(["mu1", "mu2"]).any():
            subset = subset.groupby(["mu1", "mu2"], as_index=False).agg(
                sym=("sym", "mean"),
                asym=("asym", "mean")
            )
        else:
            subset = subset.rename(columns={"sym": "sym", "asym": "asym"})

        print("---------------------------------------\n")
        print(f"Processing alpha = {alpha}")
        print(subset)
        print("---------------------------------------\n")  
    
        generated.append(make_figure(subset, alpha, output_dir, "mu"))
        generated.append(make_figure(subset, alpha, output_dir, "ratio"))

    print(f"\nCreated {len(generated)} figures in: {output_dir.resolve()}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
