#!/usr/bin/env python3

# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

# \file analyze_geant.py
# \brief Do standard TID and NIEL plots based on Geant4 fluence simulation
# \author Nicola Nicassio (nicola.nicassio@cern.ch)
# \author Rocco Liotino (rocco.liotino@cern.ch)

"""
Default layout (all paths relative to this script):
    geometry_points.csv
    analyze_geant.py
    Simulation_files/
        ALICE3.txt

Default scoring mesh matches the current full-geometry simulation:
    R = 0 ... 320 cm, 640 bins
    z = -500 ... +500 cm, 1000 bins
    nPhi = 1

The script:
  * reads the Geant4 scorers "dose" and "neq";
  * normalizes them per generated pp event;
  * scales to the requested integrated luminosity (default 18 fb^-1);
  * converts dose from Gy to rad for the full-statistics result;
  * overlays geometry_points.csv on the 2D maps;
  * makes Geant4-only TID and NIEL maps;
  * makes a display-only smoothed TID map;
  * makes mean TID vs R and mean NIEL vs R profiles, averaging over
    ALL z bins of the scoring mesh at each R;
  * writes summary and radial-profile CSV tables.

"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter


# ============================================================
# DEFAULTS — CURRENT ALICE3 FULL-GEOMETRY SIMULATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_GEANT_FILE = BASE_DIR / "Simulation_files" / "ALICE3.txt"
DEFAULT_GEOMETRY_FILE = BASE_DIR / "geometry_points.csv"

DEFAULT_PLOTS_DIR = BASE_DIR / "plots"
DEFAULT_TABLES_DIR = BASE_DIR / "tables"

DEFAULT_R_MIN = 0.0
DEFAULT_R_MAX = 320.0
DEFAULT_BINS_R = 640

DEFAULT_Z_MIN = -500.0
DEFAULT_Z_MAX = 500.0
DEFAULT_BINS_Z = 1000

DEFAULT_LUMI_FB_INV = 18.0
DEFAULT_PP_INELASTIC_XSEC_MB = 78.4

# Display-only TID outlier smoothing inherited from the notebook.
DEFAULT_TID_SMOOTH_KERNEL = 3
DEFAULT_TID_OUTLIER_FACTOR = 4.0


# ============================================================
# PLOT STYLE
# ============================================================

FIGSIZE = (12, 7)
FIG_DPI = 150
SAVE_DPI = 300

AXIS_LABEL_FONTSIZE = 24
TICK_LABEL_FONTSIZE = 20
TITLE_FONTSIZE = 20

COLORBAR_TITLE_FONTSIZE = 24
COLORBAR_TICK_FONTSIZE = 20
COLORBAR_EXPONENT_FONTSIZE = 20
COLORBAR_PAD = 0.005
COLORBAR_FRACTION = 0.050

GEOMETRY_MARKER_SIZE = 0.5
GEOMETRY_ALPHA = 0.75


# ============================================================
# HELPERS
# ============================================================

def resolve_from_base(path: str | Path) -> Path:
    """Resolve relative paths with respect to the script directory."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = BASE_DIR / p
    return p.resolve()


def read_geant_total_events(filename: Path) -> int:
    with filename.open("r") as f:
        first_line = f.readline().strip()

    match = re.match(
        r"#\s*Number of simulated events:\s*(\d+)\s*$",
        first_line,
    )

    if not match:
        raise RuntimeError(
            f"Could not read the number of simulated events from {filename}.\n"
            "Expected first line:\n"
            "  # Number of simulated events: N"
        )

    n_events = int(match.group(1))

    if n_events <= 0:
        raise RuntimeError(
            f"Invalid number of simulated events in {filename}: {n_events}"
        )

    return n_events


def read_geant_scorers(
    filename: Path,
    bins_r: int,
    bins_z: int,
    scorer_names: tuple[str, ...] = ("dose", "neq"),
) -> dict[str, np.ndarray]:
    """
    Read selected Geant4 scoring blocks.

    Expected data-row index order:
        i_z, i_phi, i_r, value, ..., ...

    The current scoring uses one phi bin.
    """
    arrays = {
        name: np.zeros((bins_r, bins_z), dtype=np.float64)
        for name in scorer_names
    }

    found = {name: False for name in scorer_names}
    rows = {name: 0 for name in scorer_names}

    current: str | None = None

    with filename.open("r") as f:
        for line_number, line in enumerate(f, 1):
            if line.startswith("# primitive scorer name:"):
                current = line.split(":", 1)[1].strip()

                if current in found:
                    found[current] = True

                continue

            if current not in arrays:
                continue

            if line.startswith("#") or not line.strip():
                continue

            fields = line.strip().split(",")

            if len(fields) != 6:
                continue

            try:
                i_z = int(fields[0])
                i_phi = int(fields[1])
                i_r = int(fields[2])
                value = float(fields[3])
            except ValueError as exc:
                raise RuntimeError(
                    f"Could not parse scorer row at {filename}:{line_number}"
                ) from exc

            if i_phi != 0:
                raise RuntimeError(
                    f"Expected one phi bin (i_phi=0), got i_phi={i_phi} "
                    f"at {filename}:{line_number}"
                )

            if not (0 <= i_r < bins_r):
                raise RuntimeError(
                    f"Radial bin index {i_r} is outside configured range "
                    f"0..{bins_r - 1}. Check --bins-r."
                )

            if not (0 <= i_z < bins_z):
                raise RuntimeError(
                    f"z bin index {i_z} is outside configured range "
                    f"0..{bins_z - 1}. Check --bins-z."
                )

            # += is safe also if a scorer dump contains repeated contributions.
            arrays[current][i_r, i_z] += value
            rows[current] += 1

    missing = [name for name, was_found in found.items() if not was_found]

    if missing:
        raise RuntimeError(
            "Missing Geant4 scorer block(s): " + ", ".join(missing)
        )

    for name in scorer_names:
        print(
            f"Read scorer '{name}': {rows[name]} data rows, "
            f"array shape {arrays[name].shape}"
        )

    return arrays


def load_geometry_points(filename: Path) -> np.ndarray:
    """
    Load geometry_points.csv.

    The original notebook used:
        column 0 -> R [cm]
        column 1 -> z [cm]

    If recognizable R/z column names are present, they are used.
    Otherwise the first two numeric columns are used in that order.
    """
    if not filename.is_file():
        raise FileNotFoundError(
            f"Geometry CSV not found: {filename}"
        )

    df = pd.read_csv(filename)

    if df.shape[1] < 2:
        raise RuntimeError(
            f"{filename} must contain at least two columns."
        )

    normalized = {
        str(column).strip().lower().replace(" ", "").replace("[cm]", ""): column
        for column in df.columns
    }

    r_column = None
    z_column = None

    for candidate in ("r", "radius", "r(cm)", "rcm"):
        key = candidate.replace(" ", "").replace("[cm]", "")
        if key in normalized:
            r_column = normalized[key]
            break

    for candidate in ("z", "z(cm)", "zcm"):
        key = candidate.replace(" ", "").replace("[cm]", "")
        if key in normalized:
            z_column = normalized[key]
            break

    if r_column is not None and z_column is not None:
        geometry = df[[r_column, z_column]].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(dtype=float)

    else:
        numeric = df.apply(pd.to_numeric, errors="coerce")

        usable_columns = [
            column
            for column in numeric.columns
            if numeric[column].notna().any()
        ]

        if len(usable_columns) < 2:
            raise RuntimeError(
                f"Could not identify two numeric geometry columns in {filename}."
            )

        geometry = numeric[usable_columns[:2]].to_numpy(dtype=float)

    geometry = geometry[np.all(np.isfinite(geometry), axis=1)]

    if len(geometry) == 0:
        raise RuntimeError(
            f"No valid geometry points found in {filename}."
        )

    return geometry


def make_mesh(
    r_min: float,
    r_max: float,
    bins_r: int,
    z_min: float,
    z_max: float,
    bins_z: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r_edges = np.linspace(r_min, r_max, bins_r + 1)
    z_edges = np.linspace(z_min, z_max, bins_z + 1)

    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    return r_edges, r_centers, z_edges, z_centers


def positive_lognorm(values: np.ndarray) -> LogNorm | None:
    positive = values[
        np.isfinite(values) & (values > 0)
    ]

    if len(positive) == 0:
        return None

    vmin = max(float(np.percentile(positive, 1.0)), 1e-300)
    vmax = float(np.percentile(positive, 99.5))

    if vmax <= vmin:
        vmax = float(np.max(positive))

    if vmax <= vmin:
        return None

    return LogNorm(vmin=vmin, vmax=vmax)


def smooth_tid_outliers_for_plot(
    hist: np.ndarray,
    kernel_size: int,
    outlier_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Display-only suppression of isolated HIGH TID spikes.

    The numerical arrays used for profiles/tables are never modified.
    """
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(
            "--tid-smooth-kernel must be an odd integer >= 3"
        )

    if outlier_factor <= 1:
        raise ValueError(
            "--tid-outlier-factor must be > 1"
        )

    arr = np.asarray(hist, dtype=np.float64)
    pad = kernel_size // 2

    padded = np.pad(
        arr,
        ((pad, pad), (pad, pad)),
        mode="edge",
    )

    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        (kernel_size, kernel_size),
    )

    local_median = np.nanmedian(
        windows,
        axis=(-2, -1),
    )

    smoothed = arr.copy()

    valid = (
        np.isfinite(arr)
        & np.isfinite(local_median)
        & (local_median > 0)
    )

    outliers = (
        valid
        & (arr > local_median * outlier_factor)
    )

    smoothed[outliers] = local_median[outliers]

    print(
        f"TID display smoothing: replaced "
        f"{np.count_nonzero(outliers)} isolated high bins "
        f"(kernel={kernel_size}x{kernel_size}, "
        f"factor>{outlier_factor:g})."
    )

    return smoothed, outliers


def validate_plot_range(
    mesh_min: float,
    mesh_max: float,
    plot_min: float,
    plot_max: float,
    name: str,
) -> None:
    if plot_min >= plot_max:
        raise ValueError(
            f"{name}: plotting minimum must be smaller than maximum."
        )

    if plot_min < mesh_min or plot_max > mesh_max:
        raise ValueError(
            f"{name}: plotting range [{plot_min}, {plot_max}] is outside "
            f"the scoring mesh [{mesh_min}, {mesh_max}]."
        )


def plot_map(
    hist: np.ndarray,
    r_edges: np.ndarray,
    z_edges: np.ndarray,
    geometry: np.ndarray,
    plot_r_min: float,
    plot_r_max: float,
    plot_z_min: float,
    plot_z_max: float,
    title: str,
    colorbar_label: str,
    output_file: Path,
) -> None:
    plt.figure(figsize=FIGSIZE, dpi=FIG_DPI)

    masked = np.ma.masked_less_equal(
        np.asarray(hist, dtype=float),
        0.0,
    )

    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    r_mask = (
        (r_centers >= plot_r_min)
        & (r_centers < plot_r_max)
    )
    z_mask = (
        (z_centers >= plot_z_min)
        & (z_centers < plot_z_max)
    )

    visible = hist[np.ix_(r_mask, z_mask)]
    norm = positive_lognorm(visible)

    if norm is None:
        raise RuntimeError(
            f"No positive values are available in the requested plotting "
            f"range for {output_file.name}."
        )

    mesh = plt.pcolormesh(
        z_edges,
        r_edges,
        masked,
        shading="auto",
        norm=norm,
        rasterized=True,
    )

    cbar = plt.colorbar(
        mesh,
        pad=COLORBAR_PAD,
        fraction=COLORBAR_FRACTION,
    )

    cbar.set_label(
        colorbar_label,
        fontsize=COLORBAR_TITLE_FONTSIZE,
    )

    # Label only full decades on the logarithmic colorbar: 10^n.
    # Intermediate logarithmic tick marks remain, but have no labels.
    cbar.locator = LogLocator(
        base=10.0,
        subs=(1.0,),
        numticks=100,
    )

    cbar.formatter = LogFormatterSciNotation(
        base=10.0,
        labelOnlyBase=True,
    )

    cbar.update_ticks()

    cbar.ax.yaxis.set_minor_locator(
        LogLocator(
            base=10.0,
            subs=np.arange(2.0, 10.0),
            numticks=100,
        )
    )
    cbar.ax.yaxis.set_minor_formatter(NullFormatter())

    cbar.ax.tick_params(
        axis="y",
        which="major",
        labelsize=COLORBAR_TICK_FONTSIZE,
    )
    cbar.ax.tick_params(
        axis="y",
        which="minor",
        labelsize=COLORBAR_TICK_FONTSIZE,
    )
    cbar.ax.yaxis.get_offset_text().set_fontsize(
        COLORBAR_EXPONENT_FONTSIZE
    )

    # geometry_points.csv convention:
    #   column 0 = R [cm]
    #   column 1 = z [cm]
    geometry_visible = (
        (geometry[:, 0] >= plot_r_min)
        & (geometry[:, 0] <= plot_r_max)
        & (geometry[:, 1] >= plot_z_min)
        & (geometry[:, 1] <= plot_z_max)
    )

    geo = geometry[geometry_visible]

    if len(geo):
        plt.plot(
            geo[:, 1],
            geo[:, 0],
            ".k",
            markersize=GEOMETRY_MARKER_SIZE,
            alpha=GEOMETRY_ALPHA,
        )

    plt.xlim(plot_z_min, plot_z_max)
    plt.ylim(plot_r_min, plot_r_max)

    plt.xlabel(
        "z [cm]",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    plt.ylabel(
        "R [cm]",
        fontsize=AXIS_LABEL_FONTSIZE,
    )

    plt.xticks(fontsize=TICK_LABEL_FONTSIZE)
    plt.yticks(fontsize=TICK_LABEL_FONTSIZE)

    plt.title(
        title,
        fontsize=TITLE_FONTSIZE,
    )

    plt.savefig(
        output_file,
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_file}")


def plot_radial_profile(
    r_centers: np.ndarray,
    values: np.ndarray,
    plot_r_min: float,
    plot_r_max: float,
    title: str,
    ylabel: str,
    output_file: Path,
    y_scale: str,
) -> None:
    mask = (
        (r_centers >= plot_r_min)
        & (r_centers < plot_r_max)
        & np.isfinite(values)
    )

    if y_scale == "log":
        mask &= values > 0

    if not np.any(mask):
        raise RuntimeError(
            f"No valid points available for {output_file.name}."
        )

    plt.figure(figsize=FIGSIZE, dpi=FIG_DPI)

    plt.plot(
        r_centers[mask],
        values[mask],
        linewidth=1.6,
    )

    if y_scale == "log":
        plt.yscale("log")

    plt.xlim(plot_r_min, plot_r_max)

    plt.xlabel(
        "R [cm]",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    plt.ylabel(
        ylabel,
        fontsize=AXIS_LABEL_FONTSIZE,
    )

    plt.xticks(fontsize=TICK_LABEL_FONTSIZE)
    plt.yticks(fontsize=TICK_LABEL_FONTSIZE)

    plt.grid(
        True,
        which="both",
        alpha=0.25,
    )

    plt.title(
        title,
        fontsize=TITLE_FONTSIZE,
    )

    plt.savefig(
        output_file,
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_file}")


# ============================================================
# MAIN ANALYSIS
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Geant4-only ALICE3 TID/NIEL analysis for ALICE3.txt"
        )
    )

    parser.add_argument(
        "--geant-file",
        default=str(DEFAULT_GEANT_FILE),
        help=(
            "Geant scoring file "
            f"(default: {DEFAULT_GEANT_FILE.relative_to(BASE_DIR)})"
        ),
    )

    parser.add_argument(
        "--geometry-file",
        default=str(DEFAULT_GEOMETRY_FILE),
        help="Geometry CSV (default: geometry_points.csv)",
    )

    parser.add_argument(
        "--plots-dir",
        default=str(DEFAULT_PLOTS_DIR),
        help="Output directory for plots (default: plots)",
    )

    parser.add_argument(
        "--tables-dir",
        default=str(DEFAULT_TABLES_DIR),
        help="Output directory for CSV tables (default: tables)",
    )

    # Full scoring mesh.
    parser.add_argument("--mesh-r-min", type=float, default=DEFAULT_R_MIN)
    parser.add_argument("--mesh-r-max", type=float, default=DEFAULT_R_MAX)
    parser.add_argument("--bins-r", type=int, default=DEFAULT_BINS_R)

    parser.add_argument("--mesh-z-min", type=float, default=DEFAULT_Z_MIN)
    parser.add_argument("--mesh-z-max", type=float, default=DEFAULT_Z_MAX)
    parser.add_argument("--bins-z", type=int, default=DEFAULT_BINS_Z)

    # Display range. None means use the full scoring mesh.
    parser.add_argument("--plot-r-min", type=float, default=None)
    parser.add_argument("--plot-r-max", type=float, default=None)
    parser.add_argument("--plot-z-min", type=float, default=None)
    parser.add_argument("--plot-z-max", type=float, default=None)

    # Physics normalization.
    parser.add_argument(
        "--lumi-fb",
        type=float,
        default=DEFAULT_LUMI_FB_INV,
        help=(
            "Integrated luminosity in fb^-1 "
            f"(default: {DEFAULT_LUMI_FB_INV:g})"
        ),
    )

    parser.add_argument(
        "--xsec-mb",
        type=float,
        default=DEFAULT_PP_INELASTIC_XSEC_MB,
        help=(
            "pp inelastic cross section in mb "
            f"(default: {DEFAULT_PP_INELASTIC_XSEC_MB:g})"
        ),
    )

    parser.add_argument(
        "--profile-scale",
        choices=("log", "linear"),
        default="log",
        help="Y-axis scale for the two radial-profile plots (default: log)",
    )

    parser.add_argument(
        "--tid-smooth-kernel",
        type=int,
        default=DEFAULT_TID_SMOOTH_KERNEL,
        help=(
            "Odd local-median kernel for the extra display-only "
            f"TID map (default: {DEFAULT_TID_SMOOTH_KERNEL})"
        ),
    )

    parser.add_argument(
        "--tid-outlier-factor",
        type=float,
        default=DEFAULT_TID_OUTLIER_FACTOR,
        help=(
            "Replace display-only TID bins above factor x local median "
            f"(default: {DEFAULT_TID_OUTLIER_FACTOR:g})"
        ),
    )

    parser.add_argument(
        "--no-tid-smoothing",
        action="store_true",
        help="Do not make the additional display-only smoothed TID map",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    geant_file = resolve_from_base(args.geant_file)
    geometry_file = resolve_from_base(args.geometry_file)
    plots_dir = resolve_from_base(args.plots_dir)
    tables_dir = resolve_from_base(args.tables_dir)

    if not geant_file.is_file():
        raise FileNotFoundError(
            f"Geant scoring file not found: {geant_file}"
        )

    if args.bins_r <= 0 or args.bins_z <= 0:
        raise ValueError("--bins-r and --bins-z must be > 0")

    if args.mesh_r_min >= args.mesh_r_max:
        raise ValueError("--mesh-r-min must be smaller than --mesh-r-max")

    if args.mesh_z_min >= args.mesh_z_max:
        raise ValueError("--mesh-z-min must be smaller than --mesh-z-max")

    if args.lumi_fb <= 0:
        raise ValueError("--lumi-fb must be > 0")

    if args.xsec_mb <= 0:
        raise ValueError("--xsec-mb must be > 0")

    plot_r_min = (
        args.mesh_r_min
        if args.plot_r_min is None
        else args.plot_r_min
    )
    plot_r_max = (
        args.mesh_r_max
        if args.plot_r_max is None
        else args.plot_r_max
    )
    plot_z_min = (
        args.mesh_z_min
        if args.plot_z_min is None
        else args.plot_z_min
    )
    plot_z_max = (
        args.mesh_z_max
        if args.plot_z_max is None
        else args.plot_z_max
    )

    validate_plot_range(
        args.mesh_r_min,
        args.mesh_r_max,
        plot_r_min,
        plot_r_max,
        "R",
    )
    validate_plot_range(
        args.mesh_z_min,
        args.mesh_z_max,
        plot_z_min,
        plot_z_max,
        "z",
    )

    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    n_events = read_geant_total_events(geant_file)

    geometry = load_geometry_points(geometry_file)

    scorers = read_geant_scorers(
        geant_file,
        bins_r=args.bins_r,
        bins_z=args.bins_z,
    )

    r_edges, r_centers, z_edges, z_centers = make_mesh(
        args.mesh_r_min,
        args.mesh_r_max,
        args.bins_r,
        args.mesh_z_min,
        args.mesh_z_max,
        args.bins_z,
    )

    dr = (args.mesh_r_max - args.mesh_r_min) / args.bins_r
    dz = (args.mesh_z_max - args.mesh_z_min) / args.bins_z

    # 1 fb^-1 = 1e15 barn^-1
    # 1 mb = 1e-3 barn
    expected_pp_collisions = (
        args.lumi_fb
        * 1.0e15
        * args.xsec_mb
        * 1.0e-3
    )

    print()
    print("=" * 72)
    print("ALICE3 GEANT4 TID/NIEL ANALYSIS")
    print("=" * 72)
    print(f"Geant file       : {geant_file}")
    print(f"Geometry CSV     : {geometry_file}")
    print(f"Simulated events : {n_events}")
    print()
    print(
        f"Scoring R mesh   : {args.mesh_r_min:g} .. "
        f"{args.mesh_r_max:g} cm, {args.bins_r} bins, dR={dr:g} cm"
    )
    print(
        f"Scoring z mesh   : {args.mesh_z_min:g} .. "
        f"{args.mesh_z_max:g} cm, {args.bins_z} bins, dz={dz:g} cm"
    )
    print(
        f"Plot R range     : {plot_r_min:g} .. {plot_r_max:g} cm"
    )
    print(
        f"Plot z range     : {plot_z_min:g} .. {plot_z_max:g} cm"
    )
    print()
    print(f"Integrated lumi  : {args.lumi_fb:g} fb^-1")
    print(f"pp cross section : {args.xsec_mb:g} mb")
    print(
        f"Expected pp collisions: {expected_pp_collisions:.6e}"
    )
    print()

    # --------------------------------------------------------
    # Per-event maps
    # --------------------------------------------------------
    dose_per_event_gy = scorers["dose"] / n_events
    niel_per_event = scorers["neq"] / n_events

    # --------------------------------------------------------
    # Full integrated luminosity
    # --------------------------------------------------------
    # doseDeposit scorer is configured in Gy.
    # Convert the integrated result to rad.
    tid_full_rad = (
        dose_per_event_gy
        * expected_pp_collisions
        * 100.0
    )

    niel_full = (
        niel_per_event
        * expected_pp_collisions
    )

    # --------------------------------------------------------
    # 2D maps
    # --------------------------------------------------------
    lumi_label = f"{args.lumi_fb:g} fb$^{{-1}}$"

    plot_map(
        tid_full_rad,
        r_edges,
        z_edges,
        geometry,
        plot_r_min,
        plot_r_max,
        plot_z_min,
        plot_z_max,
        f"Geant4 TID — {lumi_label}",
        "TID [rad]",
        plots_dir / "Geant4_TID_full_statistics.png",
    )

    plot_map(
        niel_full,
        r_edges,
        z_edges,
        geometry,
        plot_r_min,
        plot_r_max,
        plot_z_min,
        plot_z_max,
        f"Geant4 NIEL — {lumi_label}",
        r"NIEL [1-MeV $n_{\mathrm{eq}}$ cm$^{-2}$]",
        plots_dir / "Geant4_NIEL_full_statistics.png",
    )

    if not args.no_tid_smoothing:
        tid_smooth, _ = smooth_tid_outliers_for_plot(
            tid_full_rad,
            args.tid_smooth_kernel,
            args.tid_outlier_factor,
        )

        plot_map(
            tid_smooth,
            r_edges,
            z_edges,
            geometry,
            plot_r_min,
            plot_r_max,
            plot_z_min,
            plot_z_max,
            f"Geant4 TID — {lumi_label} — smoothed display",
            "TID [rad]",
            plots_dir / "Geant4_TID_full_statistics_smoothed.png",
        )

    # --------------------------------------------------------
    # NEW 1D RADIAL PROFILES
    #
    # IMPORTANT:
    # At each R bin, average over ALL z bins of the FULL scoring mesh.
    # The plot-z range does not change these averages.
    # --------------------------------------------------------
    mean_tid_vs_r = np.nanmean(
        tid_full_rad,
        axis=1,
    )

    mean_niel_vs_r = np.nanmean(
        niel_full,
        axis=1,
    )

    plot_radial_profile(
        r_centers,
        mean_tid_vs_r,
        plot_r_min,
        plot_r_max,
        f"Mean Geant4 TID vs R — {lumi_label}\n"
        f"(mean over all z: {args.mesh_z_min:g} to {args.mesh_z_max:g} cm)",
        "Mean TID over z [rad]",
        plots_dir / "Geant4_TID_mean_vs_R.png",
        args.profile_scale,
    )

    plot_radial_profile(
        r_centers,
        mean_niel_vs_r,
        plot_r_min,
        plot_r_max,
        f"Mean Geant4 NIEL vs R — {lumi_label}\n"
        f"(mean over all z: {args.mesh_z_min:g} to {args.mesh_z_max:g} cm)",
        r"Mean NIEL over z [1-MeV $n_{\mathrm{eq}}$ cm$^{-2}$]",
        plots_dir / "Geant4_NIEL_mean_vs_R.png",
        args.profile_scale,
    )

    # --------------------------------------------------------
    # RADIAL-PROFILE TABLE
    # --------------------------------------------------------
    radial_table = pd.DataFrame({
        "R (cm)": r_centers,
        "Mean TID over all z (rad)": mean_tid_vs_r,
        "Mean NIEL over all z (1-MeV n_eq/cm^2)": mean_niel_vs_r,
    })

    radial_file = tables_dir / "Geant4_radial_profiles.csv"

    radial_table.to_csv(
        radial_file,
        index=False,
        float_format="%.8e",
    )

    print(f"Saved: {radial_file}")

    # --------------------------------------------------------
    # SUMMARY TABLE OVER THE DISPLAYED R,z REGION
    # --------------------------------------------------------
    r_mask = (
        (r_centers >= plot_r_min)
        & (r_centers < plot_r_max)
    )

    z_mask = (
        (z_centers >= plot_z_min)
        & (z_centers < plot_z_max)
    )

    if not np.any(r_mask) or not np.any(z_mask):
        raise RuntimeError(
            "The requested plotting range contains no scoring-bin centres."
        )

    tid_region = tid_full_rad[np.ix_(r_mask, z_mask)]
    niel_region = niel_full[np.ix_(r_mask, z_mask)]

    summary = pd.DataFrame([
        {
            "Quantity": "TID",
            "Integrated luminosity (fb^-1)": args.lumi_fb,
            "R min (cm)": plot_r_min,
            "R max (cm)": plot_r_max,
            "z min (cm)": plot_z_min,
            "z max (cm)": plot_z_max,
            "Mean": np.nanmean(tid_region),
            "Min": np.nanmin(tid_region),
            "Max": np.nanmax(tid_region),
            "Unit": "rad",
        },
        {
            "Quantity": "NIEL",
            "Integrated luminosity (fb^-1)": args.lumi_fb,
            "R min (cm)": plot_r_min,
            "R max (cm)": plot_r_max,
            "z min (cm)": plot_z_min,
            "z max (cm)": plot_z_max,
            "Mean": np.nanmean(niel_region),
            "Min": np.nanmin(niel_region),
            "Max": np.nanmax(niel_region),
            "Unit": "1-MeV n_eq/cm^2",
        },
    ])

    summary_file = tables_dir / "Geant4_summary.csv"

    summary.to_csv(
        summary_file,
        index=False,
        float_format="%.8e",
    )

    print(f"Saved: {summary_file}")

    # --------------------------------------------------------
    # MESH / RUN METADATA TABLE
    # --------------------------------------------------------
    metadata = pd.DataFrame([
        {
            "Geant file": str(geant_file),
            "Geometry file": str(geometry_file),
            "Simulated events": n_events,
            "Integrated luminosity (fb^-1)": args.lumi_fb,
            "pp inelastic cross section (mb)": args.xsec_mb,
            "Expected pp collisions": expected_pp_collisions,
            "R min (cm)": args.mesh_r_min,
            "R max (cm)": args.mesh_r_max,
            "R bins": args.bins_r,
            "dR (cm)": dr,
            "z min (cm)": args.mesh_z_min,
            "z max (cm)": args.mesh_z_max,
            "z bins": args.bins_z,
            "dz (cm)": dz,
            "Plot R min (cm)": plot_r_min,
            "Plot R max (cm)": plot_r_max,
            "Plot z min (cm)": plot_z_min,
            "Plot z max (cm)": plot_z_max,
        }
    ])

    metadata_file = tables_dir / "Geant4_analysis_metadata.csv"

    metadata.to_csv(
        metadata_file,
        index=False,
    )

    print(f"Saved: {metadata_file}")

    print()
    print("Analysis completed successfully.")
    print()
    print("Plots:")
    print(f"  {plots_dir / 'Geant4_TID_full_statistics.png'}")
    print(f"  {plots_dir / 'Geant4_NIEL_full_statistics.png'}")
    if not args.no_tid_smoothing:
        print(
            f"  {plots_dir / 'Geant4_TID_full_statistics_smoothed.png'}"
        )
    print(f"  {plots_dir / 'Geant4_TID_mean_vs_R.png'}")
    print(f"  {plots_dir / 'Geant4_NIEL_mean_vs_R.png'}")
    print()
    print("Tables:")
    print(f"  {radial_file}")
    print(f"  {summary_file}")
    print(f"  {metadata_file}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
