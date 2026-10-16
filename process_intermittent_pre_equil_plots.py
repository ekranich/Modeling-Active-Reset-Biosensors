#!/usr/bin/env python3
"""
ChatGPT Model 6 and Claude Sonnet 5.5 were used to assist with code development, clean-up, and documentation

This script does NOT rerun estimation
Primary input
-------------
The compiled interval CSV produced by biosensor_simulation_general.py, e.g.
    combined_random_intervals.csv

Optional full-run input
-----------------------
The final three-panel drift figure requires three saved full simulation .pkl
files. Provide the three files explicitly in DRIFT_PANEL_FILES.

Figures produced
----------------
1. example_target_profiles
2. sensor_response_all_profiles
3. active_reset_cv
4. active_reset_rrmse
5. no_reset_cv
6. no_reset_rrmse
7. signed_step_change_sensorXX   (one per sensor)
8. step_change_accuracy_vs_koff
9. drift_comparison              (if DRIFT_PANEL_FILES are supplied)
"""

from pathlib import Path
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d


# insert filename and paths
COMBINED_INTERVAL_CSV = Path("combined_random_intervals.csv")
RESULTS_DIR = Path("simulation_results")
FIGURE_DIR = Path("processed_intermittent_figures")

SAVE_PNG = True
SAVE_PDF = True
PNG_DPI = 600
SHOW_FIGURES = True

# Plot styling
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

COLOR_RESET = "#0072B2"
COLOR_NONRESET = "#D55E00"

COLORBLIND_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
]


# TARGET-PROFILE EXAMPLE SETTINGS
PROFILE_EXAMPLE_TOTAL_MIN = 500.0
PROFILE_EXAMPLE_DT_STEP_MIN = 30.0
PROFILE_EXAMPLE_MIN_PM = 1.0
PROFILE_EXAMPLE_MAX_PM = 900.0
PROFILE_EXAMPLE_SEED = 1

PERIODIC_BASELINE_PM = 100.0
PERIODIC_AMPLITUDE_PM = 50.0
PERIODIC_PERIOD_MIN = 150.0

SPIKE_T0_MIN = 25.0
SPIKE_AMPLITUDE_PM = 150.0
SPIKE_BASELINE_PM = 20.0
SPIKE_TAU_RISE_MIN = 10.0
SPIKE_TAU_FALL_MIN = 90.0
SPIKE_WIDTH_MIN = 40.0


# SENSOR-RESPONSE PROFILE GRID SETTINGS
RESPONSE_GRID_KON_LIST = [1e6]
RESPONSE_GRID_KOFF_LIST = [1e-2]

RESPONSE_GRID_DT_S = 0.5
RESPONSE_GRID_STEP_MIN = 10.0
RESPONSE_GRID_TOTAL_MIN = 200.0
RESPONSE_GRID_SEED = 3
RESPONSE_GRID_DFB = 0.05
RESPONSE_GRID_PLOT_ACTIVE_RESET = True

# CV / rRMSE SETTINGS
SPLINE_SMOOTHING_STRENGTH = 0.4
SPLINE_BAND_NEIGHBORS = 15
PERFORMANCE_THRESHOLD = 0.10

# STEP-CHANGE SETTINGS
STEP_CHANGE_ERROR_THRESHOLD = 0.10
STEP_CHANGE_XY_LIMIT_PM = 100.0


# DRIFT PANEL SETTINGS
# Supply three full-run .pkl paths in the order:
#   exponential, random walk, linear
# Leave empty to skip the drift panel.
DRIFT_PANEL_FILES = [
    Path("simulation_results/random_sensor00_run000.pkl"),
    Path("simulation_results/random_sensor01_run000.pkl"),
    Path("simulation_results/random_sensor02_run000.pkl"),
]

DRIFT_PANEL_TITLES = [
    r"Random Walk Drift, $k_{\mathrm{off}} = 10^{-2}\ \mathrm{s}^{-1}$",
    r"Random Walk Drift, $k_{\mathrm{off}} = 10^{-4}\ \mathrm{s}^{-1}$",
    r"Random Walk Drift, $k_{\mathrm{off}} = 10^{-6}\ \mathrm{s}^{-1}$",

]

# GENERAL HELPERS
def save_figure(fig, stem):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if SAVE_PNG:
        path = FIGURE_DIR / f"{stem}.png"
        fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
        print(f"Saved: {path}")

    if SAVE_PDF:
        path = FIGURE_DIR / f"{stem}.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def load_compiled_intervals():
    if not COMBINED_INTERVAL_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {COMBINED_INTERVAL_CSV.resolve()}"
        )

    df = pd.read_csv(COMBINED_INTERVAL_CSV)

    required = {
        "sensor_idx",
        "sensor_name",
        "run",
        "step",
        "true_pM",
        "est_reset_pM",
        "est_nonreset_pM",
        "kon",
        "koff",
        "Kd_pM",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Compiled interval file is missing required columns: "
            + ", ".join(missing)
        )

    if "sq_error_reset" not in df.columns:
        df["sq_error_reset"] = (
            df["est_reset_pM"] - df["true_pM"]
        ) ** 2

    if "sq_error_nonreset" not in df.columns:
        df["sq_error_nonreset"] = (
            df["est_nonreset_pM"] - df["true_pM"]
        ) ** 2

    return df

# TARGET PROFILE HELPERS
def periodic_profile(
    time_min,
    baseline=PERIODIC_BASELINE_PM,
    amplitude=PERIODIC_AMPLITUDE_PM,
    period_min=PERIODIC_PERIOD_MIN,
):
    time_min = np.asarray(time_min, dtype=float)
    return baseline + amplitude * np.sin(
        2 * np.pi * time_min / period_min
    )


def spike_profile(
    time_min,
    t0=SPIKE_T0_MIN,
    amplitude=SPIKE_AMPLITUDE_PM,
    baseline=SPIKE_BASELINE_PM,
    tau_rise=SPIKE_TAU_RISE_MIN,
    tau_fall=SPIKE_TAU_FALL_MIN,
    width=SPIKE_WIDTH_MIN,
):
    time_min = np.asarray(time_min, dtype=float)

    y = np.full_like(time_min, baseline, dtype=float)

    rise_mask = (
        (time_min >= t0)
        & (time_min < t0 + width)
    )

    y[rise_mask] = baseline + amplitude * (
        1.0
        - np.exp(
            -(time_min[rise_mask] - t0)
            / tau_rise
        )
    )

    peak_val = baseline + amplitude * (
        1.0 - np.exp(-width / tau_rise)
    )

    fall_mask = time_min >= (t0 + width)

    y[fall_mask] = baseline + (
        peak_val - baseline
    ) * np.exp(
        -(time_min[fall_mask] - (t0 + width))
        / tau_fall
    )

    return y


def make_target_profiles(
    time_min,
    dt_step=30.0,
    min_pM=1.0,
    max_pM=900.0,
    random_smooth_sigma_min=4.0,
    seed=1,
):
    """
    Build the three piecewise concentration profiles used by the notebook.

    Random:
        independent log-uniform step concentrations between min_pM and max_pM.

    Periodic / Spike:
        evaluated at each step center and then held constant over each
        sensing interval.
    """
    del random_smooth_sigma_min 

    time_min = np.asarray(time_min, dtype=float)

    total_time_min = float(time_min.max())

    n_steps = int(np.ceil(total_time_min / dt_step))

    step_edges = np.arange(n_steps + 1, dtype=float) * dt_step
    step_centers = step_edges[:-1] + dt_step / 2.0

    rng = np.random.default_rng(seed)

    random_values = 10 ** rng.uniform(
        np.log10(min_pM),
        np.log10(max_pM),
        size=n_steps,
    )

    random_values = np.clip(
        np.round(random_values),
        min_pM,
        max_pM,
    )

    periodic_values = periodic_profile(step_centers)
    spike_values = spike_profile(step_centers)

    step_values_dict = {
        "Random": np.asarray(random_values, dtype=float),
        "Periodic": np.asarray(periodic_values, dtype=float),
        "Spike": np.asarray(spike_values, dtype=float),
    }

    piecewise_profiles = {}

    step_index = np.floor(time_min / dt_step).astype(int)
    step_index = np.clip(step_index, 0, n_steps - 1)

    for name, values in step_values_dict.items():
        piecewise_profiles[name] = values[step_index]

    return (
        piecewise_profiles,
        step_edges,
        step_centers,
        step_values_dict,
    )


def make_step_list_from_step_values(step_values_pM, dt_step_min):
    return [
        (float(value) * 1e-12, float(dt_step_min))
        for value in step_values_pM
    ]


def fract_bound_calc(T, kon, koff, f0, dt, measure_time):
    measure_s = float(measure_time) * 60.0
    n = int(round(measure_s / dt))

    t = np.arange(n, dtype=float) * dt

    Kd = koff / kon
    f_eq = T / (Kd + T)
    tau = 1.0 / (kon * T + koff)

    return f_eq - (f_eq - f0) * np.exp(-t / tau)


def simulate_fraction_bound_piecewise(
    target_profile_list,
    kon,
    koff,
    dt=0.5,
    reset_each_step=False,
):
    fraction_bound = []
    f0 = 0.0

    for conc_M, duration_min in target_profile_list:
        f_step = fract_bound_calc(
            T=conc_M,
            kon=kon,
            koff=koff,
            f0=f0,
            dt=dt,
            measure_time=duration_min,
        )

        fraction_bound.extend(f_step.tolist())

        # Advance to the true interval boundary rather than using the last
        # sampled point at duration-dt.
        if reset_each_step:
            f0 = 0.0
        else:
            Kd = koff / kon
            f_eq = conc_M / (Kd + conc_M)
            tau = 1.0 / (kon * conc_M + koff)
            f0 = (
                f_eq
                - (f_eq - f0)
                * np.exp(-(duration_min * 60.0) / tau)
            )

    return np.asarray(fraction_bound, dtype=float)


def make_equilibrium_fraction_profile(
    target_profile_list,
    Kd,
    dt=0.5,
):
    out = []

    for conc_M, duration_min in target_profile_list:
        n = int(round(duration_min * 60.0 / dt))
        f_eq = conc_M / (Kd + conc_M)
        out.extend([f_eq] * n)

    return np.asarray(out, dtype=float)


def make_target_concentration_profile(
    target_profile_list,
    dt=0.5,
):
    out = []

    for conc_M, duration_min in target_profile_list:
        n = int(round(duration_min * 60.0 / dt))
        out.extend([conc_M] * n)

    return np.asarray(out, dtype=float)


# FIGURE 1: EXAMPLE TARGET PROFILES
def plot_example_target_profiles():
    style_inter = {
        "font.family": "Arial",
        "font.size": 15,
        "axes.labelsize": 10,
        "axes.titlesize": 5,
        "legend.fontsize": 5,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    }

    time = np.linspace(
        0,
        PROFILE_EXAMPLE_TOTAL_MIN,
        int(PROFILE_EXAMPLE_TOTAL_MIN) + 1,
    )

    (
        piecewise_profiles,
        _,
        _,
        _,
    ) = make_target_profiles(
        time,
        dt_step=PROFILE_EXAMPLE_DT_STEP_MIN,
        min_pM=PROFILE_EXAMPLE_MIN_PM,
        max_pM=PROFILE_EXAMPLE_MAX_PM,
        random_smooth_sigma_min=4,
        seed=PROFILE_EXAMPLE_SEED,
    )

    titles = ["Random", "Periodic", "Spike"]

    colors = {
        "Random": "royalblue",
        "Periodic": "limegreen",
        "Spike": "firebrick",
    }

    with plt.rc_context(style_inter):
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(8, 9),
            gridspec_kw={
                "wspace": 0.35,
                "hspace": 0.35,
            },
        )

        for ax, title in zip(axes, titles):
            ax.step(
                time,
                piecewise_profiles[title],
                where="post",
                linewidth=2.5,
                color=colors[title],
            )

            ax.set_xlim(time.min(), time.max())
            ax.set_xlabel("Time (min)")
            ax.set_ylabel(
                "Piecewise\nconcentration (pM)"
            )
            ax.set_title(title, fontsize=15)

    save_figure(fig, "example_target_profiles")


# FIGURE 2: SENSOR RESPONSE FOR ALL THREE PROFILES
def plot_all_sensors_all_profiles(
    step_values_dict,
    kon_list,
    koff_list,
    dt_step_min=30.0,
    dt=0.5,
    DFB=None,
    plot_active_reset=False,
):
    profile_names = ["Random", "Periodic", "Spike"]

    n_sensors = len(koff_list)
    n_profiles = len(profile_names)

    Kd_list = [
        koff / kon
        for kon, koff in zip(kon_list, koff_list)
    ]

    COLOR_NO_RESET = "#6A3D9A"
    COLOR_ACTIVE_RESET = "#1B9E77"
    COLOR_EQUILIBRIUM = "#E6AB02"
    COLOR_SAMPLE = "#D95F02"
    COLOR_LOQ = "#8C8C8C"
    COLOR_TARGET = "#3F3F3F"

    SIGNAL_ALPHA = 0.60
    EQUILIBRIUM_ALPHA = 0.45

    style_inter = {
        "font.family": "Arial",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 15,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }

    with plt.rc_context(style_inter):
        fig = plt.figure(
            figsize=(17, 3.25 * n_sensors),
            constrained_layout=True,
        )

        outer = fig.add_gridspec(
            n_sensors,
            n_profiles,
            wspace=0.28,
            hspace=0.40,
        )

        legend_handles = None
        legend_labels = None

        for row_idx, (kon, koff, Kd) in enumerate(
            zip(kon_list, koff_list, Kd_list)
        ):
            for col_idx, profile_name in enumerate(
                profile_names
            ):
                inner = outer[
                    row_idx,
                    col_idx,
                ].subgridspec(
                    2,
                    1,
                    height_ratios=[3, 1],
                    hspace=0.05,
                )

                ax_top = fig.add_subplot(inner[0])
                ax_bot = fig.add_subplot(
                    inner[1],
                    sharex=ax_top,
                )

                step_values_pM = np.asarray(
                    step_values_dict[profile_name],
                    dtype=float,
                )

                target_profile_list = (
                    make_step_list_from_step_values(
                        step_values_pM,
                        dt_step_min=dt_step_min,
                    )
                )

                fraction_bound = (
                    simulate_fraction_bound_piecewise(
                        target_profile_list,
                        kon=kon,
                        koff=koff,
                        dt=dt,
                        reset_each_step=False,
                    )
                )

                equilibrium_fraction_profile = (
                    make_equilibrium_fraction_profile(
                        target_profile_list,
                        Kd=Kd,
                        dt=dt,
                    )
                )

                target_profile_M = (
                    make_target_concentration_profile(
                        target_profile_list,
                        dt=dt,
                    )
                )

                if plot_active_reset:
                    fraction_bound_active = (
                        simulate_fraction_bound_piecewise(
                            target_profile_list,
                            kon=kon,
                            koff=koff,
                            dt=dt,
                            reset_each_step=True,
                        )
                    )

                total_sec = sum(
                    60.0 * dur_min
                    for _, dur_min in target_profile_list
                )

                n = int(round(total_sec / dt))
                time_min = (
                    np.arange(n, dtype=float)
                    * dt
                    / 60.0
                )

                lengths = [
                    len(time_min),
                    len(fraction_bound),
                    len(equilibrium_fraction_profile),
                    len(target_profile_M),
                ]

                if plot_active_reset:
                    lengths.append(
                        len(fraction_bound_active)
                    )

                min_len = min(lengths)

                time_min = time_min[:min_len]
                fraction_bound = (
                    fraction_bound[:min_len]
                )
                equilibrium_fraction_profile = (
                    equilibrium_fraction_profile[:min_len]
                )
                target_profile_M = (
                    target_profile_M[:min_len]
                )

                if plot_active_reset:
                    fraction_bound_active = (
                        fraction_bound_active[:min_len]
                    )

                ax_top.plot(
                    time_min,
                    fraction_bound,
                    linewidth=2.2,
                    color=COLOR_NO_RESET,
                    alpha=SIGNAL_ALPHA,
                    label="No Reset",
                    zorder=2,
                )

                n_per_step = int(
                    round(
                        dt_step_min
                        * 60.0
                        / dt
                    )
                )

                end_indices = np.arange(
                    n_per_step - 1,
                    len(time_min),
                    n_per_step,
                )

                end_indices = end_indices[
                    end_indices < len(fraction_bound)
                ]

                ax_top.scatter(
                    time_min[end_indices],
                    fraction_bound[end_indices],
                    s=34,
                    color=COLOR_SAMPLE,
                    edgecolor="white",
                    linewidth=0.6,
                    alpha=1.0,
                    zorder=10,
                    label="Interval-end sample",
                )

                if plot_active_reset:
                    ax_top.plot(
                        time_min,
                        fraction_bound_active,
                        linewidth=2.0,
                        color=COLOR_ACTIVE_RESET,
                        alpha=SIGNAL_ALPHA,
                        label="Active Reset",
                        zorder=3,
                    )

                ax_top.plot(
                    time_min,
                    equilibrium_fraction_profile,
                    linewidth=2.0,
                    color=COLOR_EQUILIBRIUM,
                    alpha=EQUILIBRIUM_ALPHA,
                    label="Equilibrium Signal",
                    zorder=1,
                )

                if DFB is not None:
                    ax_top.axhline(
                        DFB,
                        color=COLOR_LOQ,
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.85,
                        label="LOQ",
                        zorder=1,
                    )

                    ax_top.axhspan(
                        0,
                        DFB,
                        color=COLOR_LOQ,
                        alpha=0.10,
                        zorder=0,
                    )

                max_signal = max(
                    np.nanmax(fraction_bound),
                    np.nanmax(
                        equilibrium_fraction_profile
                    ),
                    np.nanmax(
                        fraction_bound_active
                    )
                    if plot_active_reset
                    else 0,
                )

                ax_top.set_ylim(
                    0,
                    min(
                        1.05,
                        1.10 * max_signal
                        if max_signal > 0
                        else 1.0,
                    ),
                )

                if row_idx == 0:
                    ax_top.set_title(profile_name)

                if col_idx == 0:
                    ax_top.set_ylabel(
                        "Fraction bound"
                    )

                ax_top.text(
                    0.02,
                    0.95,
                    (
                        rf"$k_{{on}}={kon:.1e}$"
                        "\n"
                        rf"$k_{{off}}={koff:.1e}\ \mathrm{{s^{{-1}}}}$"
                        "\n"
                        rf"$K_D={Kd*1e12:g}\ \mathrm{{pM}}$"
                    ),
                    transform=ax_top.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.75,
                    ),
                )

                ax_top.grid(
                    True,
                    alpha=0.15,
                )

                plt.setp(
                    ax_top.get_xticklabels(),
                    visible=False,
                )

                ax_bot.plot(
                    time_min,
                    target_profile_M * 1e12,
                    color=COLOR_TARGET,
                    linewidth=1.8,
                    drawstyle="steps-post",
                )

                ax_bot.set_xlabel(
                    "Time (min)"
                )

                if col_idx == 0:
                    ax_bot.set_ylabel(
                        "[T]\n(pM)"
                    )

                ax_bot.set_yscale("log")
                ax_bot.grid(
                    True,
                    which="both",
                    alpha=0.15,
                )

                if legend_handles is None:
                    (
                        legend_handles,
                        legend_labels,
                    ) = (
                        ax_top
                        .get_legend_handles_labels()
                    )

        if legend_handles:
            unique = {}
            for h, l in zip(
                legend_handles,
                legend_labels,
            ):
                if l not in unique:
                    unique[l] = h

            fig.legend(
                list(unique.values()),
                list(unique.keys()),
                loc="lower center",
                bbox_to_anchor=(0.5, -0.04),
                ncol=min(5, len(unique)),
                frameon=False,
            )

    save_figure(
        fig,
        "sensor_response_all_profiles",
    )


def make_sensor_response_profile_grid():
    time_profile_min = np.arange(
        0,
        RESPONSE_GRID_TOTAL_MIN
        + RESPONSE_GRID_STEP_MIN,
        RESPONSE_GRID_STEP_MIN,
    )

    (
        _,
        _,
        _,
        step_values_dict,
    ) = make_target_profiles(
        time_profile_min,
        dt_step=RESPONSE_GRID_STEP_MIN,
        min_pM=1,
        max_pM=900,
        random_smooth_sigma_min=4,
        seed=RESPONSE_GRID_SEED,
    )

    plot_all_sensors_all_profiles(
        step_values_dict=step_values_dict,
        kon_list=RESPONSE_GRID_KON_LIST,
        koff_list=RESPONSE_GRID_KOFF_LIST,
        dt_step_min=RESPONSE_GRID_STEP_MIN,
        dt=RESPONSE_GRID_DT_S,
        DFB=RESPONSE_GRID_DFB,
        plot_active_reset=(
            RESPONSE_GRID_PLOT_ACTIVE_RESET
        ),
    )

# CV / rRMSE CALCULATIONS
def summarize_by_exact_true_concentration_and_sensor(
    df,
):
    df = df.copy()

    df["true_exact_pM"] = (
        df["true_pM"].round(6)
    )

    group_cols = [
        "sensor_idx",
        "sensor_name",
        "kon",
        "koff",
        "Kd_pM",
        "true_exact_pM",
    ]

    summary = (
        df.groupby(
            group_cols,
            observed=True,
        )
        .agg(
            true_mean_pM=("true_pM", "mean"),
            n=("true_pM", "size"),
            reset_mean=("est_reset_pM", "mean"),
            reset_std=("est_reset_pM", "std"),
            reset_mse=("sq_error_reset", "mean"),
            nonreset_mean=(
                "est_nonreset_pM",
                "mean",
            ),
            nonreset_std=(
                "est_nonreset_pM",
                "std",
            ),
            nonreset_mse=(
                "sq_error_nonreset",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["reset_cv"] = (
        summary["reset_std"]
        / summary["reset_mean"]
    )

    summary["nonreset_cv"] = (
        summary["nonreset_std"]
        / summary["nonreset_mean"]
    )

    summary["reset_rmse"] = np.sqrt(
        summary["reset_mse"]
    )

    summary["nonreset_rmse"] = np.sqrt(
        summary["nonreset_mse"]
    )

    summary["reset_relative_rmse"] = (
        summary["reset_rmse"]
        / summary["true_mean_pM"]
    )

    summary["nonreset_relative_rmse"] = (
        summary["nonreset_rmse"]
        / summary["true_mean_pM"]
    )

    return summary.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def plot_exact_mean_points_with_smooth_curve(
    summary,
    metric,
    ylabel,
    title,
    estimator="reset",
    smoothing_strength=1.0,
    band_neighbors=15,
    threshold=0.10,
):
    metric_col = f"{estimator}_{metric}"

    fig, ax = plt.subplots(
        figsize=(7.8, 5.0)
    )

    sensors = (
        summary[
            [
                "sensor_idx",
                "sensor_name",
                "kon",
                "koff",
                "Kd_pM",
            ]
        ]
        .drop_duplicates()
        .sort_values("sensor_idx")
    )

    arrow_specs = []

    for color_index, sensor_row in enumerate(
        sensors.itertuples(index=False)
    ):
        sensor_idx = int(
            sensor_row.sensor_idx
        )

        color = COLORBLIND_COLORS[
            color_index
            % len(COLORBLIND_COLORS)
        ]

        sub = summary[
            summary["sensor_idx"]
            == sensor_idx
        ].copy()

        sub = sub.dropna(
            subset=[
                "true_mean_pM",
                metric_col,
            ]
        ).sort_values(
            "true_mean_pM"
        )

        x = sub[
            "true_mean_pM"
        ].to_numpy(dtype=float)

        y = sub[
            metric_col
        ].to_numpy(dtype=float)

        valid = (
            np.isfinite(x)
            & np.isfinite(y)
            & (x > 0)
            & (y > 0)
        )

        x = x[valid]
        y = y[valid]

        if len(x) < 2:
            continue

        ax.scatter(
            x,
            y,
            alpha=0.25,
            s=18,
            color=color,
        )

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        unique_x = np.unique(x)

        unique_y = np.array([
            np.nanmean(
                y[x == ux]
            )
            for ux in unique_x
        ])

        koff = float(sensor_row.koff)
        kd_pM = float(sensor_row.Kd_pM)

        label = (
            rf"$k_{{off}}={koff:.0e}\ "
            rf"\mathrm{{s}}^{{-1}}$"
        )

        if len(sensors) > 1:
            label += (
                rf", $K_D={kd_pM:g}$ pM"
            )

        if len(unique_x) < 4:
            ax.plot(
                unique_x,
                unique_y,
                linewidth=2.8,
                color=color,
                label=label,
            )
            continue

        log_x = np.log10(unique_x)
        log_y = np.log10(unique_y)

        spline = UnivariateSpline(
            log_x,
            log_y,
            s=(
                smoothing_strength
                * len(unique_x)
            ),
        )

        log_x_smooth = np.linspace(
            log_x.min(),
            log_x.max(),
            500,
        )

        log_y_smooth = spline(
            log_x_smooth
        )

        x_smooth = 10 ** log_x_smooth
        y_smooth = 10 ** log_y_smooth

        fitted_at_points = spline(log_x)

        log_residual = (
            log_y
            - fitted_at_points
        )

        lower_residual = np.zeros_like(
            log_x_smooth
        )
        upper_residual = np.zeros_like(
            log_x_smooth
        )

        n_neighbors = min(
            band_neighbors,
            len(log_x),
        )

        for j, x0 in enumerate(
            log_x_smooth
        ):
            distances = np.abs(
                log_x - x0
            )

            nearest_idx = (
                np.argsort(distances)[
                    :n_neighbors
                ]
            )

            local_residuals = (
                log_residual[nearest_idx]
            )

            lower_residual[j] = (
                np.nanpercentile(
                    local_residuals,
                    2.5,
                )
            )

            upper_residual[j] = (
                np.nanpercentile(
                    local_residuals,
                    97.5,
                )
            )

        if len(lower_residual) >= 11:
            lower_residual = (
                gaussian_filter1d(
                    lower_residual,
                    sigma=8,
                )
            )

            upper_residual = (
                gaussian_filter1d(
                    upper_residual,
                    sigma=8,
                )
            )

        lower = 10 ** (
            log_y_smooth
            + lower_residual
        )

        upper = 10 ** (
            log_y_smooth
            + upper_residual
        )

        ax.fill_between(
            x_smooth,
            lower,
            upper,
            color=color,
            alpha=0.13,
            linewidth=0,
        )

        ax.plot(
            x_smooth,
            y_smooth,
            linewidth=2.8,
            alpha=0.9,
            color=color,
            linestyle="-",
            label=label,
        )

        ax.plot(
            unique_x,
            unique_y,
            linewidth=2.8,
            alpha=0.08,
            color=color,
            linestyle="-",
        )

        # First crossing of the 10% line.
        log_threshold = np.log10(
            threshold
        )

        diff = (
            log_y_smooth
            - log_threshold
        )

        crossing_indices = np.where(
            np.diff(np.sign(diff))
            != 0
        )[0]

        if len(crossing_indices) > 0:
            idx = int(
                crossing_indices[0]
            )

            x1 = log_x_smooth[idx]
            x2 = log_x_smooth[idx + 1]

            y1 = log_y_smooth[idx]
            y2 = log_y_smooth[idx + 1]

            if y2 != y1:
                frac = (
                    log_threshold - y1
                ) / (
                    y2 - y1
                )

                log_x_cross = (
                    x1
                    + frac * (x2 - x1)
                )

                x_cross = (
                    10 ** log_x_cross
                )

                arrow_specs.append(
                    (x_cross, color)
                )

    ax.axhline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.4,
        alpha=0.7,
        label="10%",
    )

    ax.set_xlabel(
        "True concentration (pM)"
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.set_xscale("log")
    ax.set_xlim(right=1e3)
    ax.set_yscale("log")

    ax.grid(True, alpha=0.25)

    ymin, ymax = ax.get_ylim()

    for x_cross, color in arrow_specs:
        ax.annotate(
            "",
            xy=(x_cross, ymin),
            xytext=(
                x_cross,
                threshold,
            ),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                alpha=0.5,
                linewidth=2.0,
            ),
            annotation_clip=False,
        )

    ax.set_ylim(ymin, ymax)

    handles, labels = (
        ax.get_legend_handles_labels()
    )

    ax.legend(
        handles[::-1],
        labels[::-1],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        frameon=False,
    )

    plt.subplots_adjust(bottom=0.32)

    stem = (
        f"{'active_reset' if estimator == 'reset' else 'no_reset'}_"
        f"{'rrmse' if metric == 'relative_rmse' else metric}"
    )

    save_figure(fig, stem)


def make_performance_figures(df):
    summary = (
        summarize_by_exact_true_concentration_and_sensor(
            df
        )
    )

    plot_exact_mean_points_with_smooth_curve(
        summary,
        metric="cv",
        ylabel="CV",
        title="Active Reset",
        estimator="reset",
        smoothing_strength=(
            SPLINE_SMOOTHING_STRENGTH
        ),
        band_neighbors=(
            SPLINE_BAND_NEIGHBORS
        ),
        threshold=(
            PERFORMANCE_THRESHOLD
        ),
    )

    plot_exact_mean_points_with_smooth_curve(
        summary,
        metric="relative_rmse",
        ylabel="rRMSE",
        title="Active Reset",
        estimator="reset",
        smoothing_strength=(
            SPLINE_SMOOTHING_STRENGTH
        ),
        band_neighbors=(
            SPLINE_BAND_NEIGHBORS
        ),
        threshold=(
            PERFORMANCE_THRESHOLD
        ),
    )

    plot_exact_mean_points_with_smooth_curve(
        summary,
        metric="cv",
        ylabel="CV",
        title="No Reset",
        estimator="nonreset",
        smoothing_strength=(
            SPLINE_SMOOTHING_STRENGTH
        ),
        band_neighbors=(
            SPLINE_BAND_NEIGHBORS
        ),
        threshold=(
            PERFORMANCE_THRESHOLD
        ),
    )

    plot_exact_mean_points_with_smooth_curve(
        summary,
        metric="relative_rmse",
        ylabel="rRMSE",
        title="No Reset",
        estimator="nonreset",
        smoothing_strength=(
            SPLINE_SMOOTHING_STRENGTH
        ),
        band_neighbors=(
            SPLINE_BAND_NEIGHBORS
        ),
        threshold=(
            PERFORMANCE_THRESHOLD
        ),
    )

# STEP-CHANGE CALCULATIONS AND PLOTS
def add_signed_estimated_step_changes(df):
    df = df.copy()

    df = df.sort_values(
        [
            "sensor_idx",
            "run",
            "step",
        ]
    ).reset_index(drop=True)

    grouping = [
        "sensor_idx",
        "run",
    ]

    df["true_delta_pM"] = (
        df.groupby(grouping)[
            "true_pM"
        ].diff()
    )

    df["reset_delta_est_pM"] = (
        df.groupby(grouping)[
            "est_reset_pM"
        ].diff()
    )

    df["nonreset_delta_est_pM"] = (
        df.groupby(grouping)[
            "est_nonreset_pM"
        ].diff()
    )

    return df


def plot_estimated_vs_true_step_change_each_sensor(
    df,
):
    sensors = (
        df[
            [
                "sensor_idx",
                "sensor_name",
                "kon",
                "koff",
                "Kd_pM",
            ]
        ]
        .drop_duplicates()
        .sort_values("sensor_idx")
    )

    for sensor in sensors.itertuples(
        index=False
    ):
        sensor_idx = int(
            sensor.sensor_idx
        )

        sub = df[
            df["sensor_idx"]
            == sensor_idx
        ].copy()

        sub = sub.dropna(
            subset=[
                "true_delta_pM",
                "reset_delta_est_pM",
                "nonreset_delta_est_pM",
            ]
        )

        sub = sub[
            sub["true_delta_pM"] != 0
        ]

        x = sub[
            "true_delta_pM"
        ].to_numpy(dtype=float)

        y_reset = sub[
            "reset_delta_est_pM"
        ].to_numpy(dtype=float)

        y_nonreset = sub[
            "nonreset_delta_est_pM"
        ].to_numpy(dtype=float)

        valid_reset = (
            np.isfinite(x)
            & np.isfinite(y_reset)
        )

        valid_nonreset = (
            np.isfinite(x)
            & np.isfinite(y_nonreset)
        )

        fig, ax = plt.subplots(
            figsize=(9.2, 6.2),
            constrained_layout=True,
        )

        ax.scatter(
            x[valid_reset],
            y_reset[valid_reset],
            color=COLOR_RESET,
            alpha=0.35,
            s=40,
            label="Reset",
        )

        ax.scatter(
            x[valid_nonreset],
            y_nonreset[valid_nonreset],
            alpha=0.35,
            color=COLOR_NONRESET,
            s=40,
            label="No reset",
        )

        lim = float(
            STEP_CHANGE_XY_LIMIT_PM
        )

        ax.plot(
            [-lim, lim],
            [-lim, lim],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=(
                "Ideal: estimated change "
                "= true change"
            ),
        )

        # Correct ±10% relative-error boundaries:
        # y = 0.9x and y = 1.1x.
        xline = np.array(
            [-lim, lim],
            dtype=float,
        )

        ax.plot(
            xline,
            0.9 * xline,
            color="black",
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
            label="±10% relative error",
        )

        ax.plot(
            xline,
            1.1 * xline,
            color="black",
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
        )

        ax.axhline(
            0,
            color="black",
            linestyle=":",
            linewidth=1.0,
            alpha=0.7,
        )

        ax.axvline(
            0,
            color="black",
            linestyle=":",
            linewidth=1.0,
            alpha=0.7,
        )

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect(
            "equal",
            adjustable="box",
        )

        ax.set_xlabel(
            "True step change (pM)"
        )

        ax.set_ylabel(
            "Estimated step change (pM)"
        )

        ax.set_title(
            rf"$k_{{off}}={float(sensor.koff):.0e}\ "
            rf"\mathrm{{s}}^{{-1}}$"
        )

        ax.grid(True, alpha=0.25)

        ax.legend(
            frameon=False,
            fontsize=12,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=4,
        )

        save_figure(
            fig,
            (
                "signed_step_change_"
                f"sensor{sensor_idx:02d}"
            ),
        )


def summarize_percent_within_error_by_sensor(
    df,
    error_threshold=0.10,
):
    rows = []

    sensors = (
        df[
            [
                "sensor_idx",
                "sensor_name",
                "kon",
                "koff",
                "Kd_pM",
            ]
        ]
        .drop_duplicates()
        .sort_values("sensor_idx")
    )

    for sensor in sensors.itertuples(
        index=False
    ):
        sub = df[
            df["sensor_idx"]
            == int(sensor.sensor_idx)
        ].copy()

        sub = sub[
            np.isfinite(
                sub["true_delta_pM"]
            )
            & (
                sub["true_delta_pM"]
                != 0
            )
        ].copy()

        true_delta = sub[
            "true_delta_pM"
        ].to_numpy(dtype=float)

        reset_est = sub[
            "reset_delta_est_pM"
        ].to_numpy(dtype=float)

        valid_reset = (
            np.isfinite(true_delta)
            & np.isfinite(reset_est)
        )

        reset_rel_error = (
            np.abs(
                reset_est[valid_reset]
                - true_delta[valid_reset]
            )
            / np.abs(
                true_delta[valid_reset]
            )
        )

        reset_percent = (
            100.0
            * np.mean(
                reset_rel_error
                <= error_threshold
            )
            if np.sum(valid_reset) > 0
            else np.nan
        )

        nonreset_est = sub[
            "nonreset_delta_est_pM"
        ].to_numpy(dtype=float)

        valid_nonreset = (
            np.isfinite(true_delta)
            & np.isfinite(nonreset_est)
        )

        nonreset_rel_error = (
            np.abs(
                nonreset_est[valid_nonreset]
                - true_delta[valid_nonreset]
            )
            / np.abs(
                true_delta[valid_nonreset]
            )
        )

        nonreset_percent = (
            100.0
            * np.mean(
                nonreset_rel_error
                <= error_threshold
            )
            if np.sum(valid_nonreset) > 0
            else np.nan
        )

        rows.append({
            "sensor_idx": int(
                sensor.sensor_idx
            ),
            "sensor_name": str(
                sensor.sensor_name
            ),
            "kon": float(sensor.kon),
            "koff": float(sensor.koff),
            "Kd_pM": float(sensor.Kd_pM),
            "Reset": reset_percent,
            "Non-reset": (
                nonreset_percent
            ),
            "n_reset": int(
                np.sum(valid_reset)
            ),
            "n_nonreset": int(
                np.sum(valid_nonreset)
            ),
        })

    return pd.DataFrame(rows)


def plot_percent_within_error_by_koff(
    df_summary,
    error_threshold=0.10,
):
    df_summary = (
        df_summary
        .sort_values("koff")
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(
        figsize=(7.0, 5.2)
    )

    ax.plot(
        df_summary["koff"],
        df_summary["Reset"],
        marker="o",
        color=COLOR_RESET,
        linewidth=2,
        label="Active Reset",
    )

    ax.plot(
        df_summary["koff"],
        df_summary["Non-reset"],
        marker="o",
        color=COLOR_NONRESET,
        linewidth=2,
        label="No Reset",
    )

    ax.set_xscale("log")

    ax.set_title(
        "Step-Change Estimation Accuracy"
    )

    ax.set_xlabel(
        r"$k_{\mathrm{off}}$ (s$^{-1}$)"
    )

    ax.set_ylabel(
        (
            "Estimates within "
            f"±{100*error_threshold:.0f}% "
            "error (%)"
        )
    )

    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)

    ax.legend(frameon=False)

    plt.tight_layout()

    save_figure(
        fig,
        "step_change_accuracy_vs_koff",
    )


def make_step_change_figures(df):
    df_step = (
        add_signed_estimated_step_changes(
            df
        )
    )

    plot_estimated_vs_true_step_change_each_sensor(
        df_step
    )

    summary = (
        summarize_percent_within_error_by_sensor(
            df_step,
            error_threshold=(
                STEP_CHANGE_ERROR_THRESHOLD
            ),
        )
    )

    plot_percent_within_error_by_koff(
        summary,
        error_threshold=(
            STEP_CHANGE_ERROR_THRESHOLD
        ),
    )

# DRIFT FIGURE
def extract_analysis_mode_arrays(
    result,
    mode_name,
):
    """
    Extract analysis-period arrays from a full-run result.
    """
    meta = result["metadata"]
    mode = result["modes"][mode_name]

    samples_per_step = int(
        round(
            float(meta["step_duration_min"])
            * 60.0
            / float(meta["dt_signal_s"])
        )
    )

    start_step, end_step = (
        mode["analysis_slice_steps"]
    )

    s = slice(
        int(start_step) * samples_per_step,
        int(end_step) * samples_per_step,
    )

    return {
        "readout_drifted": np.asarray(
            mode["readout_drifted_full"],
            dtype=float,
        )[s],
        "readout_for_fit": np.asarray(
            mode["readout_for_fit_full"],
            dtype=float,
        )[s],
    }


def extract_analysis_drift(result):
    """
    Drift is shared between reset and non-reset in the simulator.
    Use the reset mode's saved drift vector if present.
    """
    meta = result["metadata"]
    mode = result["modes"]["reset"]

    samples_per_step = int(
        round(
            float(meta["step_duration_min"])
            * 60.0
            / float(meta["dt_signal_s"])
        )
    )

    start_step, end_step = (
        mode["analysis_slice_steps"]
    )

    s = slice(
        int(start_step) * samples_per_step,
        int(end_step) * samples_per_step,
    )

    # can clean up when finalized to one name
    for key in [
        "drift_full",
        "readout_drift_full",
        "drift",
    ]:
        if key in mode:
            arr = np.asarray(
                mode[key],
                dtype=float,
            )
            return arr[s]

    # If an explicit drift vector is absent, reconstruct it from
    # drifted - no-drift readout.
    if (
        "readout_drifted_full" in mode
        and "readout_no_drift_full" in mode
    ):
        drift = (
            np.asarray(
                mode["readout_drifted_full"],
                dtype=float,
            )
            - np.asarray(
                mode["readout_no_drift_full"],
                dtype=float,
            )
        )
        return drift[s]

    raise KeyError(
        "Could not locate or reconstruct drift in the full result."
    )


def load_drift_panel_result(path):
    with open(path, "rb") as f:
        result = pickle.load(f)

    if "metadata" not in result or "modes" not in result:
        raise ValueError(
            f"{path} does not look like a full-run result."
        )

    interval_df = (
        result["interval_df"]
        .sort_values("step")
        .reset_index(drop=True)
    )

    meta = result["metadata"]

    reset = extract_analysis_mode_arrays(
        result,
        "reset",
    )

    nonreset = (
        extract_analysis_mode_arrays(
            result,
            "nonreset",
        )
    )

    drift = extract_analysis_drift(
        result
    )

    return {
        "true_pM": interval_df[
            "true_pM"
        ].to_numpy(dtype=float),
        "est_reset_pM": interval_df[
            "est_reset_pM"
        ].to_numpy(dtype=float),
        "est_nonreset_pM": interval_df[
            "est_nonreset_pM"
        ].to_numpy(dtype=float),
        "signal_nonreset_drifted": (
            nonreset["readout_drifted"]
        ),
        "signal_reset_drifted": (
            reset["readout_drifted"]
        ),
        "signal_reset_corrected": (
            reset["readout_for_fit"]
        ),
        "drift": drift,
        "dt_signal_s": float(
            meta["dt_signal_s"]
        ),
        "step_duration_min": float(
            meta["step_duration_min"]
        ),
    }


def make_drift_comparison_figure():
    if not DRIFT_PANEL_FILES:
        print(
            "\nSkipping drift_comparison: "
            "DRIFT_PANEL_FILES is empty."
        )
        return

    if len(DRIFT_PANEL_FILES) != 3:
        raise ValueError(
            "DRIFT_PANEL_FILES must contain exactly three files "
            "(exponential, random walk, linear)."
        )

    results = [
        load_drift_panel_result(
            Path(path)
        )
        for path in DRIFT_PANEL_FILES
    ]

    COL_NONRESET = COLOR_NONRESET
    COL_RESET = COLOR_RESET
    COL_DRIFT = "#666666"
    COL_TRUE = "black"

    fig = plt.figure(
        figsize=(18, 5.2)
    )

    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[3, 1],
        hspace=0.12,
        wspace=0.20,
    )

    axes_top = []
    axes_bottom = []

    for col_idx, (
        result,
        title,
    ) in enumerate(
        zip(
            results,
            DRIFT_PANEL_TITLES,
        )
    ):
        ax_top = fig.add_subplot(
            gs[0, col_idx]
        )

        ax_bot = fig.add_subplot(
            gs[1, col_idx],
            sharex=ax_top,
        )

        axes_top.append(ax_top)
        axes_bottom.append(ax_bot)

        n_samples = len(
            result[
                "signal_reset_drifted"
            ]
        )

        time_min = (
            np.arange(
                n_samples,
                dtype=float,
            )
            * result["dt_signal_s"]
            / 60.0
        )

        ax_top.plot(
            time_min,
            result[
                "signal_nonreset_drifted"
            ],
            color=COL_NONRESET,
            linewidth=1.0,
            alpha=0.85,
            label=(
                "Non-reset: noise + drift"
            ),
        )

        ax_top.plot(
            time_min,
            result[
                "signal_reset_drifted"
            ],
            color=COL_RESET,
            linewidth=0.8,
            alpha=0.25,
            label=(
                "Reset raw: noise + drift"
            ),
        )

        ax_top.plot(
            time_min,
            result[
                "signal_reset_corrected"
            ],
            color=COL_RESET,
            linewidth=1.2,
            alpha=0.95,
            label="Reset corrected",
        )

        ax_top.plot(
            time_min,
            result["drift"],
            color=COL_DRIFT,
            linestyle="--",
            linewidth=1.1,
            alpha=0.95,
            label="Added drift",
        )

        ax_top.set_title(
            title,
            fontsize=14,
            pad=10,
        )

        ax_top.grid(
            True,
            alpha=0.18,
            linewidth=0.7,
        )

        ax_top.tick_params(
            axis="x",
            labelbottom=False,
        )

        if col_idx == 0:
            ax_top.set_ylabel(
                "Sensor Readout",
                fontsize=11,
            )

        n_steps_here = len(
            result["true_pM"]
        )

        step_edges_min = (
            np.arange(
                n_steps_here + 1
            )
            * result[
                "step_duration_min"
            ]
        )

        true_step = np.r_[
            result["true_pM"],
            result["true_pM"][-1],
        ]

        reset_step = np.r_[
            result["est_reset_pM"],
            result["est_reset_pM"][-1],
        ]

        nonreset_step = np.r_[
            result[
                "est_nonreset_pM"
            ],
            result[
                "est_nonreset_pM"
            ][-1],
        ]

        ax_bot.step(
            step_edges_min,
            true_step,
            where="post",
            color=COL_TRUE,
            linewidth=1.8,
            label="True [T]",
        )

        ax_bot.step(
            step_edges_min,
            reset_step,
            where="post",
            color=COL_RESET,
            linestyle="--",
            linewidth=1.6,
            label="Estimated reset",
        )

        ax_bot.step(
            step_edges_min,
            nonreset_step,
            where="post",
            color=COL_NONRESET,
            linestyle="--",
            linewidth=1.4,
            label=(
                "Estimated non-reset"
            ),
        )

        ax_bot.set_yscale("log")

        ax_bot.set_xlabel(
            "Time (min)",
            fontsize=10,
        )

        if col_idx == 0:
            ax_bot.set_ylabel(
                "[T] (pM)",
                fontsize=10,
            )

        ax_bot.grid(
            True,
            which="both",
            alpha=0.18,
            linewidth=0.7,
        )

        ax_bot.set_xlim(
            0,
            n_steps_here
            * result[
                "step_duration_min"
            ],
        )

    top_handles, top_labels = (
        axes_top[0]
        .get_legend_handles_labels()
    )

    fig.legend(
        top_handles,
        top_labels,
        loc="upper left",
        bbox_to_anchor=(
            0.005,
            0.985,
        ),
        fontsize=8,
        frameon=False,
    )

    bottom_handles, bottom_labels = (
        axes_bottom[0]
        .get_legend_handles_labels()
    )

    fig.legend(
        bottom_handles,
        bottom_labels,
        loc="lower left",
        bbox_to_anchor=(
            0.005,
            0.015,
        ),
        fontsize=8,
        frameon=False,
    )

    plt.subplots_adjust(
        left=0.085,
        right=0.99,
        top=0.88,
        bottom=0.16,
    )

    save_figure(
        fig,
        "drift_comparison",
    )

# MAIN
def main():
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading compiled interval data from:"
    )
    print(
        f"  {COMBINED_INTERVAL_CSV}"
    )

    df = load_compiled_intervals()

    print(
        f"Loaded {len(df):,} interval rows, "
        f"{df['sensor_idx'].nunique()} sensors, "
        f"{df['run'].nunique()} run indices."
    )

    print(
        "\n1/6 Example concentration profiles"
    )
    plot_example_target_profiles()

    print(
        "\n2/6 Sensor response across profiles"
    )
    make_sensor_response_profile_grid()

    print(
        "\n3/6 CV and rRMSE figures"
    )
    make_performance_figures(df)

    print(
        "\n4/6 Signed step-change figures"
    )
    df_step = (
        add_signed_estimated_step_changes(
            df
        )
    )
    plot_estimated_vs_true_step_change_each_sensor(
        df_step
    )

    print(
        "\n5/6 Step-change accuracy vs koff"
    )
    percent_summary = (
        summarize_percent_within_error_by_sensor(
            df_step,
            error_threshold=(
                STEP_CHANGE_ERROR_THRESHOLD
            ),
        )
    )

    plot_percent_within_error_by_koff(
        percent_summary,
        error_threshold=(
            STEP_CHANGE_ERROR_THRESHOLD
        ),
    )

    print(
        "\n6/6 Drift comparison"
    )
    make_drift_comparison_figure()

    print(
        "\nFinished."
    )
    print(
        f"Figures saved in: "
        f"{FIGURE_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
