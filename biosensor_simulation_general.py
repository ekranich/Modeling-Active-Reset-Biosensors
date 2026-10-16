#!/usr/bin/env python3
"""
ChatGPT Model 6 and Claude Sonnet 5.5 were used to assist with code development, clean-up, and documentation

Full Simulation
--------
1. Generate a piecewise target-concentration profile (Random / Periodic / Spike).
2. Simulate Langmuir fraction bound for reset and/or non-reset operation.
3. Add Gaussian noise in fraction bound units.
4. Convert to sensor readout with readout = alpha * fraction bound.
5. Add additive readout drift (none / linear / exponential / random_walk).
6. Optionally baseline-correct each reset interval using its local start.
7. Estimate target concentration:
     a) pre_equilibrium:
          - PyMC estimate for every sensing interval
          - grid-search estimate for every sensing interval
          - "best-of-two" chosen independently for each interval by
            whichever estimate is closer to the known simulated truth
     b) equilibrium:
          - invert the Langmuir isotherm using predefined alpha and Kd
            from a tail-mean readout for each sensing interval
8. Save a pickle containing all full time-series arrays + interval dataframe,
   plus a CSV containing interval-level values only.

The script is slurm-array friendly. Each array task corresponds to one
(sensor, run) pair.
"""

# imports
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt


# USER SETTINGS

# Target profile
PROFILE_NAME = "Random"     # "Random", "Periodic", or "Spike"

BUILDUP_LENGTH_MIN = 300.0
ANALYSIS_LENGTH_MIN = 600.0
STEP_DURATION_MIN = 30.0

# The profile's t=0 is the beginning of the ANALYSIS period.
# The buildup therefore occupies negative profile times.
# This is useful for specifying a spike at, e.g., 150 min into analysis.
PROFILE_TIME_ZERO_AT_ANALYSIS_START = True

# Random profile: log-uniform interval concentrations.
RANDOM_MIN_PM = 1.0
RANDOM_MAX_PM = 900.0
RANDOM_ROUND_TO_PM = 1.0    # e.g. 1.0 to round to nearest 1 pM; None = no rounding

# Periodic profile parameters.
PERIODIC_BASELINE_PM = 100.0
PERIODIC_AMPLITUDE_PM = 40.0
PERIODIC_PERIOD_MIN = 600.0
PERIODIC_PHASE_RAD = 0.0

# Spike profile parameters.
SPIKE_BASELINE_PM = 20.0
SPIKE_AMPLITUDE_PM = 150.0
SPIKE_T0_MIN = 150.0
SPIKE_TAU_RISE_MIN = 10.0
SPIKE_WIDTH_MIN = 40.0
SPIKE_TAU_FALL_MIN = 120.0

# Sensors
SENSORS = [
    {"name": "sensor_1", "kon": 1e6, "koff": 1e-2},
    {"name": "sensor_2", "kon": 1e6, "koff": 1e-3},
    {"name": "sensor_3", "kon": 1e6, "koff": 1e-4},
    {"name": "sensor_4", "kon": 1e6, "koff": 1e-5},
    {"name": "sensor_5", "kon": 1e6, "koff": 1e-6},
]

# Known sensor scaling: readout = ALPHA * fraction_bound.
ALPHA = 1.0

# Optional inference-model overrides. None means use the true simulation value.
INFERENCE_ALPHA = None
INFERENCE_KD_M = None
INFERENCE_KON = None
INFERENCE_KOFF = None

# Sampling
DT_SIGNAL_S = 60.0
FIT_DT_S = 60.0

# Gaussian noise
SIGMA_FRACTION = 0.005

# Drift
DRIFT_TYPE = "exponential"          # "none", "linear", "exponential", "random_walk"

# Drift is additive in sensor-readout units, after alpha scaling.
# Linear: final drift ~= DRIFT_TOTAL_READOUT.
# Exponential: final drift is normalized to DRIFT_TOTAL_READOUT.
# Random walk: max absolute drift is rescaled to DRIFT_TOTAL_READOUT when
# RANDOM_WALK_RESCALE=True.
DRIFT_TOTAL_READOUT = 0.10
EXP_DRIFT_TAU_MIN = 180.0

RANDOM_WALK_UPDATE_MIN = 1.0
RANDOM_WALK_STEP_STD_READOUT = 0.001
RANDOM_WALK_RESCALE = True

# Apply the same drift trace to reset and non-reset conditions so the
# comparison is matched.
SHARED_DRIFT_BETWEEN_MODES = True

# Reset/non-reset simulation
SIMULATE_RESET = True
SIMULATE_NONRESET = True

# For reset operation, use the beginning of each interval as a local baseline
# estimate and subtract that baseline from the whole interval.
APPLY_RESET_BASELINE_CORRECTION = True

# None = use exactly the first sample.
# Otherwise, average this many seconds at the beginning of the interval.
RESET_BASELINE_WINDOW_S = None

# For non-reset pre-equilibrium inference, f0 is estimated from the observed
# readout at the start of the current interval:
#       f0_est = interval_start_readout / inference_alpha
# This is time-aligned with the model's t=0 sample and intentionally lets
# drift/noise contaminate f0, matching what the device sees if set to "observed"
NONRESET_F0_SOURCE = "true"   # "observed" or "true"

# Inference
INFERENCE_MODE = "pre_equilibrium"  # "pre_equilibrium" or "equilibrium"

# Pre-equilibrium options.
PRE_EQ_ESTIMATOR = "best_of_two"    # "best_of_two", "pymc", or "grid"
PRIOR_PM = (0.1, 2000.0)

GRID_N = 5000

PYMC_DRAWS = 2000
PYMC_TUNE = 2000
PYMC_CHAINS = 4
PYMC_TARGET_ACCEPT = 0.98

# Equilibrium quantification:
# use the mean of the final N fit samples in each interval.
EQUILIBRIUM_TAIL_POINTS = 5

# To avoid infinite concentrations when noisy f >= 1.
EQUILIBRIUM_F_CLIP_EPS = 1e-9

# repeated runs
N_RUNS_PER_SENSOR = 100
BASE_SEED = 1

# CHANGE PATH IF NEEDED
OUTPUT_DIR = Path("simulation_results")


# Validation
def _validate_settings():
    profile = PROFILE_NAME.lower()
    if profile not in {"random", "periodic", "spike"}:
        raise ValueError("PROFILE_NAME must be 'Random', 'Periodic', or 'Spike'.")

    if INFERENCE_MODE not in {"pre_equilibrium", "equilibrium"}:
        raise ValueError("INFERENCE_MODE must be 'pre_equilibrium' or 'equilibrium'.")

    if PRE_EQ_ESTIMATOR not in {"best_of_two", "pymc", "grid"}:
        raise ValueError("PRE_EQ_ESTIMATOR must be 'best_of_two', 'pymc', or 'grid'.")

    if DRIFT_TYPE not in {"none", "linear", "exponential", "random_walk"}:
        raise ValueError(
            "DRIFT_TYPE must be 'none', 'linear', 'exponential', or 'random_walk'."
        )

    if NONRESET_F0_SOURCE not in {"observed", "true"}:
        raise ValueError("NONRESET_F0_SOURCE must be 'observed' or 'true'.")

    if not (SIMULATE_RESET or SIMULATE_NONRESET):
        raise ValueError("At least one of SIMULATE_RESET / SIMULATE_NONRESET must be True.")

    if ALPHA == 0:
        raise ValueError("ALPHA must be nonzero.")

    if STEP_DURATION_MIN <= 0 or DT_SIGNAL_S <= 0 or FIT_DT_S <= 0:
        raise ValueError("Durations and sampling intervals must be positive.")

    samples_per_step = STEP_DURATION_MIN * 60.0 / DT_SIGNAL_S
    if not np.isclose(samples_per_step, round(samples_per_step)):
        raise ValueError(
            "STEP_DURATION_MIN * 60 / DT_SIGNAL_S must be an integer so intervals reshape cleanly."
        )

    n_full_steps = (BUILDUP_LENGTH_MIN + ANALYSIS_LENGTH_MIN) / STEP_DURATION_MIN
    if not np.isclose(n_full_steps, round(n_full_steps)):
        raise ValueError(
            "(BUILDUP_LENGTH_MIN + ANALYSIS_LENGTH_MIN) / STEP_DURATION_MIN "
            "must be an integer."
        )

    n_buildup_steps = BUILDUP_LENGTH_MIN / STEP_DURATION_MIN
    if not np.isclose(n_buildup_steps, round(n_buildup_steps)):
        raise ValueError("BUILDUP_LENGTH_MIN / STEP_DURATION_MIN must be an integer.")


def derived_inference_parameters(kon_true, koff_true):
    alpha_fit = ALPHA if INFERENCE_ALPHA is None else float(INFERENCE_ALPHA)
    kon_fit = kon_true if INFERENCE_KON is None else float(INFERENCE_KON)
    koff_fit = koff_true if INFERENCE_KOFF is None else float(INFERENCE_KOFF)

    if INFERENCE_KD_M is None:
        kd_fit = koff_fit / kon_fit
    else:
        kd_fit = float(INFERENCE_KD_M)

    return alpha_fit, kon_fit, koff_fit, kd_fit

# random generator
def spawn_rngs(run_seed):
    """
    Independent but reproducible RNGs for profile, reset noise, non-reset noise,
    and drift. Because run_seed depends only on run_idx, the same run number is
    matched across sensors.
    """
    ss = np.random.SeedSequence(run_seed)
    child = ss.spawn(5)
    return {
        "profile": np.random.default_rng(child[0]),
        "reset_noise": np.random.default_rng(child[1]),
        "nonreset_noise": np.random.default_rng(child[2]),
        "drift_a": np.random.default_rng(child[3]),
        "drift_b": np.random.default_rng(child[4]),
    }


# target profiles
def periodic_profile_pM(time_min):
    time_min = np.asarray(time_min, dtype=float)
    return (
        PERIODIC_BASELINE_PM
        + PERIODIC_AMPLITUDE_PM
        * np.sin(
            2.0 * np.pi * time_min / PERIODIC_PERIOD_MIN
            + PERIODIC_PHASE_RAD
        )
    )


def spike_profile_pM(time_min):
    """
    Spike with exponential rise for SPIKE_WIDTH_MIN followed by exponential decay.
    SPIKE_T0_MIN is measured relative to the configured profile time origin.
    """
    t = np.asarray(time_min, dtype=float)
    y = np.full_like(t, SPIKE_BASELINE_PM, dtype=float)

    rise_mask = (t >= SPIKE_T0_MIN) & (t < SPIKE_T0_MIN + SPIKE_WIDTH_MIN)
    y[rise_mask] = SPIKE_BASELINE_PM + SPIKE_AMPLITUDE_PM * (
        1.0 - np.exp(-(t[rise_mask] - SPIKE_T0_MIN) / SPIKE_TAU_RISE_MIN)
    )

    peak = SPIKE_BASELINE_PM + SPIKE_AMPLITUDE_PM * (
        1.0 - np.exp(-SPIKE_WIDTH_MIN / SPIKE_TAU_RISE_MIN)
    )

    fall_mask = t >= (SPIKE_T0_MIN + SPIKE_WIDTH_MIN)
    y[fall_mask] = SPIKE_BASELINE_PM + (peak - SPIKE_BASELINE_PM) * np.exp(
        -(t[fall_mask] - (SPIKE_T0_MIN + SPIKE_WIDTH_MIN))
        / SPIKE_TAU_FALL_MIN
    )
    return y


def make_step_profile(run_rng):
    """
    Return one target concentration per sensing interval.

    Periodic and Spike are sampled at interval centers, so the simulated target
    is piecewise constant during each sensing interval. Random is independently
    log-uniform across intervals.
    """
    total_min = BUILDUP_LENGTH_MIN + ANALYSIS_LENGTH_MIN
    n_steps = int(round(total_min / STEP_DURATION_MIN))

    centers_from_sim_start = (
        np.arange(n_steps, dtype=float) * STEP_DURATION_MIN
        + STEP_DURATION_MIN / 2.0
    )

    if PROFILE_TIME_ZERO_AT_ANALYSIS_START:
        profile_centers_min = centers_from_sim_start - BUILDUP_LENGTH_MIN
    else:
        profile_centers_min = centers_from_sim_start.copy()

    name = PROFILE_NAME.lower()

    if name == "random":
        values_pM = 10.0 ** run_rng.uniform(
            np.log10(RANDOM_MIN_PM),
            np.log10(RANDOM_MAX_PM),
            size=n_steps,
        )
        if RANDOM_ROUND_TO_PM is not None:
            inc = float(RANDOM_ROUND_TO_PM)
            values_pM = np.round(values_pM / inc) * inc
            values_pM = np.clip(values_pM, RANDOM_MIN_PM, RANDOM_MAX_PM)

    elif name == "periodic":
        values_pM = periodic_profile_pM(profile_centers_min)

    elif name == "spike":
        values_pM = spike_profile_pM(profile_centers_min)

    else:
        raise ValueError(f"Unknown PROFILE_NAME={PROFILE_NAME!r}")

    if np.any(values_pM < 0):
        raise ValueError(
            "Target profile contains negative concentration. Adjust profile parameters."
        )

    return {
        "step_values_pM_full": np.asarray(values_pM, dtype=float),
        "step_center_time_min_full": profile_centers_min,
        "step_center_from_sim_start_min_full": centers_from_sim_start,
    }

# Langmuir simulation
def fraction_bound_constant_interval(T_M, kon, koff, f0, dt_s, duration_min):
    """
    Exact Langmuir solution for a constant concentration interval.

    t = 0, dt, ..., duration-dt within each interval.
    """
    n = int(round(duration_min * 60.0 / dt_s))
    t = np.arange(n, dtype=float) * dt_s

    kd = koff / kon
    f_eq = T_M / (kd + T_M)
    tau_s = 1.0 / (kon * T_M + koff)

    return f_eq - (f_eq - f0) * np.exp(-t / tau_s)


def simulate_fraction_bound_piecewise(step_values_pM, kon, koff, reset_each_step):
    all_f = []
    f0 = 0.0

    for T_pM in np.asarray(step_values_pM, dtype=float):
        f_step = fraction_bound_constant_interval(
            T_M=T_pM * 1e-12,
            kon=kon,
            koff=koff,
            f0=f0,
            dt_s=DT_SIGNAL_S,
            duration_min=STEP_DURATION_MIN,
        )
        all_f.append(f_step)

        if reset_each_step:
            f0 = 0.0
        else:
            # Using the analytical endpoint avoids the one-sample-short propagation error that would
            # arise from using f_step[-1] as f0, which occurs at duration-dt.
            duration_s = STEP_DURATION_MIN * 60.0
            T_M = T_pM * 1e-12
            kd = koff / kon
            f_eq = T_M / (kd + T_M)
            tau_s = 1.0 / (kon * T_M + koff)
            f0 = f_eq - (f_eq - f0) * np.exp(-duration_s / tau_s)

    return np.concatenate(all_f)


def make_true_fraction_start_values(step_values_pM, kon, koff, reset_each_step):
    """
    True f(t=0) at each interval. Used only when NONRESET_F0_SOURCE='true'
    and for diagnostics.
    """
    starts = []
    f0 = 0.0

    for T_pM in np.asarray(step_values_pM, dtype=float):
        starts.append(f0)

        if reset_each_step:
            f0 = 0.0
        else:
            T_M = T_pM * 1e-12
            kd = koff / kon
            f_eq = T_M / (kd + T_M)
            tau_s = 1.0 / (kon * T_M + koff)
            duration_s = STEP_DURATION_MIN * 60.0
            f0 = f_eq - (f_eq - f0) * np.exp(-duration_s / tau_s)

    return np.asarray(starts, dtype=float)


# Noise
def add_fraction_noise(f_clean, rng):
    noise_fraction = rng.normal(0.0, SIGMA_FRACTION, size=len(f_clean))
    return f_clean + noise_fraction, noise_fraction

# fraction bound to signal
def fraction_to_readout(fraction_signal):
    return ALPHA * np.asarray(fraction_signal, dtype=float)

# Drift
def make_additive_drift(n_points, rng):
    if n_points <= 0:
        return np.array([], dtype=float)

    t_min = np.arange(n_points, dtype=float) * DT_SIGNAL_S / 60.0

    if DRIFT_TYPE == "none":
        return np.zeros(n_points, dtype=float)

    if DRIFT_TYPE == "linear":
        if n_points == 1 or t_min[-1] == 0:
            return np.zeros(n_points, dtype=float)
        return DRIFT_TOTAL_READOUT * t_min / t_min[-1]

    if DRIFT_TYPE == "exponential":
        if n_points == 1 or t_min[-1] == 0:
            return np.zeros(n_points, dtype=float)
        denom = 1.0 - np.exp(-t_min[-1] / EXP_DRIFT_TAU_MIN)
        if np.isclose(denom, 0.0):
            return np.zeros(n_points, dtype=float)
        return (
            DRIFT_TOTAL_READOUT
            * (1.0 - np.exp(-t_min / EXP_DRIFT_TAU_MIN))
            / denom
        )

    if DRIFT_TYPE == "random_walk":
        update_points = max(
            1,
            int(round(RANDOM_WALK_UPDATE_MIN * 60.0 / DT_SIGNAL_S)),
        )
        n_updates = int(np.ceil(n_points / update_points))

        increments = rng.normal(
            0.0,
            RANDOM_WALK_STEP_STD_READOUT,
            size=n_updates,
        )
        levels = np.cumsum(increments)
        levels = levels - levels[0]
        drift = np.repeat(levels, update_points)[:n_points]

        if RANDOM_WALK_RESCALE:
            max_abs = np.max(np.abs(drift))
            if max_abs > 0:
                drift = drift * DRIFT_TOTAL_READOUT / max_abs

        return drift

    raise ValueError(f"Unknown DRIFT_TYPE={DRIFT_TYPE!r}")


def subtract_reset_local_baseline(readout, n_steps):
    """
    Estimate local baseline at the start of every reset interval and subtract it
    from all samples in that interval.

    If RESET_BASELINE_WINDOW_S is None, only the first sample is used.
    """
    readout = np.asarray(readout, dtype=float)
    n_per_step = int(round(STEP_DURATION_MIN * 60.0 / DT_SIGNAL_S))

    if RESET_BASELINE_WINDOW_S is None:
        n_baseline = 1
    else:
        n_baseline = max(
            1,
            int(round(RESET_BASELINE_WINDOW_S / DT_SIGNAL_S)),
        )

    corrected = readout.copy()
    baselines = np.zeros(n_steps, dtype=float)

    for i in range(n_steps):
        a = i * n_per_step
        b = a + n_per_step
        baseline_end = min(a + n_baseline, b)
        baseline = np.mean(readout[a:baseline_end])
        baselines[i] = baseline
        corrected[a:b] = readout[a:b] - baseline

    return corrected, baselines


# Interval organization
def downsample_by_interval(signal, n_steps):
    signal = np.asarray(signal, dtype=float)
    n_full_per_step = int(round(STEP_DURATION_MIN * 60.0 / DT_SIGNAL_S))
    expected = n_steps * n_full_per_step

    if len(signal) < expected:
        raise ValueError(f"Signal has {len(signal)} points; expected at least {expected}.")

    stride = max(1, int(round(FIT_DT_S / DT_SIGNAL_S)))
    actual_fit_dt_s = DT_SIGNAL_S * stride

    matrix = signal[:expected].reshape(n_steps, n_full_per_step)
    fit_matrix = matrix[:, ::stride]

    return fit_matrix, actual_fit_dt_s


def observed_nonreset_f0_from_readout(readout_full, n_steps, alpha_fit):
    """
    Estimate f0 from the observed first sample of each interval, converted back
    to fraction-bound units with known inference alpha.

    The first simulation interval keeps f0=0 because the simulation is
    initialized unbound. For every later interval, the selected sample is
    exactly time-aligned with the inference model's t=0 point.
    """
    readout_full = np.asarray(readout_full, dtype=float)
    n_per_step = int(round(STEP_DURATION_MIN * 60.0 / DT_SIGNAL_S))
    f0 = np.zeros(n_steps, dtype=float)

    for i in range(1, n_steps):
        start_idx = i * n_per_step
        f0[i] = readout_full[start_idx] / alpha_fit

    return f0


# Forward model used by inference for pre-equilibrium
def predict_fraction_step(T_pM, f0, kon_fit, koff_fit, fit_dt_s, n_points):
    T_M = np.asarray(T_pM, dtype=float) * 1e-12
    kd = koff_fit / kon_fit
    t = np.arange(n_points, dtype=float) * fit_dt_s

    f_eq = T_M / (kd + T_M)
    tau_s = 1.0 / (kon_fit * T_M + koff_fit)

    # Supports scalar T_pM. Grid search uses a vectorized version below.
    return f_eq - (f_eq - f0) * np.exp(-t / tau_s)


def predict_readout_step(
    T_pM,
    f0,
    kon_fit,
    koff_fit,
    alpha_fit,
    fit_dt_s,
    n_points,
):
    return alpha_fit * predict_fraction_step(
        T_pM=T_pM,
        f0=f0,
        kon_fit=kon_fit,
        koff_fit=koff_fit,
        fit_dt_s=fit_dt_s,
        n_points=n_points,
    )

# Grid search
def estimate_interval_grid(
    y_step,
    f0,
    kon_fit,
    koff_fit,
    alpha_fit,
    fit_dt_s,
):
    y = np.asarray(y_step, dtype=float)
    candidates_pM = np.linspace(PRIOR_PM[0], PRIOR_PM[1], GRID_N)
    T_M = candidates_pM[:, None] * 1e-12

    kd = koff_fit / kon_fit
    t = np.arange(len(y), dtype=float)[None, :] * fit_dt_s

    f_eq = T_M / (kd + T_M)
    tau_s = 1.0 / (kon_fit * T_M + koff_fit)
    pred = alpha_fit * (f_eq - (f_eq - f0) * np.exp(-t / tau_s))

    mse = np.mean((pred - y[None, :]) ** 2, axis=1)
    return float(candidates_pM[np.argmin(mse)])


def infer_all_grid(y_matrix, f0_values, kon_fit, koff_fit, alpha_fit, fit_dt_s):
    out = np.zeros(y_matrix.shape[0], dtype=float)

    for i in range(y_matrix.shape[0]):
        out[i] = estimate_interval_grid(
            y_step=y_matrix[i],
            f0=f0_values[i],
            kon_fit=kon_fit,
            koff_fit=koff_fit,
            alpha_fit=alpha_fit,
            fit_dt_s=fit_dt_s,
        )

    return out


# PyMC
def estimate_interval_pymc(
    y_step,
    f0,
    kon_fit,
    koff_fit,
    alpha_fit,
    fit_dt_s,
    random_seed,
):
    y = np.asarray(y_step, dtype=float)
    t = np.arange(len(y), dtype=float) * fit_dt_s

    lower_pM, upper_pM = PRIOR_PM
    sigma_readout = abs(alpha_fit) * SIGMA_FRACTION

    if sigma_readout <= 0:
        raise ValueError("PyMC requires a positive readout noise standard deviation.")

    with pm.Model() as model:
        logT_pM = pm.Uniform(
            "logT_pM",
            lower=np.log(lower_pM),
            upper=np.log(upper_pM),
        )
        T_pM = pm.Deterministic("T_pM", pt.exp(logT_pM))
        T_M = T_pM * 1e-12

        kd = koff_fit / kon_fit
        f_eq = T_M / (kd + T_M)
        tau_s = 1.0 / (kon_fit * T_M + koff_fit)

        f_pred = f_eq - (f_eq - f0) * pt.exp(-t / tau_s)
        readout_pred = alpha_fit * f_pred

        pm.Normal(
            "y_obs",
            mu=readout_pred,
            sigma=sigma_readout,
            observed=y,
        )

        idata = pm.sample(
            draws=PYMC_DRAWS,
            tune=PYMC_TUNE,
            chains=PYMC_CHAINS,
            cores=1,
            target_accept=PYMC_TARGET_ACCEPT,
            progressbar=False,
            random_seed=random_seed,
            compute_convergence_checks=False,
        )

    return float(
        idata.posterior["T_pM"]
        .mean(dim=("chain", "draw"))
        .values
        .item()
    )


def infer_all_pymc(
    y_matrix,
    f0_values,
    kon_fit,
    koff_fit,
    alpha_fit,
    fit_dt_s,
    seed_base,
):
    out = np.zeros(y_matrix.shape[0], dtype=float)

    for i in range(y_matrix.shape[0]):
        out[i] = estimate_interval_pymc(
            y_step=y_matrix[i],
            f0=f0_values[i],
            kon_fit=kon_fit,
            koff_fit=koff_fit,
            alpha_fit=alpha_fit,
            fit_dt_s=fit_dt_s,
            random_seed=int(seed_base + i),
        )

    return out


# Pre-equilibrium inference + per-interval best-of-two
def run_pre_equilibrium_inference(
    y_matrix,
    f0_values,
    true_pM,
    kon_fit,
    koff_fit,
    alpha_fit,
    fit_dt_s,
    seed_base,
):
    result = {
        "est_pymc_pM": np.full(len(true_pM), np.nan),
        "est_grid_pM": np.full(len(true_pM), np.nan),
        "est_best_pM": np.full(len(true_pM), np.nan),
        "best_method": np.full(len(true_pM), "", dtype=object),
    }

    need_pymc = PRE_EQ_ESTIMATOR in {"best_of_two", "pymc"}
    need_grid = PRE_EQ_ESTIMATOR in {"best_of_two", "grid"}

    if need_pymc:
        result["est_pymc_pM"] = infer_all_pymc(
            y_matrix=y_matrix,
            f0_values=f0_values,
            kon_fit=kon_fit,
            koff_fit=koff_fit,
            alpha_fit=alpha_fit,
            fit_dt_s=fit_dt_s,
            seed_base=seed_base,
        )

    if need_grid:
        result["est_grid_pM"] = infer_all_grid(
            y_matrix=y_matrix,
            f0_values=f0_values,
            kon_fit=kon_fit,
            koff_fit=koff_fit,
            alpha_fit=alpha_fit,
            fit_dt_s=fit_dt_s,
        )

    if PRE_EQ_ESTIMATOR == "pymc":
        result["est_best_pM"] = result["est_pymc_pM"].copy()
        result["best_method"][:] = "pymc"

    elif PRE_EQ_ESTIMATOR == "grid":
        result["est_best_pM"] = result["est_grid_pM"].copy()
        result["best_method"][:] = "grid"

    elif PRE_EQ_ESTIMATOR == "best_of_two":
        pymc_err = np.abs(result["est_pymc_pM"] - true_pM)
        grid_err = np.abs(result["est_grid_pM"] - true_pM)
        choose_grid = grid_err < pymc_err

        result["est_best_pM"] = np.where(
            choose_grid,
            result["est_grid_pM"],
            result["est_pymc_pM"],
        )
        result["best_method"] = np.where(choose_grid, "grid", "pymc").astype(object)

    return result


# Equilibrium quantification
def equilibrium_quantification(y_matrix, alpha_fit, kd_fit_M):
    """
    Per interval:
        1. mean final EQUILIBRIUM_TAIL_POINTS readout samples,
        2. f_hat = readout / alpha,
        3. invert f = T / (Kd + T):
               T = Kd * f / (1 - f)

    """
    if EQUILIBRIUM_TAIL_POINTS <= 0:
        raise ValueError("EQUILIBRIUM_TAIL_POINTS must be >= 1.")

    n_tail = min(EQUILIBRIUM_TAIL_POINTS, y_matrix.shape[1])
    readout_ss = np.mean(y_matrix[:, -n_tail:], axis=1)
    f_hat_raw = readout_ss / alpha_fit

    eps = EQUILIBRIUM_F_CLIP_EPS
    f_hat_clip = np.clip(f_hat_raw, 0.0, 1.0 - eps)

    T_M = kd_fit_M * f_hat_clip / (1.0 - f_hat_clip)
    T_pM = T_M * 1e12

    return {
        "est_equilibrium_pM": T_pM,
        "equilibrium_tail_readout": readout_ss,
        "equilibrium_f_hat_raw": f_hat_raw,
        "equilibrium_f_hat_clipped": f_hat_clip,
    }


# Simulate one mode (reset or non-reset)
def simulate_one_mode(
    mode,
    step_values_pM_full,
    kon_true,
    koff_true,
    alpha_fit,
    kon_fit,
    koff_fit,
    kd_fit_M,
    rng_noise,
    drift_trace,
    seed_base,
):
    reset = mode == "reset"
    n_steps_full = len(step_values_pM_full)

    f_clean_full = simulate_fraction_bound_piecewise(
        step_values_pM=step_values_pM_full,
        kon=kon_true,
        koff=koff_true,
        reset_each_step=reset,
    )

    true_f0_full = make_true_fraction_start_values(
        step_values_pM=step_values_pM_full,
        kon=kon_true,
        koff=koff_true,
        reset_each_step=reset,
    )

    # fraction bound -> + noise -> alpha scaling -> + drift
    f_noisy_full, noise_fraction_full = add_fraction_noise(f_clean_full, rng_noise)
    readout_no_drift_full = fraction_to_readout(f_noisy_full)
    readout_drifted_full = readout_no_drift_full + drift_trace

    if reset and APPLY_RESET_BASELINE_CORRECTION:
        readout_for_fit_full, baseline_est_full = subtract_reset_local_baseline(
            readout=readout_drifted_full,
            n_steps=n_steps_full,
        )
    else:
        readout_for_fit_full = readout_drifted_full.copy()
        baseline_est_full = np.zeros(n_steps_full, dtype=float)

    y_fit_full, actual_fit_dt_s = downsample_by_interval(
        signal=readout_for_fit_full,
        n_steps=n_steps_full,
    )

    if reset:
        f0_fit_full = np.zeros(n_steps_full, dtype=float)
    else:
        if NONRESET_F0_SOURCE == "true":
            f0_fit_full = true_f0_full.copy()
        else:
            f0_fit_full = observed_nonreset_f0_from_readout(
                readout_full=readout_for_fit_full,
                n_steps=n_steps_full,
                alpha_fit=alpha_fit,
            )

    # Analyze only after buildup.
    n_buildup = int(round(BUILDUP_LENGTH_MIN / STEP_DURATION_MIN))
    n_analysis = int(round(ANALYSIS_LENGTH_MIN / STEP_DURATION_MIN))
    sl = slice(n_buildup, n_buildup + n_analysis)

    true_pM = np.asarray(step_values_pM_full[sl], dtype=float)
    y_fit = y_fit_full[sl]
    f0_fit = f0_fit_full[sl]

    if INFERENCE_MODE == "pre_equilibrium":
        inference = run_pre_equilibrium_inference(
            y_matrix=y_fit,
            f0_values=f0_fit,
            true_pM=true_pM,
            kon_fit=kon_fit,
            koff_fit=koff_fit,
            alpha_fit=alpha_fit,
            fit_dt_s=actual_fit_dt_s,
            seed_base=seed_base,
        )
        est_primary = inference["est_best_pM"]

    else:
        inference = equilibrium_quantification(
            y_matrix=y_fit,
            alpha_fit=alpha_fit,
            kd_fit_M=kd_fit_M,
        )
        est_primary = inference["est_equilibrium_pM"]

    return {
        "mode": mode,
        "f_clean_full": f_clean_full,
        "noise_fraction_full": noise_fraction_full,
        "f_noisy_full": f_noisy_full,
        "readout_no_drift_full": readout_no_drift_full,
        "readout_drifted_full": readout_drifted_full,
        "readout_for_fit_full": readout_for_fit_full,
        "baseline_estimate_per_step_full": baseline_est_full,
        "true_f0_per_step_full": true_f0_full,
        "fit_f0_per_step_full": f0_fit_full,
        "y_fit_matrix_full": y_fit_full,
        "actual_fit_dt_s": actual_fit_dt_s,
        "analysis_slice_steps": (n_buildup, n_buildup + n_analysis),
        "true_pM": true_pM,
        "y_fit_matrix": y_fit,
        "fit_f0_per_step": f0_fit,
        "inference": inference,
        "est_primary_pM": est_primary,
    }


# One complete sensor/run simulation
def run_one_simulation(sensor_idx, run_idx):
    sensor = SENSORS[sensor_idx]
    sensor_name = str(sensor.get("name", f"sensor_{sensor_idx}"))
    kon_true = float(sensor["kon"])
    koff_true = float(sensor["koff"])
    kd_true_M = koff_true / kon_true
    kd_true_pM = kd_true_M * 1e12

    alpha_fit, kon_fit, koff_fit, kd_fit_M = derived_inference_parameters(
        kon_true=kon_true,
        koff_true=koff_true,
    )

    run_seed = int(BASE_SEED + run_idx)
    rngs = spawn_rngs(run_seed)

    profile = make_step_profile(rngs["profile"])
    step_values_pM_full = profile["step_values_pM_full"]
    n_steps_full = len(step_values_pM_full)

    n_points = (
        n_steps_full
        * int(round(STEP_DURATION_MIN * 60.0 / DT_SIGNAL_S))
    )

    drift_reset = make_additive_drift(n_points, rngs["drift_a"])
    if SHARED_DRIFT_BETWEEN_MODES:
        drift_nonreset = drift_reset.copy()
    else:
        drift_nonreset = make_additive_drift(n_points, rngs["drift_b"])

    modes = {}

    if SIMULATE_RESET:
        modes["reset"] = simulate_one_mode(
            mode="reset",
            step_values_pM_full=step_values_pM_full,
            kon_true=kon_true,
            koff_true=koff_true,
            alpha_fit=alpha_fit,
            kon_fit=kon_fit,
            koff_fit=koff_fit,
            kd_fit_M=kd_fit_M,
            rng_noise=rngs["reset_noise"],
            drift_trace=drift_reset,
            seed_base=10_000_000 + 100_000 * run_idx + 10_000 * sensor_idx,
        )

    if SIMULATE_NONRESET:
        modes["nonreset"] = simulate_one_mode(
            mode="nonreset",
            step_values_pM_full=step_values_pM_full,
            kon_true=kon_true,
            koff_true=koff_true,
            alpha_fit=alpha_fit,
            kon_fit=kon_fit,
            koff_fit=koff_fit,
            kd_fit_M=kd_fit_M,
            rng_noise=rngs["nonreset_noise"],
            drift_trace=drift_nonreset,
            seed_base=20_000_000 + 100_000 * run_idx + 10_000 * sensor_idx,
        )

    result = {
        "metadata": {
            "profile_name": PROFILE_NAME,
            "sensor_idx": sensor_idx,
            "sensor_name": sensor_name,
            "run_idx": run_idx,
            "run_seed": run_seed,
            "kon_true_M_inv_s": kon_true,
            "koff_true_s_inv": koff_true,
            "Kd_true_M": kd_true_M,
            "Kd_true_pM": kd_true_pM,
            "alpha_true": ALPHA,
            "kon_fit_M_inv_s": kon_fit,
            "koff_fit_s_inv": koff_fit,
            "Kd_fit_M": kd_fit_M,
            "Kd_fit_pM": kd_fit_M * 1e12,
            "alpha_fit": alpha_fit,
            "sigma_fraction": SIGMA_FRACTION,
            "sigma_readout_nominal": abs(ALPHA) * SIGMA_FRACTION,
            "drift_type": DRIFT_TYPE,
            "drift_total_readout": DRIFT_TOTAL_READOUT,
            "inference_mode": INFERENCE_MODE,
            "pre_eq_estimator": PRE_EQ_ESTIMATOR if INFERENCE_MODE == "pre_equilibrium" else None,
            "best_of_two_is_oracle": (
                INFERENCE_MODE == "pre_equilibrium"
                and PRE_EQ_ESTIMATOR == "best_of_two"
            ),
            "step_duration_min": STEP_DURATION_MIN,
            "dt_signal_s": DT_SIGNAL_S,
            "fit_dt_s_requested": FIT_DT_S,
            "buildup_length_min": BUILDUP_LENGTH_MIN,
            "analysis_length_min": ANALYSIS_LENGTH_MIN,
        },
        "profile": profile,
        "drift_reset_full": drift_reset,
        "drift_nonreset_full": drift_nonreset,
        "modes": modes,
    }

    result["interval_df"] = make_interval_dataframe(result)
    return result


# Interval dataframe
def make_interval_dataframe(result):
    meta = result["metadata"]
    profile = result["profile"]

    n_buildup = int(round(BUILDUP_LENGTH_MIN / STEP_DURATION_MIN))
    n_analysis = int(round(ANALYSIS_LENGTH_MIN / STEP_DURATION_MIN))
    sl = slice(n_buildup, n_buildup + n_analysis)

    true_pM = np.asarray(profile["step_values_pM_full"][sl], dtype=float)
    center_time = np.asarray(profile["step_center_time_min_full"][sl], dtype=float)

    df = pd.DataFrame({
        "sensor_idx": meta["sensor_idx"],
        "sensor_name": meta["sensor_name"],
        "run": meta["run_idx"],
        "step": np.arange(n_analysis, dtype=int),
        "time_center_min": center_time,
        "profile": meta["profile_name"],
        "kon": meta["kon_true_M_inv_s"],
        "koff": meta["koff_true_s_inv"],
        "Kd_pM": meta["Kd_true_pM"],
        "alpha": meta["alpha_true"],
        "sigma_fraction": meta["sigma_fraction"],
        "drift_type": meta["drift_type"],
        "true_pM": true_pM,
    })

    df["prev_true_pM"] = df["true_pM"].shift(1)
    df["delta_true_pM"] = df["true_pM"].diff()

    for mode, mode_result in result["modes"].items():
        prefix = "reset" if mode == "reset" else "nonreset"

        df[f"f0_{prefix}"] = mode_result["fit_f0_per_step"]
        df[f"est_{prefix}_pM"] = mode_result["est_primary_pM"]
        df[f"error_{prefix}_pM"] = df[f"est_{prefix}_pM"] - df["true_pM"]
        df[f"sq_error_{prefix}"] = df[f"error_{prefix}_pM"] ** 2

        inference = mode_result["inference"]

        if INFERENCE_MODE == "pre_equilibrium":
            df[f"est_{prefix}_pymc_pM"] = inference["est_pymc_pM"]
            df[f"est_{prefix}_grid_pM"] = inference["est_grid_pM"]
            df[f"best_method_{prefix}"] = inference["best_method"]

        else:
            df[f"equilibrium_tail_readout_{prefix}"] = inference[
                "equilibrium_tail_readout"
            ]
            df[f"equilibrium_f_hat_raw_{prefix}"] = inference[
                "equilibrium_f_hat_raw"
            ]
            df[f"equilibrium_f_hat_clipped_{prefix}"] = inference[
                "equilibrium_f_hat_clipped"
            ]

    return df


# Saving
def save_result(result):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = result["metadata"]
    sensor_idx = meta["sensor_idx"]
    run_idx = meta["run_idx"]

    stem = (
        f"{PROFILE_NAME.lower()}_"
        f"sensor{sensor_idx:02d}_"
        f"run{run_idx:03d}"
    )

    pickle_path = OUTPUT_DIR / f"{stem}.pkl"
    csv_path = OUTPUT_DIR / f"{stem}_intervals.csv"

    with open(pickle_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    result["interval_df"].to_csv(csv_path, index=False)

    return pickle_path, csv_path


# Slurm array mapping
def get_task_mapping():
    n_sensors = len(SENSORS)
    n_total = n_sensors * N_RUNS_PER_SENSOR

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    if task_id < 0 or task_id >= n_total:
        raise ValueError(
            f"SLURM_ARRAY_TASK_ID={task_id}; valid IDs are 0 through {n_total - 1}."
        )

    sensor_idx = task_id // N_RUNS_PER_SENSOR
    run_idx = task_id % N_RUNS_PER_SENSOR
    return task_id, sensor_idx, run_idx, n_total


# Main
def main():
    _validate_settings()
    task_id, sensor_idx, run_idx, n_total = get_task_mapping()

    sensor = SENSORS[sensor_idx]
    kon = float(sensor["kon"])
    koff = float(sensor["koff"])
    kd_pM = (koff / kon) * 1e12

    print("=" * 72, flush=True)
    print(f"Task:          {task_id}/{n_total - 1}", flush=True)
    print(f"Profile:       {PROFILE_NAME}", flush=True)
    print(f"Sensor:        {sensor_idx} ({sensor.get('name', f'sensor_{sensor_idx}')})", flush=True)
    print(f"Run:           {run_idx + 1}/{N_RUNS_PER_SENSOR}", flush=True)
    print(f"kon:           {kon:g} M^-1 s^-1", flush=True)
    print(f"koff:          {koff:g} s^-1", flush=True)
    print(f"Kd:            {kd_pM:g} pM", flush=True)
    print(f"alpha:         {ALPHA:g}", flush=True)
    print(f"sigma_f:       {SIGMA_FRACTION:g}", flush=True)
    print(f"drift:         {DRIFT_TYPE}", flush=True)
    print(f"inference:     {INFERENCE_MODE}", flush=True)
    if INFERENCE_MODE == "pre_equilibrium":
        print(f"estimator:     {PRE_EQ_ESTIMATOR}", flush=True)
        if PRE_EQ_ESTIMATOR == "best_of_two":
            print("NOTE: best_of_two is an interval-wise oracle benchmark.", flush=True)
    print("=" * 72, flush=True)

    result = run_one_simulation(sensor_idx=sensor_idx, run_idx=run_idx)
    pickle_path, csv_path = save_result(result)

    print("", flush=True)
    print("=" * 72, flush=True)
    print(f"COMPLETED task {task_id}", flush=True)
    print(f"Saved full result: {pickle_path}", flush=True)
    print(f"Saved intervals:   {csv_path}", flush=True)
    print(f"Rows:               {len(result['interval_df'])}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
