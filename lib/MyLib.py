
#coomento
from __future__ import annotations


import importlib
import json
import threading
import time

import pandas as pd

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.stats import gaussian_kde
from scipy.signal import deconvolve
from scipy import stats

from scipy.signal.windows import tukey

from scipy.fft import fft, ifft, fftshift, ifftshift

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from landaupy import langauss
import io

import io
import importlib
import json

import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning   
import warnings


global constant_parameters

# =====================================================
# Standard automatic binning protocol
# =====================================================

AUTO_BINNING_LOW_PERCENTILE = 0.1
AUTO_BINNING_HIGH_PERCENTILE = 99.9

AUTO_BINNING_PADDING_FRACTION = 0.10

AUTO_BINNING_MIN_BINS = 80
AUTO_BINNING_MAX_BINS = 400

AUTO_BINNING_FINE_GRID_SIZE = 2048

HISTOGRAM_N_BINS = None


def configure_histogram_binning(constant_parameters):
    """
    Configura il numero comune di bin degli istogrammi.
    Il valore deve essere presente nei parametri.
    """
    global HISTOGRAM_N_BINS

    if "histogram_n_bins" not in constant_parameters:
        raise KeyError(
            "Manca 'histogram_n_bins' nei constant_parameters."
        )

    n_bins = constant_parameters["histogram_n_bins"]

    if (
        isinstance(n_bins, bool)
        or not isinstance(n_bins, int)
        or n_bins < 2
    ):
        raise ValueError(
            "'histogram_n_bins' deve essere "
            "un numero intero maggiore o uguale a 2."
        )

    HISTOGRAM_N_BINS = n_bins

    print(
        "[INFO] Numero comune di bin degli istogrammi: "
        f"{HISTOGRAM_N_BINS}"
    )
from scipy.signal import fftconvolve


def reconvolve_same_grid(pdf, kernel, xgrid):
    """
    Linear convolution of two PDFs sampled on the same uniform grid.

    Approximates:
        h(x) = integral f(u) k(x-u) du

    The output is represented on the same xgrid.
    """
    pdf = np.asarray(pdf, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    xgrid = np.asarray(xgrid, dtype=float)

    if pdf.ndim != 1:
        raise ValueError("pdf must be one-dimensional.")

    if kernel.ndim != 1:
        raise ValueError("kernel must be one-dimensional.")

    if xgrid.ndim != 1:
        raise ValueError("xgrid must be one-dimensional.")

    if (
        pdf.size != kernel.size
        or pdf.size != xgrid.size
    ):
        raise ValueError(
            "pdf, kernel and xgrid must have the same length."
        )

    if xgrid.size < 3:
        raise ValueError(
            "xgrid must contain at least three points."
        )

    dx_values = np.diff(xgrid)

    if np.any(dx_values <= 0.0):
        raise ValueError(
            "xgrid must be strictly increasing."
        )

    dx = float(np.mean(dx_values))

    if not np.allclose(
        dx_values,
        dx,
        rtol=1.0e-8,
        atol=1.0e-12,
    ):
        raise ValueError(
            "xgrid must be uniformly spaced."
        )

    pdf = np.where(
        np.isfinite(pdf),
        pdf,
        0.0,
    )

    kernel = np.where(
        np.isfinite(kernel),
        kernel,
        0.0,
    )

    pdf = normalize_pdf(
        pdf,
        xgrid,
    )

    kernel = normalize_pdf(
        kernel,
        xgrid,
    )

    result = fftconvolve(
        pdf,
        kernel,
        mode="same",
    )

    result *= dx

    result = np.where(
        np.isfinite(result),
        result,
        0.0,
    )

    result = np.clip(
        result,
        0.0,
        None,
    )

    return normalize_pdf(
        result,
        xgrid,
    )


def run_wiener_synthetic_test(
    xgrid,
    sigma_intrinsic=6.30,
    sigma_kernel=0.63,
    lambd=1.0e-3,
    make_plot=False,
):
    """
    Synthetic validation of the currently active convolution/Wiener pair.

    The test uses the SAME xgrid that will be used by the main analysis:

        intrinsic Gaussian
        -> reconvolve_same_grid
        -> wiener_deconvolution_fixed_lambda
        -> reconvolve_same_grid

    The function does not modify the main analysis data.
    """
    xgrid = np.asarray(
        xgrid,
        dtype=float,
    )

    if xgrid.ndim != 1 or xgrid.size < 3:
        raise ValueError(
            "xgrid must be a one-dimensional grid "
            "with at least three points."
        )

    dx_values = np.diff(xgrid)

    if (
        np.any(dx_values <= 0.0)
        or not np.allclose(
            dx_values,
            dx_values[0],
            rtol=1.0e-8,
            atol=1.0e-12,
        )
    ):
        raise ValueError(
            "xgrid must be uniformly spaced and increasing."
        )

    dx = float(dx_values[0])

    sigma_intrinsic = float(
        sigma_intrinsic
    )

    sigma_kernel = float(
        sigma_kernel
    )

    lambd = float(lambd)

    def gaussian_pdf(x, sigma):
        pdf = np.exp(
            -0.5 * (x / float(sigma)) ** 2
        )

        return normalize_pdf(
            pdf,
            x,
        )

    pdf_intrinsic = gaussian_pdf(
        xgrid,
        sigma_intrinsic,
    )

    pdf_kernel = gaussian_pdf(
        xgrid,
        sigma_kernel,
    )

    pdf_measured = reconvolve_same_grid(
        pdf_intrinsic,
        pdf_kernel,
        xgrid,
    )

    pdf_recovered = wiener_deconvolution_fixed_lambda(
        pdf_measured,
        pdf_kernel,
        xgrid,
        lambd,
    )

    pdf_reconvolved = reconvolve_same_grid(
        pdf_recovered,
        pdf_kernel,
        xgrid,
    )
    dx_check = float(np.mean(np.diff(xgrid)))

    intrinsic_fft = np.fft.fft(
        np.fft.ifftshift(pdf_intrinsic)
    )
    kernel_fft = np.fft.fft(
        np.fft.ifftshift(pdf_kernel)
    )

    measured_circular_raw = np.fft.fftshift(
        np.fft.ifft(
            intrinsic_fft * kernel_fft
        ).real
    ) * dx_check

    measured_circular = normalize_pdf(
        np.maximum(measured_circular_raw, 0.0),
        xgrid,
    )

    measured_linear = normalize_pdf(
        pdf_measured,
        xgrid,
    )

    linear_circular_relative_l2 = np.sqrt(
        np.trapezoid(
            (measured_linear - measured_circular) ** 2,
            xgrid,
        )
        / np.trapezoid(
            measured_linear ** 2,
            xgrid,
        )
    )

    print(
        "Differenza relativa L2 forward lineare/circolare = "
        f"{linear_circular_relative_l2:.8e}"
    )

    sigma_intrinsic_actual = measure_sigma(
        xgrid,
        pdf_intrinsic,
        method="fit",
        fit_window=None,
    )

    sigma_kernel_actual = measure_sigma(
        xgrid,
        pdf_kernel,
        method="fit",
        fit_window=None,
    )

    sigma_measured_actual = measure_sigma(
        xgrid,
        pdf_measured,
        method="fit",
        fit_window=None,
    )

    sigma_recovered_actual = measure_sigma(
        xgrid,
        pdf_recovered,
        method="fit",
        fit_window=None,
    )

    sigma_reconvolved_actual = measure_sigma(
        xgrid,
        pdf_reconvolved,
        method="fit",
        fit_window=None,
    )

    sigma_expected = np.sqrt(
        sigma_intrinsic**2
        + sigma_kernel**2
    )

    forward_error = relative_forward_error(
        measured_pdf=pdf_measured,
        reconvolved_pdf=pdf_reconvolved,
        xgrid=xgrid,
    )

    result = {
        "npoints": int(xgrid.size),
        "dx_mm": dx,
        "sigma_intrinsic_true_mm": sigma_intrinsic,
        "sigma_kernel_true_mm": sigma_kernel,
        "sigma_intrinsic_actual_mm": (
            sigma_intrinsic_actual
        ),
        "sigma_kernel_actual_mm": sigma_kernel_actual,
        "sigma_measured_expected_mm": sigma_expected,
        "sigma_measured_actual_mm": sigma_measured_actual,
        "sigma_recovered_mm": sigma_recovered_actual,
        "sigma_reconvolved_mm": sigma_reconvolved_actual,
        "forward_error": forward_error,
        "lambda": lambd,
    }

    print("\n" + "=" * 78)
    print("SYNTHETIC WIENER VALIDATION")
    print("=" * 78)

    print(f"N points                = {result['npoints']}")
    print(f"dx [mm]                 = {result['dx_mm']:.10f}")
    print(
        f"lambda                  = "
        f"{result['lambda']:.8e}"
    )

    print(
        f"Intrinsic sigma true     = "
        f"{result['sigma_intrinsic_true_mm']:.8f} mm"
    )

    print(
        f"Kernel sigma true        = "
        f"{result['sigma_kernel_true_mm']:.8f} mm"
    )

    print(
        f"Measured sigma expected  = "
        f"{result['sigma_measured_expected_mm']:.8f} mm"
    )

    print(
        f"Measured sigma actual    = "
        f"{result['sigma_measured_actual_mm']:.8f} mm"
    )

    print(
        f"Recovered sigma          = "
        f"{result['sigma_recovered_mm']:.8f} mm"
    )

    print(
        f"Reconvolved sigma        = "
        f"{result['sigma_reconvolved_mm']:.8f} mm"
    )

    print(
        f"Forward relative error   = "
        f"{result['forward_error']:.8e}"
    )

    return result

def _safe_filename(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return value.strip("._") or "unnamed"


def _surface_directory_name(index, z_position):
    try:
        z_text = f"{float(z_position):g}"
    except (TypeError, ValueError):
        z_text = str(z_position)
    return f"surface_{index + 1:02d}_z_{_safe_filename(z_text)}_cm"


def _save_figure(fig, path, dpi=180):
    if fig is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_figures(value, output_dir, stem):
    """Salva ricorsivamente Figure matplotlib contenute in tuple/list/dict."""
    if value is None:
        return 0
    if hasattr(value, "savefig"):
        _save_figure(value, Path(output_dir) / f"{_safe_filename(stem)}.png")
        return 1
    if isinstance(value, dict):
        return sum(
            _save_figures(child, output_dir, f"{stem}_{key}")
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(
            _save_figures(child, output_dir, f"{stem}_{index + 1:02d}")
            for index, child in enumerate(value)
        )
    return 0


def save_surface_outputs(
    stored_surface_plots,
    stored_distribution_plots,
    results_masked,
    positions_scoring,
    ripple_filters,
    particle_type,
    medium,
    case_output_dir,
):
    """Crea istogrammi e phase-space 2-D, una volta per superficie."""
    case_output_dir = Path(case_output_dir)

    if not (
        len(stored_surface_plots)
        == len(stored_distribution_plots)
        == len(results_masked)
    ):
        raise RuntimeError(
            f"{medium}: plot payload e risultati hanno dimensioni incompatibili."
        )

    for index, (hist_item, distribution_item, masked_result) in enumerate(
        zip(stored_surface_plots, stored_distribution_plots, results_masked)
    ):
        z_position = hist_item.get("z_position", positions_scoring[index])
        surface_dir = case_output_dir / _surface_directory_name(index, z_position)
        surface_dir.mkdir(parents=True, exist_ok=True)

        fig_hist, fig_variations = plot_histograms_surface(
            surface_df_masked=hist_item["surface_df_masked"],
            surface_df_filtered=hist_item["surface_df_filtered"],
            surface_data_masked_cut=hist_item["surface_data_masked_cut"],
            z_position=z_position,
            theta_peak=hist_item.get("theta_peak"),
            A=hist_item.get("A"),
            B=hist_item.get("B"),
            sigma=hist_item.get("sigma"),
            label=hist_item.get("label", medium.title()),
            ripple_filters=ripple_filters,
            particle_type=particle_type,
            positions_scoring=positions_scoring,
            show=False,
            axis_limits=None,
        )
        _save_figure(fig_hist, surface_dir / f"{medium}_histograms.png")
        _save_figure(
            fig_variations,
            surface_dir / f"{medium}_variation_histograms.png",
        )

        fig_phase_space = plot_distribution(
            surface_df_original=distribution_item["surface_df_original"],
            surface_df_filtered=distribution_item["surface_df_masked"],
            z_pos=distribution_item["z_position"],
            label=distribution_item.get("label", medium.title()),
            results_filtered=masked_result,
            show=False,
            axis_limits=None,
        )
        _save_figure(
            fig_phase_space,
            surface_dir / f"{medium}_phase_space.png",
        )


def save_generic_outputs(result, case_output_dir, constant_parameters):
    """Salva figure di evoluzione/confronto e tutte le tabelle principali."""
    generic_dir = Path(case_output_dir) / "generic_output"
    generic_dir.mkdir(parents=True, exist_ok=True)

    with (generic_dir / "run_parameters.json").open("w", encoding="utf-8") as handle:
        json.dump(constant_parameters, handle, indent=2, default=str)

    dataframe_keys = (
        "export_table_scatterers",
        "dataframe_std",
        "masked_summary",
        "scatterers_comparison",
        "direct_methods_comparison",
        "inverse_methods_comparison",
        "direct_methods_comparison_withair",
        "inverse_methods_comparison_withair",
        "direct_methods_comparison_withair_angle",
        "direct_methods_comparison_withair_total",
        "inverse_methods_comparison_withair_total",
        "direct_methods_comparison_withair_angle_total",
        "comparison_convolution",
    )
    for key in dataframe_keys:
        table = result.get(key)
        if isinstance(table, pd.DataFrame):
            table.to_csv(generic_dir / f"{key}.csv", index=False)

    figure_keys = (
        "evolution_plot_vacuum",
        "evolution_plot_air",
        "plot_comparisons",
        "plot_comparisons_of_dataframes",
        "plots_final_comparison_all_methods",
    )
    for key in figure_keys:
        _save_figures(result.get(key), generic_dir, key)

def load_data(filepath, column_names,nRows=None):
    path = Path(filepath)

    print(f"\n[DEBUG] Percorso ricevuto: {filepath}")
    print(f"[DEBUG] Percorso assoluto: {path.resolve()}")
    print(f"[DEBUG] File esistente: {path.is_file()}")

    try:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=column_names,
            nrows=nRows,
        )
        print(f"[DEBUG] File caricato: {df.shape}")
        return df

    except Exception as e:
        print(f"[DEBUG] Tipo di errore: {type(e).__name__}")
        print(f"[DEBUG] Dettaglio: {e!r}")
        return None

def conversion(df, E_0):
  df['x_prime'] = (df['cx']/df['cz'])*1e3 # milli radians
  df['y_prime'] = (df['cy']/df['cz'])*1e3 # milli radians
  df['theta'] = (np.sqrt((df['x_prime'])**2 + (df['y_prime'])**2)) # milli radians

  df.drop(columns=['cx', 'cy', 'cz'], inplace=True)

  df['x - coordinate'] = df['x - coordinate']*10 # mm
  df['y - coordinate'] = df['y - coordinate']*10 # mm

  df['total energy'] = df['total energy']*1e3 # MeV
  df['momentum'] = df['momentum']*1e3 # MeV/c

  df['delta energy'] = (- (df['total energy'] - E_0))
  
  return df

def assign_scoring_surface(df, positions, tolerance):
    df['scoring_surface'] = np.nan

    for position in positions:
        mask_exact = (df['scoring position'] == position)
        df.loc[mask_exact, 'scoring_surface'] = position

        mask_near = ((df['scoring position'] >= position - tolerance) & (df['scoring position'] <= position + tolerance) & (df['scoring position'] != position))

        n_matched = mask_near.sum()
        if n_matched > 0:
            print(f"Assigned scoring_surface = {position} cm to {n_matched} rows (corrected values).")
            examples = df.loc[mask_near, ['scoring position']].head()
            print("  Examples of corrected rows:")
            for idx, row in examples.iterrows():
                sp = row['scoring position']
                print(f"    index {idx}: scoring position = {sp}")

        df.loc[mask_near, 'scoring_surface'] = position

    unassigned_mask = df['scoring_surface'].isnull()
    no_position_count = unassigned_mask.sum()
    if no_position_count > 0:
        print(f"{no_position_count} rows were lost (no scoring_surface assigned).")
        print("Examples of unassigned rows (up to 5):")
        examples_unassigned = df.loc[unassigned_mask, ['scoring position']].head(5)
        for idx, row in examples_unassigned.iterrows():
            sp = row['scoring position']
            print(f"  index {idx}: scoring position = {sp}")
    if no_position_count == 0:
        print("All rows were successfully assigned a scoring_surface.")

    return df


def build_sequential_particles(df, label, positions_scoring):
    surface_col = "scoring_surface" 
    pattern = positions_scoring
    n = len(df)

    particle_ids = np.full(n, np.nan)
    i = 0
    pid = 1

    while i <= n - len(pattern):
        window = df.iloc[i:i+len(pattern)][surface_col].tolist()
        if window == pattern:
            particle_ids[i:i+len(pattern)] = pid
            pid += 1
            i += len(pattern)
        else:
            i += 1

    df_new = df.copy()
    df_new["particle_id"] = particle_ids

    df_particles = df_new.dropna(subset=["particle_id"]).copy()
    df_particles["particle_id"] = df_particles["particle_id"].astype(int)

    df_dropped = df_new[df_new["particle_id"].isna()].copy()
    dropped_indices = df_dropped.index

    print(f"Initial rows in {label}: {len(df)}")
    print(f"Kept rows in {label}: {len(df_particles)}")
    print(f"Dropped rows in {label}: {len(df_dropped)}")
    if not df_dropped.empty:
        print("Indexes of the dropped rows:")
        print(df_dropped.head(5)[[surface_col,"x - coordinate","y - coordinate", "delta energy"]])

    return df_particles, dropped_indices


def momentum_deviation(df):
    p0 = df['momentum'].mean()
    df['momentum_deviation'] = (df['momentum'] - p0) / p0
    return df

def beam_parameters(df, col_space, col_angle):
    space = df[col_space].values
    angle = df[col_angle].values

    mask = np.isfinite(space) & np.isfinite(angle)
    space = space[mask]
    angle = angle[mask]

    space = space - space.mean()
    angle = angle - angle.mean()

    beam_matrix = np.cov(space, angle, ddof=0)
    s11 = beam_matrix[0, 0]
    s12 = beam_matrix[0, 1]
    s22 = beam_matrix[1, 1]

    det = s11 * s22 - s12**2

    emittance = np.sqrt(det)
    alpha = -s12 / emittance
    beta  =  s11 / emittance
    gamma =  s22 / emittance

    return emittance, alpha, beta, gamma

def get_clean_data(df, column):
    """
    Return finite numeric values from a dataframe column.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if column not in df.columns:
        raise KeyError(
            f"Column {column!r} is absent from the dataframe. "
            f"Available columns: {list(df.columns)}"
        )

    data = pd.to_numeric(
        df[column],
        errors="coerce",
    ).to_numpy(dtype=float)

    return data[
        np.isfinite(data)
    ]


def make_auto_binning_from_dataframe(
    df,
    column,
    *,
    low_percentile=0.1,
    high_percentile=99.9,
    padding_fraction=0.10,
    fine_grid_size=2048,
):
    """
    Costruisce un binning con:
    - numero di bin comune, letto dai constant_parameters;
    - range dinamico calcolato dal dataframe;
    - range esteso per includere tutti i dati finiti.

    Il nome è mantenuto per compatibilità con le chiamate esistenti.
    """
    if HISTOGRAM_N_BINS is None:
        raise RuntimeError(
            "Binning non configurato. Chiamare "
            "configure_histogram_binning(constant_parameters) "
            "prima dell'analisi."
        )

    if not (
        0.0 <= low_percentile
        < high_percentile
        <= 100.0
    ):
        raise ValueError(
            "Percentiles must satisfy "
            "0 <= low_percentile < high_percentile <= 100."
        )

    if (
        not np.isfinite(padding_fraction)
        or padding_fraction < 0.0
    ):
        raise ValueError(
            "padding_fraction must be finite and non-negative."
        )

    if (
        isinstance(fine_grid_size, bool)
        or not isinstance(fine_grid_size, (int, np.integer))
        or fine_grid_size < 2
    ):
        raise ValueError(
            "fine_grid_size must be an integer of at least 2."
        )

    data = get_clean_data(
        df,
        column,
    )

    if data.size == 0:
        raise ValueError(
            f"Column {column!r} contains no finite values."
        )

    data_min = float(np.min(data))
    data_max = float(np.max(data))

    robust_min, robust_max = np.percentile(
        data,
        [low_percentile, high_percentile],
    )

    robust_min = float(robust_min)
    robust_max = float(robust_max)
    robust_width = robust_max - robust_min

    if (
        not np.isfinite(robust_width)
        or robust_width <= 1.0e-14
    ):
        centre = float(np.mean(data))

        half_width = max(
            abs(centre) * 0.05,
            1.0e-6,
        )

        robust_min = centre - half_width
        robust_max = centre + half_width
        robust_width = robust_max - robust_min

    padding = max(
        float(padding_fraction) * robust_width,
        1.0e-9,
    )

    xmin = min(
        data_min,
        robust_min - padding,
    )

    xmax = max(
        data_max,
        robust_max + padding,
    )

    if not np.isfinite(xmin) or not np.isfinite(xmax):
        raise ValueError(
            f"Non-finite binning range for column {column!r}."
        )

    if xmax <= xmin:
        centre = 0.5 * (xmin + xmax)

        half_width = max(
            abs(centre) * 0.05,
            1.0e-6,
        )

        xmin = centre - half_width
        xmax = centre + half_width

    nbins = HISTOGRAM_N_BINS

    edges = np.linspace(
        xmin,
        xmax,
        nbins + 1,
        dtype=float,
    )

    centers = 0.5 * (
        edges[:-1] + edges[1:]
    )

    finegrid = np.linspace(
        xmin,
        xmax,
        fine_grid_size,
        dtype=float,
    )

    return {
        "column": column,
        "range": (
            float(xmin),
            float(xmax),
        ),
        "nbins": nbins,
        "edges": edges,
        "centers": centers,
        "bin_width": float(
            edges[1] - edges[0]
        ),
        "finegrid": finegrid,
        "finedx": float(
            finegrid[1] - finegrid[0]
        ),
        "data_min": data_min,
        "data_max": data_max,
        "protocol": (
            "fixed_nbins_dynamic_range"
        ),
    }


def histogram_from_data(data, binning):
    """
    Build histogram data using already prepared binning.
    """

    hist, _ = np.histogram(
        data,
        bins=binning["edges"],
        density=False,
    )

    return {
        "hist": hist.astype(float),
        "edges": binning["edges"],
        "centers": binning["centers"],
        "bin_width": float(
            binning["bin_width"]
        ),
        "N": float(
            np.sum(hist)
        ),
    }


def histogram_from_df(
    df,
    column,
    binning=None,
    **binning_params,
):
    """
    Build an histogram and automatic binning from one dataframe.

    The same dataframe is deliberately used for:
    - binning-range calculation;
    - histogram values;
    - all subsequent fit functions.
    """

    if binning is None:
        binning = make_auto_binning_from_dataframe(
            df,
            column,
            **binning_params,
        )

    data = get_clean_data(
        df,
        column,
    )

    result = histogram_from_data(
        data,
        binning,
    )

    result["data"] = data
    result["binning"] = binning

    return result

def gaussian_model_counts(
    x,
    n_events,
    mu,
    sigma,
    baseline,
    bin_width,
):
    sigma = max(
        abs(float(sigma)),
        1.0e-12,
    )

    gaussian_pdf = (
        1.0
        / (
            np.sqrt(2.0 * np.pi)
            * sigma
        )
        * np.exp(
            -0.5
            * (
                (x - mu)
                / sigma
            ) ** 2
        )
    )

    return (
        baseline
        + n_events
        * gaussian_pdf
        * bin_width
    )


def gaussian_model(
    df,
    column,
    binning=None,
    n_points=2048,
    withbaseline=False,
):
    """
    Fit a Gaussian to a histogram built from df[column].
    """

    if binning is None:
        binning = make_auto_binning_from_dataframe(
            df,
            column,
        )

    histogram = histogram_from_df(
        df=df,
        column=column,
        binning=binning,
    )

    data = histogram["data"]
    hist = histogram["hist"]
    edges = histogram["edges"]
    centers = histogram["centers"]
    bin_width = histogram["bin_width"]

    xfit = np.linspace(
        edges[0],
        edges[-1],
        int(n_points),
    )

    if data.size < 5 or np.sum(hist) <= 0:
        mu = float(
            np.mean(data)
        ) if data.size else 0.0

        sigma = max(
            float(
                np.std(
                    data,
                    ddof=1,
                )
            )
            if data.size > 1
            else 0.0,
            1.0e-6,
        )

        yfit = np.zeros_like(
            xfit
        )

        return (
            xfit,
            yfit,
            float(data.size),
            mu,
            sigma,
            0.0,
        )

    n0 = float(
        np.sum(hist)
    )

    mu0 = float(
        np.mean(data)
    )

    sigma0 = max(
        float(
            np.std(
                data,
                ddof=1,
            )
        ),
        1.0e-6,
    )

    baseline0 = (
        float(np.min(hist))
        if withbaseline
        else 0.0
    )

    if withbaseline:

        def model(
            x,
            n_events,
            mu,
            sigma,
            baseline,
        ):
            return gaussian_model_counts(
                x,
                n_events,
                mu,
                sigma,
                baseline,
                bin_width,
            )

        p0 = (
            n0,
            mu0,
            sigma0,
            baseline0,
        )

        lower_bounds = (
            0.0,
            float(edges[0]),
            1.0e-9,
            0.0,
        )

        upper_bounds = (
            np.inf,
            float(edges[-1]),
            np.inf,
            np.inf,
        )

    else:

        def model(
            x,
            n_events,
            mu,
            sigma,
        ):
            return gaussian_model_counts(
                x,
                n_events,
                mu,
                sigma,
                0.0,
                bin_width,
            )

        p0 = (
            n0,
            mu0,
            sigma0,
        )

        lower_bounds = (
            0.0,
            float(edges[0]),
            1.0e-9,
        )

        upper_bounds = (
            np.inf,
            float(edges[-1]),
            np.inf,
        )

    margin = max(
        1.0e-12,
        1.0e-9
        * (
            edges[-1] - edges[0]
        ),
    )

    p0 = list(p0)

    p0[1] = float(
        np.clip(
            p0[1],
            edges[0] + margin,
            edges[-1] - margin,
        )
    )

    p0[2] = max(
        float(p0[2]),
        1.0e-9,
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                OptimizeWarning,
            )

            popt, _ = curve_fit(
                model,
                centers,
                hist,
                p0=p0,
                bounds=(
                    lower_bounds,
                    upper_bounds,
                ),
                maxfev=50000,
            )

    except (
        RuntimeError,
        ValueError,
    ):
        popt = p0

    if withbaseline:
        n_fit, mu_fit, sigma_fit, baseline_fit = (
            map(float, popt)
        )
    else:
        n_fit, mu_fit, sigma_fit = map(
            float,
            popt,
        )
        baseline_fit = 0.0

    sigma_fit = max(
        abs(sigma_fit),
        1.0e-9,
    )

    yfit = gaussian_model_counts(
        xfit,
        n_fit,
        mu_fit,
        sigma_fit,
        baseline_fit,
        bin_width,
    )

    return (
        xfit,
        yfit,
        n_fit,
        mu_fit,
        sigma_fit,
        baseline_fit,
    )


def gaussian_curve(
    df,
    column,
    binning=None,
    n_points=2048,
    withbaseline=False,
):
    """
    Public Gaussian-fit wrapper.
    """

    return gaussian_model(
        df=df,
        column=column,
        binning=binning,
        n_points=n_points,
        withbaseline=withbaseline,
    )


def langau_curve(
    df,
    column,
    x_scale_factor=1.0,
    binning=None,
    n_points=2048,
):
    """
    Fit a Landau-Gaussian convolution to df[column].

    All quantities used by curve_fit are consistently expressed in
    the scaled coordinate system. The returned MPV, xi and sigma are
    converted back to the original physical scale.
    """

    if x_scale_factor <= 0.0:
        raise ValueError(
            "x_scale_factor must be positive."
        )

    if binning is None:
        binning = make_auto_binning_from_dataframe(
            df,
            column,
        )

    histogram = histogram_from_df(
        df=df,
        column=column,
        binning=binning,
    )

    data = histogram["data"]
    hist = histogram["hist"]
    edges = histogram["edges"]
    centers = histogram["centers"]
    bin_width = histogram["bin_width"]

    xfit = np.linspace(
        edges[0],
        edges[-1],
        int(n_points),
    )

    if data.size < 10 or np.sum(hist) <= 0:
        mpv = float(
            np.median(data)
        ) if data.size else 0.0

        sigma = max(
            float(
                np.std(
                    data,
                    ddof=1,
                )
            )
            if data.size > 1
            else 0.0,
            1.0e-6,
        )

        xi = max(
            0.30 * sigma,
            1.0e-6,
        )

        return (
            xfit,
            np.zeros_like(xfit),
            float(data.size),
            mpv,
            xi,
            sigma,
        )

    nonzero_mask = hist > 0.0

    xhist_physical = centers[
        nonzero_mask
    ]

    yhist = hist[
        nonzero_mask
    ]

    if xhist_physical.size < 5:
        mpv = float(
            np.median(data)
        )

        sigma = max(
            float(
                np.std(
                    data,
                    ddof=1,
                )
            ),
            1.0e-6,
        )

        xi = max(
            0.30 * sigma,
            1.0e-6,
        )

        return (
            xfit,
            np.zeros_like(xfit),
            float(np.sum(hist)),
            mpv,
            xi,
            sigma,
        )

    # Everything below is represented in scaled coordinates.
    xhist = (
        xhist_physical
        / float(x_scale_factor)
    )

    bin_width_scaled = (
        float(bin_width)
        / float(x_scale_factor)
    )

    xmin = float(
        edges[0]
        / float(x_scale_factor)
    )

    xmax = float(
        edges[-1]
        / float(x_scale_factor)
    )

    a0 = float(
        np.sum(yhist)
    )

    mpv0 = float(
        np.median(data)
        / float(x_scale_factor)
    )

    std0 = max(
        float(
            np.std(
                data,
                ddof=1,
            )
        )
        / float(x_scale_factor),
        1.0e-6,
    )

    xi0 = max(
        0.30 * std0,
        1.0e-6,
    )

    sigma0 = max(
        0.20 * std0,
        1.0e-6,
    )

    # Prevent initial MPV from landing on/outside a finite bound.
    margin = max(
        1.0e-9,
        1.0e-9
        * abs(xmax - xmin),
    )

    mpv0 = float(
        np.clip(
            mpv0,
            xmin + margin,
            xmax - margin,
        )
    )

    def langau_model(
        x,
        amplitude,
        mpv,
        xi,
        sigma,
    ):
        xi = max(
            abs(float(xi)),
            1.0e-12,
        )

        sigma = max(
            abs(float(sigma)),
            1.0e-12,
        )

        return (
            amplitude
            * langauss.pdf(
                x,
                landau_x_mpv=mpv,
                landau_xi=xi,
                gauss_sigma=sigma,
            )
            * bin_width_scaled
        )

    lower_bounds = np.array(
        [
            0.0,
            xmin,
            1.0e-9,
            1.0e-9,
        ],
        dtype=float,
    )

    upper_bounds = np.array(
        [
            np.inf,
            xmax,
            np.inf,
            np.inf,
        ],
        dtype=float,
    )

    p0 = np.array(
        [
            a0,
            mpv0,
            xi0,
            sigma0,
        ],
        dtype=float,
    )

    # Guarantee that all starting values lie in their valid bounds.
    p0[0] = max(
        p0[0],
        1.0e-12,
    )

    p0[1] = float(
        np.clip(
            p0[1],
            xmin + margin,
            xmax - margin,
        )
    )

    p0[2] = max(
        p0[2],
        1.0e-9,
    )

    p0[3] = max(
        p0[3],
        1.0e-9,
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                OptimizeWarning,
            )

            popt, _ = curve_fit(
                langau_model,
                xhist,
                yhist,
                p0=p0,
                bounds=(
                    lower_bounds,
                    upper_bounds,
                ),
                maxfev=50000,
            )

    except (
        RuntimeError,
        ValueError,
    ):
        # A controlled fallback is preferable to interrupting
        # the whole simulation due to a single unstable fit.
        popt = p0

    amplitude_scaled, mpv_scaled, xi_scaled, sigma_scaled = map(
        float,
        popt,
    )

    amplitude = amplitude_scaled

    mpv = (
        mpv_scaled
        * float(x_scale_factor)
    )

    xi = (
        xi_scaled
        * float(x_scale_factor)
    )

    sigma = (
        sigma_scaled
        * float(x_scale_factor)
    )

    xfit_scaled = (
        xfit
        / float(x_scale_factor)
    )

    yfit = langau_model(
        xfit_scaled,
        amplitude_scaled,
        mpv_scaled,
        xi_scaled,
        sigma_scaled,
    )

    return (
        xfit,
        yfit,
        amplitude,
        mpv,
        xi,
        sigma,
    )


def mask_theta(
    theta,
    amplitude,
    tail_amplitude,
    sigma,
    cut1,
    cut2,
):
    theta = np.asarray(
        theta,
        dtype=float,
    )

    y = np.zeros_like(
        theta,
        dtype=float,
    )

    sigma = max(
        abs(float(sigma)),
        1.0e-12,
    )

    core_mask = theta <= cut1
    tail_mask = theta >= cut2

    y[core_mask] = (
        amplitude
        * theta[core_mask]
        * np.exp(
            -(
                theta[core_mask] ** 2
            )
            / (
                2.0
                * sigma**2
            )
        )
    )

    positive_tail_mask = (
        tail_mask
        & (theta > 0.0)
    )

    y[positive_tail_mask] = (
        tail_amplitude
        / np.maximum(
            theta[positive_tail_mask],
            1.0e-12,
        ) ** 4
    )

    return y


def theta_model(
    df,
    column,
    binning=None,
    n_points=2048,
):
    """
    Fit the theta distribution with a core and a Rutherford-like tail.
    """

    if binning is None:
        binning = make_auto_binning_from_dataframe(
            df,
            column,
        )

    histogram = histogram_from_df(
        df=df,
        column=column,
        binning=binning,
    )

    data = histogram["data"]
    hist = histogram["hist"]
    edges = histogram["edges"]
    centers = histogram["centers"]

    xfit = np.linspace(
        edges[0],
        edges[-1],
        int(n_points),
    )

    if data.size < 5 or np.sum(hist) <= 0:
        sigma = max(
            float(
                np.std(
                    data,
                    ddof=1,
                )
            )
            if data.size > 1
            else 0.0,
            1.0e-6,
        )

        return (
            xfit,
            np.zeros_like(xfit),
            0.0,
            0.0,
            0.0,
            sigma,
            0.0,
            0.0,
        )

    peak_index = int(
        np.argmax(hist)
    )

    theta_peak = float(
        centers[peak_index]
    )

    if theta_peak <= 0.0:
        positive_centers = centers[
            centers > 0.0
        ]

        theta_peak = (
            float(positive_centers[0])
            if positive_centers.size
            else 1.0e-6
        )

    cut1 = 1.8 * theta_peak
    cut2 = 2.5 * theta_peak

    mask = hist > 0.0

    xhist = centers[mask]
    yhist = hist[mask].astype(float)

    if xhist.size < 5:
        sigma = max(
            float(
                np.std(
                    data,
                    ddof=1,
                )
            )
            if data.size > 1
            else theta_peak,
            1.0e-6,
        )

        return (
            xfit,
            np.zeros_like(xfit),
            theta_peak,
            float(np.max(hist)),
            0.0,
            sigma,
            cut1,
            cut2,
        )

    a0 = max(
        float(np.max(yhist)),
        1.0,
    )

    sigma0 = max(
        theta_peak,
        1.0e-6,
    )

    positive_x = xhist[
        xhist > 0.0
    ]

    if positive_x.size:
        xlast = float(
            positive_x[-1]
        )

        ylast = float(
            yhist[
                xhist > 0.0
            ][-1]
        )

        b0 = max(
            ylast
            * xlast**4,
            1.0e-12,
        )
    else:
        b0 = 1.0e-12

    def model(
        theta,
        amplitude,
        tail_amplitude,
        sigma,
    ):
        return mask_theta(
            theta,
            amplitude,
            tail_amplitude,
            sigma,
            cut1,
            cut2,
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                OptimizeWarning,
            )

            popt, _ = curve_fit(
                model,
                xhist,
                yhist,
                p0=(
                    a0,
                    b0,
                    sigma0,
                ),
                bounds=(
                    (
                        0.0,
                        0.0,
                        1.0e-9,
                    ),
                    (
                        np.inf,
                        np.inf,
                        np.inf,
                    ),
                ),
                maxfev=100000,
            )

    except (
        RuntimeError,
        ValueError,
    ):
        popt = (
            a0,
            b0,
            sigma0,
        )

    amplitude, tail_amplitude, sigma = map(
        float,
        popt,
    )

    sigma = max(
        abs(sigma),
        1.0e-9,
    )

    yfit = mask_theta(
        xfit,
        amplitude,
        tail_amplitude,
        sigma,
        cut1,
        cut2,
    )

    return (
        xfit,
        yfit,
        theta_peak,
        amplitude,
        tail_amplitude,
        sigma,
        cut1,
        cut2,
    )


def fit_theta(
    df,
    column,
    binning=None,
    n_points=2048,
):
    """
    Public wrapper for the standardized theta fit.
    """

    return theta_model(
        df=df,
        column=column,
        binning=binning,
        n_points=n_points,
    )

def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
    buf.seek(0)
    png_bytes = buf.getvalue()
    buf.close()
    plt.close(fig)
    return png_bytes


def triangular_symmetric_curve(
    df,
    column,
    binning=None,
    n_points=2048,
):
    """
    Fit a symmetric triangular model to one histogram.
    """

    if binning is None:
        binning = make_auto_binning_from_dataframe(
            df,
            column,
        )

    histogram = histogram_from_df(
        df=df,
        column=column,
        binning=binning,
    )

    data = histogram["data"]
    hist = histogram["hist"]
    edges = histogram["edges"]
    centers = histogram["centers"]
    bin_width = histogram["bin_width"]

    xfit = np.linspace(
        edges[0],
        edges[-1],
        int(n_points),
    )

    if data.size < 10 or np.sum(hist) <= 0:
        mpv = float(
            np.median(data)
        ) if data.size else 0.0

        xi = max(
            0.5 * (
                edges[-1] - edges[0]
            ),
            bin_width,
            1.0e-6,
        )

        sigma = max(
            xi / np.sqrt(6.0),
            1.0e-6,
        )

        return (
            xfit,
            np.zeros_like(xfit),
            float(data.size),
            mpv,
            xi,
            sigma,
        )

    mask = hist > 0.0

    xhist = centers[mask]
    yhist = hist[mask].astype(float)

    peak_index = int(
        np.argmax(yhist)
    )

    a0 = max(
        float(yhist[peak_index]),
        1.0e-12,
    )

    mpv0 = float(
        xhist[peak_index]
    )

    xi0 = max(
        0.5
        * (
            float(xhist.max())
            - float(xhist.min())
        ),
        bin_width,
        1.0e-6,
    )

    def model(
        x,
        amplitude,
        mpv,
        xi,
    ):
        xi = max(
            abs(float(xi)),
            1.0e-12,
        )

        distance = np.abs(
            x - mpv
        )

        return np.where(
            distance <= xi,
            amplitude
            * (
                1.0
                - distance / xi
            ),
            0.0,
        )

    margin = max(
        1.0e-9,
        1.0e-9
        * (
            edges[-1] - edges[0]
        ),
    )

    mpv0 = float(
        np.clip(
            mpv0,
            edges[0] + margin,
            edges[-1] - margin,
        )
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                OptimizeWarning,
            )

            popt, _ = curve_fit(
                model,
                xhist,
                yhist,
                p0=(
                    a0,
                    mpv0,
                    xi0,
                ),
                bounds=(
                    (
                        0.0,
                        edges[0],
                        1.0e-9,
                    ),
                    (
                        np.inf,
                        edges[-1],
                        np.inf,
                    ),
                ),
                maxfev=20000,
            )

    except (
        RuntimeError,
        ValueError,
    ):
        popt = (
            a0,
            mpv0,
            xi0,
        )

    amplitude, mpv, xi = map(
        float,
        popt,
    )

    sigma = max(
        xi / np.sqrt(6.0),
        1.0e-9,
    )

    yfit = model(
        xfit,
        amplitude,
        mpv,
        xi,
    )

    return (
        xfit,
        yfit,
        amplitude,
        mpv,
        xi,
        sigma,
    )

def get_histogram_and_fit(
    surface_df,
    column,
    ripple_filters,
    particle_type,
    fit_kind=None,
    binning=None,
):
    """
    Build the histogram and optional fit for one variable.

    If binning is None, an automatic binning is built from the
    finite values in the supplied DataFrame and column.
    """
    if binning is None:
        binning = make_auto_binning_from_dataframe(
            surface_df,
            column=column,
        )

    histogram = histogram_from_df(
        surface_df,
        column,
        binning=binning,
    )

    result = {
        "column": column,
        "unit": binning.get("unit"),
        "binning": binning,
        "hist": histogram["hist"],
        "edges": histogram["edges"],
        "centers": histogram["centers"],
        "bin_width": histogram["bin_width"],
        "data": histogram["data"],
        "fit_kind": fit_kind,
        "xfit": None,
        "yfit": None,
        "params": None,
    }

    if fit_kind == "gaussian":
        (
            xfit,
            yfit,
            nfit,
            mu_fit,
            sigma_fit,
            baseline_fit,
        ) = gaussian_model(
            surface_df,
            column=column,
            binning=binning,
            n_points=2048,
            withbaseline=False,
        )

        result["xfit"] = xfit
        result["yfit"] = yfit
        result["params"] = {
            "N": float(nfit),
            "mu": float(mu_fit),
            "sigma": float(sigma_fit),
            "baseline": float(baseline_fit),
        }

    elif fit_kind == "theta":
        (
            xfit,
            yfit,
            theta_peak,
            A_fit,
            B_fit,
            sigma_fit,
            cut1,
            cut2,
        ) = theta_model(
            surface_df,
            column=column,
            binning=binning,
            n_points=2048,
        )

        result["xfit"] = xfit
        result["yfit"] = yfit
        result["params"] = {
            "theta_peak": float(theta_peak),
            "A": float(A_fit),
            "B": float(B_fit),
            "sigma": float(sigma_fit),
            "cut1": float(cut1),
            "cut2": float(cut2),
        }

    elif fit_kind == "langau":
        (
            xfit,
            yfit,
            A_fit,
            mpv_fit,
            xi_fit,
            sigma_fit,
        ) = langau_curve(
            surface_df,
            column=column,
            x_scale_factor=(
                1.0
                if particle_type == "proton"
                else 10.0
            ),
            binning=binning,
            n_points=2048,
        )

        result["xfit"] = xfit
        result["yfit"] = yfit
        result["params"] = {
            "A": float(A_fit),
            "mpv": float(mpv_fit),
            "xi": float(xi_fit),
            "sigma": float(sigma_fit),
        }

    elif fit_kind == "triangular_symmetric":
        (
            xfit,
            yfit,
            A_fit,
            mpv_fit,
            xi_fit,
            sigma_fit,
        ) = triangular_symmetric_curve(
            surface_df,
            column=column,
            binning=binning,
            n_points=2048,
        )

        result["xfit"] = xfit
        result["yfit"] = yfit
        result["params"] = {
            "A": float(A_fit),
            "mpv": float(mpv_fit),
            "xi": float(xi_fit),
            "sigma": float(sigma_fit),
        }

    return result

def histogram_limits_for_surface(
    surface_df_masked,
    surface_df_filtered,
    surface_data_masked_cut,
    ripple_filters,
    particle_type,
    padding_frac=0.06
):
    cols_main = [
        "x - coordinate",
        "y - coordinate",
        "x_prime",
        "y_prime",
        "theta",
        "delta energy",
    ]

    cols_var = [
        "delta_xp",
        "delta_yp",
        "delta_energy",
    ]

    limits = {"main": {}, "var": {}}

    for col in cols_main:
        if col in surface_df_masked.columns:
            df = surface_df_masked
        else:
            continue

        binning = make_auto_binning_from_dataframe(surface_df_masked, column=col)

        res = get_histogram_and_fit(
            surface_df_masked,
            column=col,
            ripple_filters = ripple_filters,
            particle_type= particle_type,
            fit_kind=None,
            binning=binning
        )

        hist = res["hist"]
        edges = res["edges"]

        if edges.size < 2 or hist.size == 0:
            continue

        xmin = float(edges[0])
        xmax = float(edges[-1])
        ymax = float(np.max(hist)) if np.max(hist) > 0 else 1.0
        dx = xmax - xmin if xmax > xmin else 1.0

        limits["main"][col] = {
            "xlim": (xmin - padding_frac * dx, xmax + padding_frac * dx),
            "ylim": (0.0, ymax + padding_frac * ymax),
        }

    for col in cols_var:
        if col not in surface_data_masked_cut.columns:
            continue

        binning = make_auto_binning_from_dataframe(surface_data_masked_cut, column=col)

        res = get_histogram_and_fit(
            surface_data_masked_cut,
            column=col,
            ripple_filters = ripple_filters,
            particle_type= particle_type,
            fit_kind=None,
            binning=binning
        )

        hist = res["hist"]
        edges = res["edges"]

        if edges.size < 2 or hist.size == 0:
            continue

        xmin = float(edges[0])
        xmax = float(edges[-1])
        ymax = float(np.max(hist)) if np.max(hist) > 0 else 1.0
        dx = xmax - xmin if xmax > xmin else 1.0

        limits["var"][col] = {
            "xlim": (xmin - padding_frac * dx, xmax + padding_frac * dx),
            "ylim": (0.0, ymax + padding_frac * ymax),
        }

    return limits


def plot_histograms_surface(
    surface_df_masked,
    surface_df_filtered,
    surface_data_masked_cut,
    z_position,
    theta_peak,
    A,
    B,
    sigma,
    label,
    ripple_filters,
    particle_type,
    positions_scoring,
    show=True,
    axis_limits=None, 
    binning = None,
):

    def set_ylim_from_res(ax, res, safety_factor=1.10):
        hist = res["hist"]
        yfit = res["yfit"]

        ymax_hist = float(hist.max()) if hist is not None and hist.size else 0.0
        ymax_fit = float(yfit.max()) if yfit is not None and len(yfit) > 0 else 0.0

        ymax = max(ymax_hist, ymax_fit, 1.0)
        ax.set_ylim(0.0, safety_factor * ymax)

    def set_safe_ylim_from_ax(ax, margin=0.10):
        ymax = 0.0

        for line in ax.lines:
            y = np.asarray(line.get_ydata(), dtype=float)
            if y.size:
                y = y[np.isfinite(y)]
                if y.size:
                    ymax = max(ymax, float(np.max(y)))

        for patch in ax.patches:
            try:
                ymax = max(ymax, float(patch.get_height()))
            except Exception:
                pass

        ymax = max(ymax, 1.0)
        ax.set_ylim(0.0, ymax * (1.0 + margin))

    def apply_limits(ax, limits_dict, col, apply_x=True, apply_y=False):
        if limits_dict is not None and col in limits_dict:
            if apply_x and "xlim" in limits_dict[col]:
                ax.set_xlim(limits_dict[col]["xlim"])
            if apply_y and "ylim" in limits_dict[col]:
                ax.set_ylim(limits_dict[col]["ylim"])

    fig, axes = plt.subplots(3, 2, figsize=(17, 24))
    fig.suptitle(
        f'Histograms regarding the scoring surface at {z_position} cm in {label}',
        fontsize=16
    )
    axes = axes.flatten()

    # X
    res_x = get_histogram_and_fit(
        surface_df_masked,
        "x - coordinate",
        ripple_filters =ripple_filters,
        particle_type= particle_type,
        fit_kind="gaussian",
        binning=make_auto_binning_from_dataframe(surface_df_masked, column="x - coordinate"),
    )

    axes[0].hist(
        res_x["data"],
        bins=res_x["binning"]["edges"],
        density=False,
        color='blue',
        alpha=0.7,
        label='x data'
    )

    if res_x["xfit"] is not None:
        axes[0].plot(res_x["xfit"], res_x["yfit"], 'r-', lw=2, label='Gaussian fit')

    axes[0].set_title("Distribution of X")
    axes[0].set_xlabel("X [mm]")
    axes[0].set_ylabel("Counts")
    axes[0].grid(axis='y', alpha=0.5)
    axes[0].legend()

    apply_limits(
        axes[0],
        axis_limits["main"] if axis_limits else None,
        "x - coordinate",
        apply_x=True,
        apply_y=False
    )
    if axis_limits is None or "x - coordinate" not in axis_limits.get("main", {}):
        axes[0].set_xlim(res_x["binning"]["range"])

    set_ylim_from_res(axes[0], res_x)

    # Y
    res_y = get_histogram_and_fit(
        surface_df_masked,
        "y - coordinate",
        ripple_filters = ripple_filters,
        particle_type= particle_type,
        fit_kind="gaussian",
        binning=make_auto_binning_from_dataframe(surface_df_masked, column="y - coordinate")
    )   

    axes[1].hist(
        res_y["data"],
        bins=res_y["binning"]["edges"],
        density=False,
        color='green',
        alpha=0.7,
        label='y data'
    )

    if res_y["xfit"] is not None:
        axes[1].plot(res_y["xfit"], res_y["yfit"], 'r-', lw=2, label='Gaussian fit')

    axes[1].set_title("Distribution of Y")
    axes[1].set_xlabel("Y [mm]")
    axes[1].set_ylabel("Counts")
    axes[1].grid(axis='y', alpha=0.5)
    axes[1].legend()

    apply_limits(
        axes[1],
        axis_limits["main"] if axis_limits else None,
        "y - coordinate",
        apply_x=True,
        apply_y=False
    )
    if axis_limits is None or "y - coordinate" not in axis_limits.get("main", {}):
        axes[1].set_xlim(res_y["binning"]["range"])

    set_ylim_from_res(axes[1], res_y)

    # X'
    res_xp = get_histogram_and_fit(
        surface_df_masked,
        "x_prime",
        ripple_filters = ripple_filters,
        particle_type= particle_type,
        fit_kind="gaussian",
        binning=make_auto_binning_from_dataframe(surface_df_masked, column="x_prime")
    )

    axes[2].hist(
        res_xp["data"],
        bins=res_xp["binning"]["edges"],
        density=False,
        color='red',
        alpha=0.7,
        label="x' data"
    )

    if res_xp["xfit"] is not None:
        axes[2].plot(res_xp["xfit"], res_xp["yfit"], 'k-', lw=2, label='Gaussian fit')

    axes[2].set_title("Distribution of X'")
    axes[2].set_xlabel("X' [mrad]")
    axes[2].set_ylabel("Counts")
    axes[2].grid(axis='y', alpha=0.5)
    axes[2].legend()

    apply_limits(
        axes[2],
        axis_limits["main"] if axis_limits else None,
        "x_prime",
        apply_x=True,
        apply_y=False
    )
    if axis_limits is None or "x_prime" not in axis_limits.get("main", {}):
        axes[2].set_xlim(res_xp["binning"]["range"])

    set_ylim_from_res(axes[2], res_xp)

    # Y'
    res_yp = get_histogram_and_fit(
        surface_df_masked,
        "y_prime",
        ripple_filters = ripple_filters,
        particle_type= particle_type,
        fit_kind="gaussian",
        binning=make_auto_binning_from_dataframe(surface_df_masked, column="y_prime")
    )

    axes[3].hist(
        res_yp["data"],
        bins=res_yp["binning"]["edges"],
        density=False,
        color='purple',
        alpha=0.7,
        label="y' data"
    )

    if res_yp["xfit"] is not None:
        axes[3].plot(res_yp["xfit"], res_yp["yfit"], 'k-', lw=2, label='Gaussian fit')

    axes[3].set_title("Distribution of Y'")
    axes[3].set_xlabel("Y' [mrad]")
    axes[3].set_ylabel("Counts")
    axes[3].grid(axis='y', alpha=0.5)
    axes[3].legend()

    apply_limits(
        axes[3],
        axis_limits["main"] if axis_limits else None,
        "y_prime",
        apply_x=True,
        apply_y=False
    )
    if axis_limits is None or "y_prime" not in axis_limits.get("main", {}):
        axes[3].set_xlim(res_yp["binning"]["range"])

    set_ylim_from_res(axes[3], res_yp)

    # Theta
    res_theta = get_histogram_and_fit(
        surface_df_masked,
        "theta",
        ripple_filters = ripple_filters,
        particle_type= particle_type,
        fit_kind="theta",
        binning=make_auto_binning_from_dataframe(surface_df_masked, column="theta")
    )

    axes[4].hist(
        res_theta["data"],
        bins=res_theta["binning"]["edges"],
        density=False,
        color='cyan',
        alpha=0.7,
        label='theta data'
    )

    if res_theta["xfit"] is not None:
        axes[4].plot(
            res_theta["xfit"],
            res_theta["yfit"],
            'r-',
            label='LinGaus + 1/theta^4 fit'
        )

        A_fit = res_theta["params"]["A"]
        sigma_fit = res_theta["params"]["sigma"]
        B_fit = res_theta["params"]["B"]
        theta_peak_fit = res_theta["params"]["theta_peak"]

        def lin_gauss(theta_arr, A_loc, sigma_loc):
            theta_arr = np.asarray(theta_arr, dtype=float)
            return A_loc * theta_arr * np.exp(-theta_arr**2 / (2.0 * sigma_loc**2))

        def theta_tail(theta_arr, B_loc):
            theta_arr = np.asarray(theta_arr, dtype=float)
            out = np.zeros_like(theta_arr, dtype=float)
            mask = theta_arr > 0
            out[mask] = B_loc / np.maximum(theta_arr[mask], 1e-12)**4
            return out

        xs = res_theta["xfit"]

        A_use = A if A is not None else A_fit
        sigma_use = sigma if sigma is not None else sigma_fit
        ys_gauss = lin_gauss(xs, A_use, sigma_use)
        axes[4].plot(xs, ys_gauss, 'g--', label='Linear Gaussian only')

        B_use = B if B is not None else B_fit
        theta_peak_use = theta_peak if theta_peak is not None else theta_peak_fit

        if theta_peak_use is not None and B_use is not None:
            mask_tail = xs >= max(theta_peak_use * 1.5, 1e-9)
            xs_tail = xs[mask_tail]
            if xs_tail.size > 0:
                ys_tail = theta_tail(xs_tail, B_use)
                axes[4].plot(xs_tail, ys_tail, 'm-.', label='1/theta^4 only')

    axes[4].set_title("Distribution of theta")
    axes[4].set_xlabel("Theta [mrad]")
    axes[4].set_ylabel("Counts")
    axes[4].grid(axis='y', alpha=0.5)
    axes[4].legend()

    apply_limits(
        axes[4],
        axis_limits["main"] if axis_limits else None,
        "theta",
        apply_x=True,
        apply_y=False
    )
    if axis_limits is None or "theta" not in axis_limits.get("main", {}):
        axes[4].set_xlim(res_theta["binning"]["range"])

    set_safe_ylim_from_ax(axes[4])

    first_surface = positions_scoring[0]
    is_first_surface = np.isclose(z_position, first_surface, rtol=0, atol=1e-9)
    second_surface = positions_scoring[1]
    is_second_surface = np.isclose(z_position, second_surface, rtol=0, atol=1e-9)

    if ripple_filters and (not is_first_surface):
        energy_fit_kind_int = "triangular_symmetric"
        energy_fit_label_int = "Triangular symmetric fit"
    else:
        energy_fit_kind_int = "langau"
        energy_fit_label_int = "Langau fit"


    if ripple_filters and is_second_surface:
        energy_fit_kind_diff = "triangular_symmetric"
        energy_fit_label_diff = "Triangular symmetric fit"
    else:
        energy_fit_kind_diff = "langau"
        energy_fit_label_diff = "Langau fit"


    # Energy
    res_de = get_histogram_and_fit(
        surface_df_masked,
        "delta energy",
        ripple_filters = ripple_filters,
        particle_type= particle_type,
        fit_kind=energy_fit_kind_int,
        binning=make_auto_binning_from_dataframe(surface_df_masked, column="delta energy")
    )
    print("\n--- Delta energy diagnostic ---")
    print("Range binning:", res_de["binning"]["range"])
    print("Primo e ultimo edge:", res_de["edges"][0], res_de["edges"][-1])
    print("Min dato:", np.min(res_de["data"]))
    print("Max dato:", np.max(res_de["data"]))
    print("Numero bin:", len(res_de["hist"]))
    print("Numero bin non vuoti:", np.count_nonzero(res_de["hist"]))

    axes[5].hist(
        res_de["data"],
        bins=res_de["binning"]["edges"],
        density=False,
        color='orange',
        alpha=0.7,
        label='delta energy data'
    )

    if res_de["xfit"] is not None:
        axes[5].plot(res_de["xfit"], res_de["yfit"], 'r-', label=energy_fit_label_int)

    axes[5].set_title("Distribution of delta energy")
    xlabel = "Delta energy [MeV]"
    axes[5].set_xlabel(xlabel)
    axes[5].set_ylabel("Counts")
    axes[5].grid(axis='y', alpha=0.5)
    axes[5].legend()

    apply_limits(
        axes[5],
        axis_limits["main"] if axis_limits else None,
        "delta energy",
        apply_x=True,
        apply_y=False
    )
    if axis_limits is None or "delta energy" not in axis_limits.get("main", {}):
        axes[5].set_xlim(res_de["binning"]["range"])

    set_ylim_from_res(axes[5], res_de)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if show:
        plt.show()

    fig2, axes2 = plt.subplots(1, 4, figsize=(22, 7))
    fig2.suptitle(
        f'Variation histograms regarding the scoring surface at {z_position} cm in {label}',
        fontsize=16
    )
    axes2 = axes2.flatten()

    # X' variation
    res_dxp = get_histogram_and_fit(
        surface_data_masked_cut,
        "delta_xp",
        ripple_filters = ripple_filters,
        particle_type= particle_type,
        fit_kind="gaussian",
        binning=make_auto_binning_from_dataframe(surface_data_masked_cut, column="delta_xp")
    )

    axes2[0].hist(
        res_dxp["data"],
        bins=res_dxp["binning"]["edges"],
        density=False,
        color='red',
        alpha=0.7,
        label='delta x prime data'
    )

    if res_dxp["xfit"] is not None:
        axes2[0].plot(res_dxp["xfit"], res_dxp["yfit"], 'k-', lw=2, label='Gaussian fit')

    axes2[0].set_title("Distribution of variation of X'")
    axes2[0].set_xlabel("delta X' [mrad]")
    axes2[0].set_ylabel("Counts")
    axes2[0].grid(axis='y', alpha=0.5)
    axes2[0].legend()

    apply_limits(
        axes2[0],
        axis_limits["var"] if axis_limits else None,
        "delta_xp",
        apply_x=True,
        apply_y=False
    )
    if axis_limits is None or "delta_xp" not in axis_limits.get("var", {}):
        axes2[0].set_xlim(res_dxp["binning"]["range"])

    set_ylim_from_res(axes2[0], res_dxp)

    # Y' variation
    res_dyp = get_histogram_and_fit(
        surface_data_masked_cut,
        "delta_yp",
        ripple_filters = ripple_filters,
        particle_type= particle_type,
        fit_kind="gaussian",
        binning=make_auto_binning_from_dataframe(surface_data_masked_cut, column="delta_yp")
    )

    axes2[1].hist(
        res_dyp["data"],
        bins=res_dyp["binning"]["edges"],
        density=False,
        color='purple',
        alpha=0.7,
        label='delta y prime data'
    )

    if res_dyp["xfit"] is not None:
        axes2[1].plot(res_dyp["xfit"], res_dyp["yfit"], 'k-', lw=2, label='Gaussian fit')

    axes2[1].set_title("Distribution of variation of Y'")
    axes2[1].set_xlabel("delta Y' [mrad]")
    axes2[1].set_ylabel("Counts")
    axes2[1].grid(axis='y', alpha=0.5)
    axes2[1].legend()

    apply_limits(
        axes2[1],
        axis_limits["var"] if axis_limits else None,
        "delta_yp",
        apply_x=True,
        apply_y=False
    )
    if axis_limits is None or "delta_yp" not in axis_limits.get("var", {}):
        axes2[1].set_xlim(res_dyp["binning"]["range"])

    set_ylim_from_res(axes2[1], res_dyp)

    #Theta variation
    res_dtheta = get_histogram_and_fit(
        surface_data_masked_cut,
        "delta_theta",
        ripple_filters = ripple_filters,
        particle_type= particle_type,
        fit_kind="theta",
        binning=make_auto_binning_from_dataframe(surface_data_masked_cut,column="delta_theta")
    )

    axes2[2].hist(
        res_dtheta["data"],
        bins=res_dtheta["binning"]["edges"],
        density=False,
        color='blue',
        alpha=0.7,
        label='delta theta data'
    )

    if res_dtheta["xfit"] is not None:
        axes2[2].plot(res_dtheta["xfit"], res_dtheta["yfit"], 'k-', lw=2, label='LinGauss fit')

    axes2[2].set_title("Distribution of variation of theta")
    axes2[2].set_xlabel("delta theta [mrad]")
    axes2[2].set_ylabel("Counts")
    axes2[2].grid(axis='y', alpha=0.5)
    axes2[2].legend()

    apply_limits(
        axes2[2],
        axis_limits["var"] if axis_limits else None,
        "delta_theta",
        apply_x=True,
        apply_y=False
    )
    if axis_limits is None or "delta_theta" not in axis_limits.get("var", {}):
        axes2[2].set_xlim(res_dtheta["binning"]["range"])

    set_ylim_from_res(axes2[2], res_dtheta)

    # Delta energy variation
    #energy_fit_kind_delta = "triangular_symmetric" if ripple_filters else "langau"
    res_dde = get_histogram_and_fit(
        surface_data_masked_cut,
        "delta_energy",
        ripple_filters = ripple_filters,
        particle_type= particle_type,
        fit_kind=energy_fit_kind_diff,
        binning=make_auto_binning_from_dataframe(surface_data_masked_cut, column="delta_energy")
    )

    axes2[3].hist(
        res_dde["data"],
        bins=res_dde["binning"]["edges"],
        density=False,
        color='orange',
        alpha=0.7,
        label='delta energy data'
    )

    if res_dde["xfit"] is not None:
        axes2[3].plot(res_dde["xfit"], res_dde["yfit"], 'r-', label=energy_fit_label_diff)

    axes2[3].set_title("Distribution of variation of energy")
    xlabel = "Variation of delta energy [MeV]"
    axes2[3].set_xlabel(xlabel)
    axes2[3].set_ylabel("Counts")
    axes2[3].grid(axis='y', alpha=0.5)
    axes2[3].legend()

    apply_limits(
        axes2[3],
        axis_limits["var"] if axis_limits else None,
        "delta_energy",
        apply_x=True,
        apply_y=False
    )
    if axis_limits is None or "delta_energy" not in axis_limits.get("var", {}):
        axes2[3].set_xlim(res_dde["binning"]["range"])

    set_ylim_from_res(axes2[3], res_dde)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if show:
        plt.show()

    return fig, fig2


def build_histogram_frames(
    stored_surface_plots,
    surface_df_masked,
    surface_df_filtered,
    surface_data_masked_cut,
    ripple_filters, 
    particle_type,
    positions_scoring
):
    frames_main = []
    frames_var = []

    axis_limits = histogram_limits_for_surface(
        surface_df_masked,
        surface_df_filtered,
        surface_data_masked_cut,
        ripple_filters = ripple_filters,
        particle_type= particle_type
    )

    for item in stored_surface_plots:
        fig1, fig2 = plot_histograms_surface(
            surface_df_masked=item["surface_df_masked"],
            surface_df_filtered=item["surface_df_filtered"],
            surface_data_masked_cut=item["surface_data_masked_cut"],
            z_position=item["z_position"],
            theta_peak=item.get("theta_peak", None),
            A=item.get("A", None),
            B=item.get("B", None),
            sigma=item.get("sigma", None),
            label=item.get("label", "Dataset"),
            ripple_filters=ripple_filters,
            particle_type= particle_type,
            positions_scoring=positions_scoring,
            show=False,
            axis_limits=axis_limits
        )

        frames_main.append(fig_to_png_bytes(fig1))
        frames_var.append(fig_to_png_bytes(fig2))

    return frames_main, frames_var


def show_surface_plots_slider(stored_surface_plots,  surface_df_masked, surface_df_filtered, surface_data_masked_cut, ripple_filters, particle_type, positions_scoring,show = True):
    frames_main, frames_var = build_histogram_frames(stored_surface_plots, surface_df_masked, surface_df_filtered, surface_data_masked_cut, ripple_filters,particle_type, positions_scoring=positions_scoring)

    options = [(f"{item['z_position']} cm", i) for i, item in enumerate(stored_surface_plots)]

    if not show: 
        return{"frames_main": frames_main,
               "frames_var": frames_var,
               "options":options}

    return {"frames_main": frames_main,
            "frames_var": frames_var,
            "options":options
            }


def get_density_colors(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return x, y, np.array([])

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    return x[idx], y[idx], z[idx]


def compute_distribution_global_limits(stored_distribution_plots, padding_frac=0.08):
    limits = {}

    pairs = {
        "x_xp": ("x - coordinate", "x_prime"),
        "y_yp": ("y - coordinate", "y_prime"),
        "xp_yp": ("x_prime", "y_prime"),
        "x_y": ("x - coordinate", "y - coordinate"),
    }

    for key, (colx, coly) in pairs.items():
        xs_all = []
        ys_all = []

        for item in stored_distribution_plots:
            df_orig = item["surface_df_original"]
            df_filt = item["surface_df_masked"]

            x1 = df_orig[colx].dropna().values
            y1 = df_orig[coly].dropna().values
            x2 = df_filt[colx].dropna().values
            y2 = df_filt[coly].dropna().values

            if len(x1) > 0:
                xs_all.append(x1)
            if len(x2) > 0:
                xs_all.append(x2)
            if len(y1) > 0:
                ys_all.append(y1)
            if len(y2) > 0:
                ys_all.append(y2)

        xcat = np.concatenate(xs_all)
        ycat = np.concatenate(ys_all)

        xmin, xmax = np.nanmin(xcat), np.nanmax(xcat)
        ymin, ymax = np.nanmin(ycat), np.nanmax(ycat)

        dx = xmax - xmin
        dy = ymax - ymin

        limits[key] = {
            "xlim": (xmin - padding_frac * dx, xmax + padding_frac * dx),
            "ylim": (ymin - padding_frac * dy, ymax + padding_frac * dy),
        }

    return limits


def set_zoom_limits(ax_ins, x, y, zoom_frac):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return

    x0 = np.median(x)
    y0 = np.median(y)

    dx = np.nanmax(x) - np.nanmin(x)
    dy = np.nanmax(y) - np.nanmin(y)

    if dx == 0:
        dx = 1.0
    if dy == 0:
        dy = 1.0

    half_x = 0.5 * zoom_frac * dx
    half_y = 0.5 * zoom_frac * dy

    ax_ins.set_xlim(x0 - half_x, x0 + half_x)
    ax_ins.set_ylim(y0 - half_y, y0 + half_y)



def plot_distribution(
    surface_df_original,
    surface_df_filtered,
    z_pos,
    label,
    results_filtered,
    show=True,
    axis_limits=None
):
    alpha_stat_x = float(results_filtered["alpha_x"])
    beta_stat_x = float(results_filtered["beta_x"])
    epsilon_stat_x = float(results_filtered["emittance_x"])

    alpha_stat_y = float(results_filtered["alpha_y"])
    beta_stat_y = float(results_filtered["beta_y"])
    epsilon_stat_y = float(results_filtered["emittance_y"])

    x_stat_ellipse_x, y_stat_ellipse_x = twiss_ellipse_points(
        alpha=alpha_stat_x,
        beta=beta_stat_x,
        epsilon=epsilon_stat_x,
        x0=0.0,
        xp0=0.0,
        npts=600
    )

    x_stat_ellipse_y, y_stat_ellipse_y = twiss_ellipse_points(
        alpha=alpha_stat_y,
        beta=beta_stat_y,
        epsilon=epsilon_stat_y,
        x0=0.0,
        xp0=0.0,
        npts=600
    )

    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle(f'Distributions regarding the scoring surface at {z_pos} cm', fontsize=16)

    plot_df_filtered = surface_df_filtered.sample(
        n=min(len(surface_df_filtered), 25000),
        random_state=42
    )
    plot_df_original = surface_df_original.sample(
        n=min(len(surface_df_original), 25000),
        random_state=42
    )

    # =========================
    # X vs X'
    # =========================
    axes[0, 0].set_title("Phase space X vs X'")
    axes[0, 0].set_xlabel("X [mm]")
    axes[0, 0].set_ylabel("X' [mrad]")
    axes[0, 0].grid(True, linestyle='--')
    axes[0, 0].set_xlim(-50, 50)
    axes[0, 0].set_ylim(-50, 50)
    axes[0, 0].set_aspect('equal')

    axes[0, 0].scatter(
        plot_df_original['x - coordinate'].values,
        plot_df_original['x_prime'].values,
        c='0.8',
        s=5,
        alpha=0.5,
        label='Original dataset'
    )

    x_c, x_prime_c, z_c = get_density_colors(
        plot_df_filtered['x - coordinate'].values,
        plot_df_filtered['x_prime'].values
    )

    if len(z_c) > 0:
        axes[0, 0].scatter(
            x_c, x_prime_c,
            c=z_c,
            s=5,
            cmap='viridis',
            label='Filtered dataset'
        )

    axes[0, 0].plot(
        x_stat_ellipse_x,
        y_stat_ellipse_x,
        color="tab:red",
        lw=2.2,
        ls="-",
        alpha=0.9,
        label="Twiss ellipse"
    )

    axes[0, 0].legend(loc='upper left', markerscale=2)

    ax_ins = inset_axes(axes[0, 0], width="35%", height="35%", loc="upper right")
    ax_ins.scatter(x_c, x_prime_c, c=z_c, s=5, cmap='viridis')
    ax_ins.plot(
        x_stat_ellipse_x,
        y_stat_ellipse_x,
        color="tab:red",
        lw=2.2,
        ls="-",
        alpha=0.9
    )
    set_zoom_limits(
        ax_ins,
        np.concatenate([x_c, x_stat_ellipse_x]),
        np.concatenate([x_prime_c, y_stat_ellipse_x]),
        zoom_frac=0.25
    )
    ax_ins.grid(True, linestyle='--', alpha=0.4)
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])

    # =========================
    # Y vs Y'
    # =========================
    axes[0, 1].set_title("Phase space Y vs Y'")
    axes[0, 1].set_xlabel("Y [mm]")
    axes[0, 1].set_ylabel("Y' [mrad]")
    axes[0, 1].grid(True, linestyle='--')
    axes[0, 1].set_xlim(-50, 50)
    axes[0, 1].set_ylim(-50, 50)
    axes[0, 1].set_aspect('equal')

    axes[0, 1].scatter(
        plot_df_original['y - coordinate'].values,
        plot_df_original['y_prime'].values,
        c='0.8',
        s=5,
        alpha=0.5,
        label='Original dataset'
    )

    y_c, y_prime_c, z_c = get_density_colors(
        plot_df_filtered['y - coordinate'].values,
        plot_df_filtered['y_prime'].values
    )

    if len(z_c) > 0:
        axes[0, 1].scatter(
            y_c, y_prime_c,
            c=z_c,
            s=5,
            cmap='viridis',
            label='Filtered dataset'
        )

    axes[0, 1].plot(
        x_stat_ellipse_y,
        y_stat_ellipse_y,
        color="tab:red",
        lw=2.2,
        ls="-",
        alpha=0.9,
        label="Twiss ellipse"
    )

    axes[0, 1].legend(loc='upper left', markerscale=2)

    ax_ins = inset_axes(axes[0, 1], width="35%", height="35%", loc="upper right")
    ax_ins.scatter(y_c, y_prime_c, c=z_c, s=5, cmap='viridis')
    ax_ins.plot(
        x_stat_ellipse_y,
        y_stat_ellipse_y,
        color="tab:red",
        lw=2.2,
        ls="-",
        alpha=0.9
    )
    set_zoom_limits(
        ax_ins,
        np.concatenate([y_c, x_stat_ellipse_y]),
        np.concatenate([y_prime_c, y_stat_ellipse_y]),
        zoom_frac=0.25
    )
    ax_ins.grid(True, linestyle='--', alpha=0.4)
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])

    # =========================
    # X' vs Y'
    # =========================
    axes[1, 0].set_title("X' vs Y'")
    axes[1, 0].set_xlabel("X' [mrad]")
    axes[1, 0].set_ylabel("Y' [mrad]")
    axes[1, 0].grid(True, linestyle='--')
    axes[1, 0].set_aspect('equal')
    axes[1, 0].set_xlim(-50, 50)
    axes[1, 0].set_ylim(-50, 50)

    axes[1, 0].scatter(
        plot_df_original['x_prime'].values,
        plot_df_original['y_prime'].values,
        c='0.8',
        s=5,
        alpha=0.25,
        label='Original dataset'
    )

    x_p, y_p, z_p = get_density_colors(
        plot_df_filtered['x_prime'].values,
        plot_df_filtered['y_prime'].values
    )

    axes[1, 0].scatter(
        x_p, y_p,
        c=z_p,
        s=5,
        cmap='cividis',
        label='Filtered dataset'
    )

    axes[1, 0].legend(loc='upper left', markerscale=2)

    ax_ins = inset_axes(axes[1, 0], width="35%", height="35%", loc="upper right")
    ax_ins.scatter(x_p, y_p, c=z_p, s=5, cmap='cividis')
    set_zoom_limits(ax_ins, x_p, y_p, zoom_frac=0.25)
    ax_ins.set_aspect('equal')
    ax_ins.grid(True, linestyle='--', alpha=0.4)
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])

    # =========================
    # X vs Y
    # =========================
    axes[1, 1].set_title('Beam profile')
    axes[1, 1].set_xlabel('X [mm]')
    axes[1, 1].set_ylabel('Y [mm]')
    axes[1, 1].grid(True, linestyle='--')
    axes[1, 1].set_aspect('equal')
    axes[1, 1].set_xlim(-50, 50)
    axes[1, 1].set_ylim(-50, 50)

    axes[1, 1].scatter(
        plot_df_original['x - coordinate'].values,
        plot_df_original['y - coordinate'].values,
        c='0.8',
        s=5,
        alpha=0.5,
        label='Original dataset'
    )

    x_p, y_p, z_p = get_density_colors(
        plot_df_filtered['x - coordinate'].values,
        plot_df_filtered['y - coordinate'].values
    )

    axes[1, 1].scatter(
        x_p, y_p,
        c=z_p,
        s=5,
        cmap='viridis',
        label='Filtered dataset'
    )

    axes[1, 1].legend(loc='upper left', markerscale=2)

    ax_ins = inset_axes(axes[1, 1], width="35%", height="35%", loc="upper right")
    ax_ins.scatter(x_p, y_p, c=z_p, s=5, cmap='viridis')
    set_zoom_limits(ax_ins, x_p, y_p, zoom_frac=0.55)
    ax_ins.set_aspect('equal')
    ax_ins.grid(True, linestyle='--', alpha=0.4)
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if show:
        plt.show()

    return fig



def build_distribution_frames(stored_distribution_plots, results_filtered):
    frames = []

    axis_limits = compute_distribution_global_limits(stored_distribution_plots)

    for item, res in zip(stored_distribution_plots, results_filtered):
        fig = plot_distribution(
            item["surface_df_original"],
            item["surface_df_masked"],
            item["z_position"],
            item["label"],
            res,
            show=False,
            axis_limits=axis_limits
        )

        frames.append(fig_to_png_bytes(fig))
        plt.close(fig)

    return frames

def show_distribution_slider(stored_distribution_plots, results_filtered, show=True):
    frames = build_distribution_frames(stored_distribution_plots, results_filtered)

    options = [(f"{item['z_position']} cm", i) for i, item in enumerate(stored_distribution_plots)]

    if not show:
        return {
            "frames_main": frames,
            "options": options
        }

    return frames

def plot_evolution_fit (result_list_original, result_list_filtered, result_list_masked, show = True):

    z_positions = [r['z_position'] for r in result_list_original]
    counts = [r['counts'] for r in result_list_original]

    mu_x = [r['mu_x'] for r in result_list_filtered]
    sigma_x = [r['sigma_x'] for r in result_list_filtered]
    mu_y = [r['mu_y'] for r in result_list_filtered]
    sigma_y = [r['sigma_y'] for r in result_list_filtered]

    mu_xp = [r['mu_xp'] for r in result_list_filtered]
    sigma_xp = [r['sigma_xp'] for r in result_list_filtered]
    mu_yp = [r['mu_yp'] for r in result_list_filtered]
    sigma_yp = [r['sigma_yp'] for r in result_list_filtered]

    mu_delta_energy = [r['mu_delta_energy'] for r in result_list_filtered]

    sigma_theta = [r['sigma_theta'] for r in result_list_filtered]

    emittance_x = [r['emittance_x'] for r in result_list_masked]
    emittance_y = [r['emittance_y'] for r in result_list_masked]

    alpha_x = [r['alpha_x'] for r in result_list_masked]
    alpha_y = [r['alpha_y'] for r in result_list_masked]

    beta_x = [r['beta_x'] for r in result_list_masked]
    beta_y = [r['beta_y'] for r in result_list_masked]

    gamma_x = [r['gamma_x'] for r in result_list_masked]
    gamma_y = [r['gamma_y'] for r in result_list_masked]


    fig1, axes = plt.subplots(3, 2, figsize=(12, 11))
    fig1.suptitle('Evolution of the beam along the different scoring surfaces', fontsize=16)

    # Survival
    axes[0, 0].plot(z_positions, counts, 'o-', color='black')
    axes[0, 0].set_title('Particle survival')
    axes[0, 0].set_xlabel('Scoring position [cm]')
    axes[0, 0].set_ylabel('Number of particles')
    axes[0, 0].grid(True)

    # Mean of the delta energy
    axes[0, 1].plot(z_positions, mu_delta_energy, 'o-', color='cyan', label ='Mean theta')
    axes[0, 1].set_title('Mean value of delta energy')
    axes[0, 1].set_xlabel('Scoring position [cm]')
    axes[0, 1].set_ylabel('Mean energy [MeV]')
    axes[0, 1].grid(True)

    # RMS X and Y
    axes[1, 0].plot(z_positions, sigma_x, 'o--', color='blue', label='std(x)')
    axes[1, 0].plot(z_positions, sigma_y, '*-', color='green', label='std(y)')
    axes[1, 0].set_title('STD of X and Y')
    axes[1, 0].set_xlabel('Scoring position [cm]')
    axes[1, 0].set_ylabel('Standard deviation [mm]')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    # Mean X and Y
    axes[1, 1].plot(z_positions, mu_x, 'o--', color='blue')
    axes[1, 1].plot(z_positions, mu_y, '*-', color='green')
    axes[1, 1].set_title('Mean of X and Y')
    axes[1, 1].set_xlabel('Scoring position [cm]')
    axes[1, 1].set_ylabel('Mean valued [mm]')
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    # RMS X' and Y'
    axes[2, 0].plot(z_positions, sigma_xp, 'o--', color='blue', label='std(x)')
    axes[2, 0].plot(z_positions, sigma_yp, '*-', color='green', label='std(y)')
    axes[2, 0].plot(z_positions, sigma_theta, 'o-', color='cyan', label ='Mean theta')
    axes[2, 0].set_title("STD of X', Y' and theta")
    axes[2, 0].set_xlabel('Scoring position [cm]')
    axes[2, 0].set_ylabel('Standard deviation [mrad]')
    axes[2, 0].grid(True)
    axes[2, 0].legend()
    
    # Mean X and Y
    axes[2, 1].plot(z_positions, mu_xp, 'o--', color='blue')
    axes[2, 1].plot(z_positions, mu_yp, '*-', color='green')
    axes[2, 1].set_title("'Mean of X' and Y'")
    axes[2, 1].set_xlabel('Scoring position [cm]')
    axes[2, 1].set_ylabel('Mean valued [mm]')
    axes[2, 1].grid(True)
    axes[2, 1].legend()

    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    if show: 
        plt.show()

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 11))
    fig2.suptitle('Evolution of the Twiss parameters along the different scoring surfaces', fontsize=16)

    # Emittance X vs Y
    axes2[0, 0].plot(z_positions, emittance_x, 'o--', color='blue', label='std(x)')
    axes2[0, 0].plot(z_positions, emittance_y, '*-', color='green', label='std(y)')
    axes2[0, 0].set_title('Emittance in X vs Y')
    axes2[0, 0].set_xlabel('Scoring position [cm]')
    axes2[0, 0].set_ylabel('Emittance [mm*mrad]')
    axes2[0, 0].grid(True)
    axes2[0, 0].legend()

    # Alpha X vs Y
    axes2[0, 1].plot(z_positions, alpha_x, 'o--', color='blue', label='std(x)')
    axes2[0, 1].plot(z_positions, alpha_y, '*-', color='green', label='std(y)')
    axes2[0, 1].set_title('Alpha in X vs Y')
    axes2[0, 1].set_xlabel('Scoring position [cm]')
    axes2[0, 1].set_ylabel('Alpha []')
    axes2[0, 1].grid(True)
    axes2[0, 1].legend()

    # Beta X vs Y
    axes2[1, 0].plot(z_positions, beta_x, 'o--', color='blue', label='std(x)')
    axes2[1, 0].plot(z_positions, beta_y, '*-', color='green', label='std(y)')
    axes2[1, 0].set_title('Beta in X vs Y')
    axes2[1, 0].set_xlabel('Scoring position [cm]')
    axes2[1, 0].set_ylabel('Beta [m]')
    axes2[1, 0].grid(True)
    axes2[1, 0].legend()

    # Gamma X vs Y
    axes2[1, 1].plot(z_positions, gamma_x, 'o--', color='blue', label='std(x)')
    axes2[1, 1].plot(z_positions, gamma_y, '*-', color='green', label='std(y)')
    axes2[1, 1].set_title('Gamma in X vs Y')
    axes2[1, 1].set_xlabel('Scoring position [cm]')
    axes2[1, 1].set_ylabel('Gamma [1/m]')
    axes2[1, 1].grid(True)
    axes2[1, 1].legend()
    

    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    if show: 
        plt.show()

    return fig1, fig2


def _plot_one_surface_twiss_evolution(
    cut_percentages,
    cut_labels,
    alpha_x_cuts,
    alpha_y_cuts,
    alpha_x_statistics,
    alpha_y_statistics,
    beta_x_cuts,
    beta_y_cuts,
    beta_x_statistics,
    beta_y_statistics,
    gamma_x_cuts,
    gamma_y_cuts,
    gamma_x_statistics,
    gamma_y_statistics,
    emittance_x_cuts,
    emittance_y_cuts,
    emittance_x_statistics,
    emittance_y_statistics,
    surface_label=None,
    axis_limits=None,
    show=False,
):
    """
    Plot Twiss parameters at one scoring surface as a function
    of the KDE elliptical-cut percentage.

    Solid blue curve:
        Twiss parameter calculated from particles selected in x-x'.

    Solid orange curve:
        Twiss parameter calculated from particles selected in y-y'.

    Horizontal dashed lines:
        Twiss parameters obtained from the standard statistical
        beam analysis at the same scoring surface.
    """

    cut_percentages = np.asarray(
        cut_percentages,
        dtype=float,
    )

    cut_labels = list(
        cut_labels
    )

    alpha_x_cuts = np.asarray(
        alpha_x_cuts,
        dtype=float,
    )

    alpha_y_cuts = np.asarray(
        alpha_y_cuts,
        dtype=float,
    )

    beta_x_cuts = np.asarray(
        beta_x_cuts,
        dtype=float,
    )

    beta_y_cuts = np.asarray(
        beta_y_cuts,
        dtype=float,
    )

    gamma_x_cuts = np.asarray(
        gamma_x_cuts,
        dtype=float,
    )

    gamma_y_cuts = np.asarray(
        gamma_y_cuts,
        dtype=float,
    )

    emittance_x_cuts = np.asarray(
        emittance_x_cuts,
        dtype=float,
    )

    emittance_y_cuts = np.asarray(
        emittance_y_cuts,
        dtype=float,
    )

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(13, 9),
        sharex=True,
    )

    axs = axs.ravel()

    curve_style_x = {
        "color": "tab:blue",
        "marker": "s",
        "linewidth": 2.4,
        "markersize": 6,
    }

    curve_style_y = {
        "color": "tab:orange",
        "marker": "o",
        "linewidth": 2.0,
        "markersize": 5,
    }

    stat_style_x = {
        "color": "tab:blue",
        "linestyle": "--",
        "linewidth": 2.0,
        "alpha": 0.85,
    }

    stat_style_y = {
        "color": "tab:orange",
        "linestyle": "--",
        "linewidth": 1.7,
        "alpha": 0.85,
    }

    configs = [
        {
            "axis": axs[0],
            "title": "Twiss α",
            "ylabel": "α",
            "x_values": alpha_x_cuts,
            "y_values": alpha_y_cuts,
            "x_statistics": alpha_x_statistics,
            "y_statistics": alpha_y_statistics,
        },
        {
            "axis": axs[1],
            "title": "Twiss β",
            "ylabel": "β [mm/mrad]",
            "x_values": beta_x_cuts,
            "y_values": beta_y_cuts,
            "x_statistics": beta_x_statistics,
            "y_statistics": beta_y_statistics,
        },
        {
            "axis": axs[2],
            "title": "Twiss γ",
            "ylabel": "γ [mrad/mm]",
            "x_values": gamma_x_cuts,
            "y_values": gamma_y_cuts,
            "x_statistics": gamma_x_statistics,
            "y_statistics": gamma_y_statistics,
        },
        {
            "axis": axs[3],
            "title": "Emittance ε",
            "ylabel": "ε [mm·mrad]",
            "x_values": emittance_x_cuts,
            "y_values": emittance_y_cuts,
            "x_statistics": emittance_x_statistics,
            "y_statistics": emittance_y_statistics,
        },
    ]

    for config in configs:
        ax = config["axis"]

        ax.plot(
            cut_percentages,
            config["x_values"],
            label="KDE elliptical cut x-x'",
            **curve_style_x,
        )

        ax.plot(
            cut_percentages,
            config["y_values"],
            label="KDE elliptical cut y-y'",
            **curve_style_y,
        )

        ax.axhline(
            float(config["x_statistics"]),
            label="Statistical x-x'",
            **stat_style_x,
        )

        ax.axhline(
            float(config["y_statistics"]),
            label="Statistical y-y'",
            **stat_style_y,
        )

        ax.set_title(
            config["title"]
        )

        ax.set_ylabel(
            config["ylabel"]
        )

        ax.grid(
            True,
            alpha=0.30,
        )

        ax.legend(
            fontsize=8,
            loc="best",
        )

    for ax in axs:
        ax.set_xticks(
            cut_percentages
        )

        ax.set_xticklabels(
            cut_labels,
            rotation=0,
        )

        ax.set_xlabel(
            "KDE contour level [% of peak density]"
        )

    if axis_limits is not None:
        if axis_limits.get("alpha") is not None:
            axs[0].set_ylim(
                axis_limits["alpha"]
            )

        if axis_limits.get("beta") is not None:
            axs[1].set_ylim(
                axis_limits["beta"]
            )

        if axis_limits.get("gamma") is not None:
            axs[2].set_ylim(
                axis_limits["gamma"]
            )

        if axis_limits.get("epsilon") is not None:
            axs[3].set_ylim(
                axis_limits["epsilon"]
            )

    title = (
        "Twiss parameters versus KDE elliptical-cut level"
    )

    if surface_label is not None:
        title += f" — {surface_label}"

    fig.suptitle(
        title,
        fontsize=15,
    )

    fig.tight_layout(
        rect=[0.0, 0.03, 1.0, 0.95]
    )

    if show:
        plt.show()

    return fig


def build_twiss_fit_frames(
    statistics_reference,
    twiss_cuts_dataframe,
):
    """
    Build one PNG frame per scoring surface.

    Each frame shows alpha, beta, gamma and emittance as a
    function of KDE cut percentage, for both x-x' and y-y'.
    """

    if (
        twiss_cuts_dataframe is None
        or twiss_cuts_dataframe.empty
    ):
        raise ValueError(
            "twiss_cuts_dataframe is empty."
        )

    if "z_position" not in twiss_cuts_dataframe.columns:
        raise KeyError(
            "twiss_cuts_dataframe requires the "
            "'z_position' column."
        )

    if isinstance(
        statistics_reference,
        dict,
    ):
        statistics_reference = list(
            statistics_reference.values()
        )

    elif isinstance(
        statistics_reference,
        tuple,
    ):
        statistics_reference = list(
            statistics_reference
        )

    elif not isinstance(
        statistics_reference,
        list,
    ):
        statistics_reference = [
            statistics_reference
        ]

    cut_levels = [
        ("90", 90.0, "90%"),
        ("75", 75.0, "75%"),
        ("6065", 60.65, "60.65%"),
        ("50", 50.0, "50%"),
        ("10", 10.0, "10%"),
    ]

    cut_percentages = np.asarray(
        [
            item[1]
            for item in cut_levels
        ],
        dtype=float,
    )

    cut_labels = [
        item[2]
        for item in cut_levels
    ]

    parameter_prefixes = {
        "alpha": "alpha",
        "beta": "beta",
        "gamma": "gamma",
        "epsilon": "epsilon",
    }

    required_columns = [
        "z_position",
    ]

    for parameter in parameter_prefixes:
        for plane in ("x", "y"):
            for suffix, _, _ in cut_levels:
                required_columns.append(
                    f"{parameter}{plane}{suffix}"
                )

    missing_columns = [
        column
        for column in required_columns
        if column not in twiss_cuts_dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing KDE-cut columns in "
            "twiss_cuts_dataframe:\n"
            + "\n".join(
                missing_columns
            )
        )

    stats_by_z = {}

    for item in statistics_reference:
        if not isinstance(item, dict):
            continue

        if "z_position" not in item:
            continue

        z_position = float(
            item["z_position"]
        )

        stats_by_z[z_position] = item

    frames = []
    options = []

    for _, row in twiss_cuts_dataframe.iterrows():
        z_position = float(
            row["z_position"]
        )

        matching_stats = None

        for z_stats, stats in stats_by_z.items():
            if np.isclose(
                z_position,
                z_stats,
                rtol=0.0,
                atol=1e-9,
            ):
                matching_stats = stats
                break

        if matching_stats is None:
            print(
                f"WARNING: no reference Twiss statistics "
                f"found for z={z_position} cm. "
                "This surface is skipped."
            )
            continue

        alpha_x_cuts = np.asarray(
            [
                float(
                    row[f"alphax{suffix}"]
                )
                for suffix, _, _ in cut_levels
            ],
            dtype=float,
        )

        alpha_y_cuts = np.asarray(
            [
                float(
                    row[f"alphay{suffix}"]
                )
                for suffix, _, _ in cut_levels
            ],
            dtype=float,
        )

        beta_x_cuts = np.asarray(
            [
                float(
                    row[f"betax{suffix}"]
                )
                for suffix, _, _ in cut_levels
            ],
            dtype=float,
        )

        beta_y_cuts = np.asarray(
            [
                float(
                    row[f"betay{suffix}"]
                )
                for suffix, _, _ in cut_levels
            ],
            dtype=float,
        )

        gamma_x_cuts = np.asarray(
            [
                float(
                    row[f"gammax{suffix}"]
                )
                for suffix, _, _ in cut_levels
            ],
            dtype=float,
        )

        gamma_y_cuts = np.asarray(
            [
                float(
                    row[f"gammay{suffix}"]
                )
                for suffix, _, _ in cut_levels
            ],
            dtype=float,
        )

        emittance_x_cuts = np.asarray(
            [
                float(
                    row[f"epsilonx{suffix}"]
                )
                for suffix, _, _ in cut_levels
            ],
            dtype=float,
        )

        emittance_y_cuts = np.asarray(
            [
                float(
                    row[f"epsilony{suffix}"]
                )
                for suffix, _, _ in cut_levels
            ],
            dtype=float,
        )

        fig = _plot_one_surface_twiss_evolution(
            cut_percentages=cut_percentages,
            cut_labels=cut_labels,

            alpha_x_cuts=alpha_x_cuts,
            alpha_y_cuts=alpha_y_cuts,
            alpha_x_statistics=float(
                matching_stats["alpha_x"]
            ),
            alpha_y_statistics=float(
                matching_stats["alpha_y"]
            ),

            beta_x_cuts=beta_x_cuts,
            beta_y_cuts=beta_y_cuts,
            beta_x_statistics=float(
                matching_stats["beta_x"]
            ),
            beta_y_statistics=float(
                matching_stats["beta_y"]
            ),

            gamma_x_cuts=gamma_x_cuts,
            gamma_y_cuts=gamma_y_cuts,
            gamma_x_statistics=float(
                matching_stats["gamma_x"]
            ),
            gamma_y_statistics=float(
                matching_stats["gamma_y"]
            ),

            emittance_x_cuts=emittance_x_cuts,
            emittance_y_cuts=emittance_y_cuts,
            emittance_x_statistics=float(
                matching_stats["emittance_x"]
            ),
            emittance_y_statistics=float(
                matching_stats["emittance_y"]
            ),

            surface_label=f"{z_position:.3f} cm",

            axis_limits=None,

            show=False,
        )

        frames.append(
            fig_to_png_bytes(
                fig
            )
        )

        plt.close(
            fig
        )

        options.append(
            (
                f"{z_position:.3f} cm",
                len(options),
            )
        )

    if not frames:
        raise RuntimeError(
            "No Twiss KDE-cut frames were generated. "
            "Check matching z_position values and "
            "Twiss input keys."
        )

    return frames, options

def plot_comparison(results_masked_vacuum, results_masked_air, results_original_vacuum, results_original_air, show = True):
    
    z_positions = [r['z_position'] for r in results_masked_vacuum]

    mu_x_vacuum = [r["mu_x"] for r in results_masked_vacuum]
    mu_x_air = [r["mu_x"] for r in results_masked_air]
    mu_y_vacuum = [r["mu_y"] for r in results_masked_vacuum]
    mu_y_air = [r["mu_y"] for r in results_masked_air]

    mu_xp_vacuum = [r["mu_xp"] for r in results_masked_vacuum]
    mu_xp_air = [r["mu_xp"] for r in results_masked_air]
    mu_yp_vacuum = [r["mu_yp"] for r in results_masked_vacuum]
    mu_yp_air = [r["mu_yp"] for r in results_masked_air]

    sigma_x_vacuum = [r["sigma_x"] for r in results_masked_vacuum]
    sigma_x_air = [r["sigma_x"] for r in results_masked_air]
    sigma_y_vacuum = [r["sigma_y"] for r in results_masked_vacuum]
    sigma_y_air = [r["sigma_y"] for r in results_masked_air]
    
    sigma_xp_vacuum = [r["sigma_xp"] for r in results_masked_vacuum]
    sigma_xp_air = [r["sigma_xp"] for r in results_masked_air]
    sigma_yp_vacuum = [r["sigma_yp"] for r in results_masked_vacuum]
    sigma_yp_air = [r["sigma_yp"] for r in results_masked_air] 

    mu_delta_energy_air = [r["mu_delta_energy"] for r in results_masked_air]
    mu_delta_energy_vacuum = [r["mu_delta_energy"] for r in results_masked_vacuum]

    sigma_theta_air = [r["sigma_theta"] for r in results_masked_air]
    sigma_theta_vacuum = [r["sigma_theta"] for r in results_masked_vacuum]

    emittance_x_air = [r["emittance_x"] for r in results_masked_air]
    emittance_x_vacuum = [r["emittance_x"] for r in results_masked_vacuum]

    emittance_y_air = [r["emittance_y"] for r in results_masked_air]
    emittance_y_vacuum = [r["emittance_y"] for r in results_masked_vacuum]

    alpha_x_air = [r["alpha_x"] for r in results_masked_air]
    alpha_x_vacuum = [r["alpha_x"] for r in results_masked_vacuum]

    alpha_y_air = [r["alpha_y"] for r in results_masked_air]
    alpha_y_vacuum = [r["alpha_y"] for r in results_masked_vacuum]

    beta_x_air = [r["beta_x"] for r in results_masked_air]
    beta_x_vacuum = [r["beta_x"] for r in results_masked_vacuum]

    beta_y_air = [r["beta_y"] for r in results_masked_air]
    beta_y_vacuum = [r["beta_y"] for r in results_masked_vacuum]

    gamma_x_air = [r["gamma_x"] for r in results_masked_air]
    gamma_x_vacuum = [r["gamma_x"] for r in results_masked_vacuum]

    gamma_y_air = [r["gamma_y"] for r in results_masked_air]
    gamma_y_vacuum = [r["gamma_y"] for r in results_masked_vacuum]

    counts_vacuum = [r['counts'] for r in results_original_vacuum]
    counts_air = [r['counts'] for r in results_original_air]

    fig1, axes1 = plt.subplots(3, 2, figsize=(18, 18))
    fig1.suptitle('Comparison between Vacuum and Air datasets', fontsize=16)

    # STD X
    axes1[0, 0].plot(z_positions, sigma_x_vacuum, 'o-', label='STD of X in Vacuum', color='blue')
    axes1[0, 0].plot(z_positions, sigma_x_air, 'o--', label='STD of X in Air', color='cyan')
    axes1[0, 0].plot(z_positions, sigma_y_vacuum, 'o-', label='STD of Y in Vacuum', color='red')
    axes1[0, 0].plot(z_positions, sigma_y_air, 'o--', label='STD of Y in Air', color='magenta')
    axes1[0, 0].set_title('Standard deviation in X and Y')
    axes1[0, 0].set_xlabel('Scoring Position (cm)')
    axes1[0, 0].set_ylabel('Sigma [mm]')
    axes1[0, 0].legend()
    axes1[0, 0].grid(True)

    # Mean X
    axes1[0, 1].plot(z_positions, mu_x_vacuum, 'o-', label='Mean of X in Vacuum', color='green')
    axes1[0, 1].plot(z_positions, mu_x_air, 'o--', label='Mean of X in Air', color='lime')
    axes1[0, 1].plot(z_positions, mu_y_vacuum, 'o-', label='Mean of Y in Vacuum', color='purple')
    axes1[0, 1].plot(z_positions, mu_y_air, 'o--', label='Mean of Y in Air', color='pink')
    axes1[0, 1].set_title('Mean value of X and Y')
    axes1[0, 1].set_xlabel('Scoring Position (cm)')
    axes1[0, 1].set_ylabel('Mean value [mm]')
    axes1[0, 1].legend()
    axes1[0, 1].grid(True)

    # STD X'
    axes1[1, 0].plot(z_positions, sigma_xp_vacuum, 'o-', label='STD of X\' in Vacuum', color='blue')
    axes1[1, 0].plot(z_positions, sigma_xp_air, 'o--', label='STD of X\' in Air', color='cyan')
    axes1[1, 0].plot(z_positions, sigma_yp_vacuum, 'o-', label='STD of Y\' in Vacuum', color='red')
    axes1[1, 0].plot(z_positions, sigma_yp_air, 'o--', label='STD of Y\' in Air', color='magenta')
    axes1[1, 0].set_title("Standard deviation in X' and Y'")
    axes1[1, 0].set_xlabel('Scoring Position (cm)')
    axes1[1, 0].set_ylabel("Sigma [mrad]")
    axes1[1, 0].legend()
    axes1[1, 0].grid(True)

    # Mean X'
    axes1[1, 1].plot(z_positions, mu_xp_vacuum, 'o-', label='Mean of X\' in Vacuum', color='green')
    axes1[1, 1].plot(z_positions, mu_xp_air, 'o--', label='Mean of X\' in Air', color='lime')
    axes1[1, 1].plot(z_positions, mu_yp_vacuum, 'o-', label='Mean of Y\' in Vacuum', color='purple')
    axes1[1, 1].plot(z_positions, mu_yp_air, 'o--', label='Mean of Y\' in Air', color='pink')
    axes1[1, 1].set_title("Mean value of X' and Y'")
    axes1[1, 1].set_xlabel('Scoring Position (cm)')
    axes1[1, 1].set_ylabel("Mean value [mrad]")
    axes1[1, 1].legend()
    axes1[1, 1].grid(True)

    # STD theta'
    axes1[2, 0].plot(z_positions, sigma_theta_vacuum, 'o-', label='Vacuum', color='blue')
    axes1[2, 0].plot(z_positions, sigma_theta_air, 'o--', label='Air', color='cyan')
    axes1[2, 0].set_title("STD of theta'")
    axes1[2, 0].set_xlabel('Scoring Position (cm)')
    axes1[2, 0].set_ylabel("Sigma of theta [mrad]")
    axes1[2, 0].legend()
    axes1[2, 0].grid(True)
   
   # Mean value delta energy
    axes1[2, 1].plot(z_positions, mu_delta_energy_vacuum, 'o-', label='Vacuum', color='green')
    axes1[2, 1].plot(z_positions, mu_delta_energy_air, 'o--', label='Air', color='lime')
    axes1[2, 1].set_title("Mean value of the delta energy")
    axes1[2, 1].set_xlabel('Scoring Position (cm)')
    axes1[2, 1].set_ylabel("Mean value of energy [MeV]")
    axes1[2, 1].legend()
    axes1[2, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if show: 
        plt.show



    fig2, axes2 = plt.subplots(3, 2, figsize=(12, 18))
    fig2.suptitle('Comparison between Vacuum and Air datasets', fontsize=16)

    # Emittance X
    axes2[0, 0].plot(z_positions, emittance_x_vacuum, 'o-', label='Emittance in X in Vacuum', color='green')
    axes2[0, 0].plot(z_positions, emittance_x_air, 'o-', label='Emittance in X in Air', color='lime')
    axes2[0, 0].plot(z_positions, emittance_y_vacuum, 'o--', label='Emittance in Y in Vacuum', color='red')
    axes2[0, 0].plot(z_positions, emittance_y_air, 'o--', label='Emittance in Y in Air', color='orange')
    axes2[0, 0].set_title("Emittance in X and Y")
    axes2[0, 0].set_xlabel('Scoring Position (cm)')
    axes2[0, 0].set_ylabel("Emittance [mm*mrad]")
    axes2[0, 0].legend()
    axes2[0, 0].grid(True)

    # Alpha X
    axes2[0, 1].plot(z_positions, alpha_x_vacuum, 'o-', label='Alpha in X in Vacuum', color='green')
    axes2[0, 1].plot(z_positions, alpha_x_air, 'o-', label='Alpha in X in Air', color='lime')
    axes2[0, 1].plot(z_positions, alpha_y_vacuum, 'o--', label='Alpha in Y in Vacuum', color='red')
    axes2[0, 1].plot(z_positions, alpha_y_air, 'o--', label='Alpha in Y in Air', color='orange')
    axes2[0, 1].set_title("Alpha in X and Y")
    axes2[0, 1].set_xlabel('Scoring Position (cm)')
    axes2[0, 1].set_ylabel("Alpha []")
    axes2[0, 1].legend()
    axes2[0, 1].grid(True)

    # Beta X
    axes2[1, 0].plot(z_positions, beta_x_vacuum, 'o-', label='Beta in X in Vacuum', color='green')
    axes2[1, 0].plot(z_positions, beta_x_air, 'o-', label='Beta in X in Air', color='lime')
    axes2[1, 0].plot(z_positions, beta_y_vacuum, 'o--', label='Beta in Y in Vacuum', color='red')
    axes2[1, 0].plot(z_positions, beta_y_air, 'o--', label='Beta in Y in Air', color='orange')
    axes2[1, 0].set_title("Beta in X and Y")
    axes2[1, 0].set_xlabel('Scoring Position (cm)')
    axes2[1, 0].set_ylabel("Beta [m]")
    axes2[1, 0].legend()
    axes2[1, 0].grid(True)

    # Gamma X
    axes2[1, 1].plot(z_positions, gamma_x_vacuum, 'o-', label='Gamma in X in Vacuum', color='green')
    axes2[1, 1].plot(z_positions, gamma_x_air, 'o-', label='Gamma in X in Air', color='lime')
    axes2[1, 1].plot(z_positions, gamma_y_vacuum, 'o--', label=' Gamma in Y in Vacuum', color='red')
    axes2[1, 1].plot(z_positions, gamma_y_air, 'o--', label='Gamma in Y in Air', color='orange')
    axes2[1, 1].set_title("Gamma in X and Y")
    axes2[1, 1].set_xlabel('Scoring Position (cm)')
    axes2[1, 1].set_ylabel("Gamma [1/m]")
    axes2[1, 1].legend()
    axes2[1, 1].grid(True)


    # Survival
    axes2[2, 0].plot(z_positions, counts_vacuum, 'o-', label='Vacuum', color='green')
    axes2[2, 0].plot(z_positions, counts_air, 'o--', label='Air', color='lime')
    axes2[2, 0].set_title('Particle survival')
    axes2[2, 0].set_xlabel('Scoring Position (cm)')
    axes2[2, 0].set_ylabel('Number of particles')
    axes2[2, 0].legend()
    axes2[2, 0].grid(True)

    axes2[2, 1].axis('off')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if show: 
        plt.show()

    return fig1, fig2

try:
    from sklearn.covariance import MinCovDet
    _HAS_SKLEARN = True
except Exception:  # sklearn non installato
    _HAS_SKLEARN = False


def _estimate_center_cov(vf, robust=True, mcd_support_fraction=None,
                         random_state=0):
    
    if robust and _HAS_SKLEARN:
        mcd = MinCovDet(support_fraction=mcd_support_fraction,
                        random_state=random_state).fit(vf)
        return mcd.location_, mcd.covariance_

    if robust and not _HAS_SKLEARN:
        mu = np.median(vf, axis=0)
        d = vf - mu
        cov = (d.T @ d) / (len(d) - 1)
        return mu, cov

    mu = vf.mean(axis=0)
    cov = np.cov(vf, rowvar=False)
    return mu, cov

def percentile_cut(percentile_ranges, df, cols):
    if not cols or df.empty:
        return df.copy()

    df_out = df.copy()
    for col in cols:
        p_low, p_high = percentile_ranges[col]
        col_data = df_out[col].dropna().values
        low_c = np.percentile(col_data, p_low)
        high_c = np.percentile(col_data, p_high)
        mask_col = (df_out[col] >= low_c) & (df_out[col] <= high_c)
        df_out.loc[~mask_col, col] = np.nan
    return df_out

def theta_percentile_cut(
    df,
    percentile_min=0.0,
    percentile_max=99.0,
    theta_column="theta",
):
    """
    Keep only rows whose theta value lies between the selected
    percentiles, inclusive.

    Unlike percentile_cut(), this function removes rejected rows
    instead of replacing a single column with NaN.

    The input dataframe is never modified.
    """

    if df is None:
        raise ValueError(
            "df cannot be None."
        )

    if df.empty:
        return df.copy()

    if theta_column not in df.columns:
        raise KeyError(
            f"Missing theta column '{theta_column}'."
        )

    percentile_min = float(
        percentile_min
    )

    percentile_max = float(
        percentile_max
    )

    if (
        percentile_min < 0.0
        or percentile_max > 100.0
        or percentile_min >= percentile_max
    ):
        raise ValueError(
            "Theta percentiles must satisfy "
            "0 <= min < max <= 100."
        )

    theta_values = pd.to_numeric(
        df[theta_column],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    finite_mask = np.isfinite(
        theta_values
    )

    if np.count_nonzero(finite_mask) == 0:
        return df.iloc[0:0].copy()

    finite_theta = theta_values[
        finite_mask
    ]

    theta_min = float(
        np.percentile(
            finite_theta,
            percentile_min,
        )
    )

    theta_max = float(
        np.percentile(
            finite_theta,
            percentile_max,
        )
    )

    keep_mask = (
        finite_mask
        & (theta_values >= theta_min)
        & (theta_values <= theta_max)
    )

    df_out = df.loc[
        keep_mask
    ].copy()

    print(
        f"Theta cut [{percentile_min:.2f}%, "
        f"{percentile_max:.2f}%]: "
        f"theta in [{theta_min:.6g}, "
        f"{theta_max:.6g}], "
        f"kept {len(df_out)}/{len(df)} rows."
    )

    return df_out

def ellipse_cut(
    df,
    pairs,
    peak_fraction,
    robust=True,
    mcd_support_fraction=None,
):
    """
    Apply a true row-wise elliptical cut.

    A particle is retained only if it belongs to every requested
    phase-space ellipse. Rows outside at least one ellipse are removed.

    No percentile cut is applied here.
    """

    if df.empty or not pairs:
        return df.copy()

    dfout = df.copy()

    keep_mask = np.ones(
        len(dfout),
        dtype=bool,
    )

    r2_limit = -2.0 * np.log(
        float(peak_fraction)
    )

    for column_1, column_2 in pairs:
        if (
            column_1 not in dfout.columns
            or column_2 not in dfout.columns
        ):
            continue

        values = dfout[
            [
                column_1,
                column_2,
            ]
        ].to_numpy(
            dtype=float,
        )

        finite = np.isfinite(values).all(
            axis=1
        )

        # If there are too few valid particles, do not attempt a
        # covariance estimate for that phase-space projection.
        if np.count_nonzero(finite) < 10:
            keep_mask &= finite
            continue

        values_finite = values[finite]

        center, covariance = _estimate_center_cov(
            values_finite,
            robust=robust,
            mcd_support_fraction=mcd_support_fraction,
        )

        try:
            covariance_inv = np.linalg.inv(
                covariance
            )
        except np.linalg.LinAlgError:
            covariance_inv = np.linalg.pinv(
                covariance
            )

        difference = values - center

        mahalanobis_squared = np.einsum(
            "ij,jk,ik->i",
            difference,
            covariance_inv,
            difference,
        )

        inside = (
            finite
            & np.isfinite(mahalanobis_squared)
            & (mahalanobis_squared <= r2_limit)
        )

        keep_mask &= inside

    return dfout.loc[
        keep_mask
    ].copy()

def apply_dataset_cleaning_cut(
    df,
    cut_mode="ellipse",
    theta_percentile_min=None,
    theta_percentile_max=None,
    ellipse_peak_fraction=None,
    angle_percentile_min=None,
    angle_percentile_max=None
):
    """
    Apply exactly one final cleaning cut.

    cut_mode="ellipse":
        Apply only the phase-space elliptical cut.
        Theta percentiles are ignored and may be None.

    cut_mode="theta":
        Apply only the theta percentile cut.
        ellipse_peak_fraction is ignored and may be None.
    """

    if df is None:
        raise ValueError(
            "Input dataframe cannot be None."
        )

    if df.empty:
        return df.copy()

    cut_mode = str(
        cut_mode
    ).strip().lower()

    # ============================================================
    # ELLIPTICAL CUT ONLY
    # ============================================================

    if cut_mode == "ellipse":
        if ellipse_peak_fraction is None:
            raise ValueError(
                "ellipse_peak_fraction is required when "
                "cut_mode='ellipse'."
            )

        ellipse_peak_fraction = float(
            ellipse_peak_fraction
        )

        if (
            ellipse_peak_fraction <= 0.0
            or ellipse_peak_fraction >= 1.0
        ):
            raise ValueError(
                "ellipse_peak_fraction must satisfy "
                "0 < ellipse_peak_fraction < 1."
            )

        return ellipse_cut(
            df,
            pairs=[
                (
                    "x - coordinate",
                    "x_prime",
                ),
                (
                    "y - coordinate",
                    "y_prime",
                ),
            ],
            peak_fraction=ellipse_peak_fraction,
            robust=True,
        )

    # ============================================================
    # THETA PERCENTILE CUT ONLY
    # ============================================================

    if cut_mode == "theta":
        if "theta" not in df.columns:
            raise KeyError(
                "Column 'theta' is required when "
                "cut_mode='theta'."
            )

        if theta_percentile_min is None:
            raise ValueError(
                "theta_percentile_min is required when "
                "cut_mode='theta'."
            )

        if theta_percentile_max is None:
            raise ValueError(
                "theta_percentile_max is required when "
                "cut_mode='theta'."
            )

        theta_percentile_min = float(
            theta_percentile_min
        )

        theta_percentile_max = float(
            theta_percentile_max
        )

        if (
            theta_percentile_min < 0.0
            or theta_percentile_max > 100.0
            or theta_percentile_min >= theta_percentile_max
        ):
            raise ValueError(
                "Theta percentiles must satisfy "
                "0 <= min < max <= 100."
            )

        theta_values = pd.to_numeric(
            df["theta"],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        finite_mask = np.isfinite(
            theta_values
        )

        if np.count_nonzero(finite_mask) == 0:
            return df.iloc[0:0].copy()

        theta_values_finite = theta_values[
            finite_mask
        ]

        theta_lower = float(
            np.percentile(
                theta_values_finite,
                theta_percentile_min,
            )
        )

        theta_upper = float(
            np.percentile(
                theta_values_finite,
                theta_percentile_max,
            )
        )

        keep_mask = (
            finite_mask
            & (theta_values >= theta_lower)
            & (theta_values <= theta_upper)
        )

        df_theta_cleaned = df.loc[
            keep_mask
        ].copy()

        print(
            f"Theta cut: "
            f"[{theta_percentile_min:.3f}%, "
            f"{theta_percentile_max:.3f}%] | "
            f"theta range: "
            f"[{theta_lower:.6g}, "
            f"{theta_upper:.6g}] | "
            f"kept {len(df_theta_cleaned)}/{len(df)} rows."
        )
    # ============================================================
    # ANGLE PERCENTILE CUT ONLY
    # ============================================================   

    if cut_mode == "angle_percentile":
        
        missing_columns = [
            column
            for column in ("x_prime", "y_prime")
            if column not in df.columns
        ]

        if missing_columns:
            raise KeyError(
                "Missing columns required when "
                "cut_mode='angle_percentile': "
                f"{missing_columns}."
            )

        if angle_percentile_min is None:
            raise ValueError(
                "angle_percentile_min is required when "
                "cut_mode='angle_percentile'."
            )

        if angle_percentile_max is None:
            raise ValueError(
                "angle_percentile_max is required when "
                "cut_mode='angle_percentile'."
            )

        angle_percentile_min = float(
            angle_percentile_min
        )

        angle_percentile_max = float(
            angle_percentile_max
        )

        if (
            not np.isfinite(angle_percentile_min)
            or not np.isfinite(angle_percentile_max)
            or not (
                0.0
                <= angle_percentile_min
                < angle_percentile_max
                <= 100.0
            )
        ):
            raise ValueError(
                "Angle percentiles must be finite and satisfy "
                "0 <= min < max <= 100."
            )

        xp_values = pd.to_numeric(
            df["x_prime"],
            errors="coerce",
        ).to_numpy(dtype=float)

        yp_values = pd.to_numeric(
            df["y_prime"],
            errors="coerce",
        ).to_numpy(dtype=float)

        finite_mask = (
            np.isfinite(xp_values)
            & np.isfinite(yp_values)
        )

        if not np.any(finite_mask):
            return df.iloc[0:0].copy()

        xp_lower, xp_upper = np.percentile(
            xp_values[finite_mask],
            [
                angle_percentile_min,
                angle_percentile_max,
            ],
        )

        yp_lower, yp_upper = np.percentile(
            yp_values[finite_mask],
            [
                angle_percentile_min,
                angle_percentile_max,
            ],
        )

        keep_mask = (
            finite_mask
            & (xp_values >= xp_lower)
            & (xp_values <= xp_upper)
            & (yp_values >= yp_lower)
            & (yp_values <= yp_upper)
        )

        df_angle_cleaned = df.loc[
            keep_mask
        ].copy()

        print(
            f"Angular percentile cut: "
            f"[{angle_percentile_min:.3f}%, "
            f"{angle_percentile_max:.3f}%] | "
            f"x_prime range: "
            f"[{xp_lower:.6g}, {xp_upper:.6g}] mrad | "
            f"y_prime range: "
            f"[{yp_lower:.6g}, {yp_upper:.6g}] mrad | "
            f"kept {len(df_angle_cleaned)}/{len(df)} rows."
        )

        return df_angle_cleaned

    raise ValueError(
        f"Unsupported cut_mode={cut_mode!r}. "
        "Use 'ellipse' or 'theta' or 'angle_percentile'."
    )


def calculate_variations(df, z_position, label):
    surface_df = df[df['scoring_surface'] == z_position].copy()

    positions = sorted(df['scoring_surface'].unique())
    idx = positions.index(z_position)

    if idx == 0:
        surface_df['delta_xp'] = surface_df['x_prime'].values
        surface_df['delta_yp'] = surface_df['y_prime'].values
        surface_df['delta_energy'] = surface_df['delta energy'].values
        surface_df['delta_theta'] = surface_df['theta'].values
        
        return surface_df, surface_df['delta_xp'].values, surface_df['delta_yp'].values, surface_df['delta_energy'].values, surface_df['delta_theta'].values

    z_prev = positions[idx - 1]
    prev_df = df[df['scoring_surface'] == z_prev].copy()
    merged = surface_df.merge(prev_df[['particle_id', 'x_prime', 'y_prime', 'delta energy', 'theta']],on='particle_id',suffixes=('', '_prev'))

    merged['delta_xp'] = merged['x_prime'] - merged['x_prime_prev']
    merged['delta_yp'] = merged['y_prime'] - merged['y_prime_prev']
    merged['delta_energy'] = merged['delta energy'] - merged['delta energy_prev']
    merged['delta_theta'] = np.sqrt(merged['delta_xp']**2 + merged['delta_yp']**2)
    return merged, merged['delta_xp'].values, merged['delta_yp'].values, merged['delta_energy'].values, merged['delta_theta'].values

def energy_fit_function_for_surface(z_position, positions_scoring, ripple_filters):
    if not ripple_filters:
        return langau_curve

    first_surface = positions_scoring[0]
    if np.isclose(z_position, first_surface, rtol=0, atol=1e-9):
        return langau_curve
    return triangular_symmetric_curve

def energy_fit_function_for_surface_diff(z_position, positions_scoring, ripple_filters):
    if not ripple_filters:
        return langau_curve

    second_surface = positions_scoring[1]
    if not np.isclose(z_position, second_surface, rtol=0, atol=1e-9):
        return langau_curve
    return triangular_symmetric_curve

def analyze_surface_masked(
    df_masked,
    z_position,
    label,
    ripple_filters,
    positions_scoring,
    particle_type,
):
    """
    Perform all fits and variation analysis using only the masked data.

    Parameters
    ----------
    df_masked:
        Complete masked dataset containing every scoring surface.

    z_position:
        Current scoring surface.

    No percentile cut is applied.
    """

    surface_df = df_masked.loc[
        df_masked["scoring_surface"] == z_position
    ].copy()

    if surface_df.empty:
        raise ValueError(
            f"No masked particles found at "
            f"z = {z_position}."
        )

    # ---------------------------------------------------------------
    # Main spatial and angular fits:
    # all performed on the masked DataFrame.
    # ---------------------------------------------------------------

    (
        _,
        _,
        A_x,
        mu_x,
        sigma_x,
        _,
    ) = gaussian_curve(
        surface_df,
        "x - coordinate",
    )

    (
        _,
        _,
        A_y,
        mu_y,
        sigma_y,
        _,
    ) = gaussian_curve(
        surface_df,
        "y - coordinate",
    )

    (
        _,
        _,
        A_xp,
        mu_xp,
        sigma_xp,
        _,
    ) = gaussian_curve(
        surface_df,
        "x_prime",
    )

    (
        _,
        _,
        A_yp,
        mu_yp,
        sigma_yp,
        _,
    ) = gaussian_curve(
        surface_df,
        "y_prime",
    )

    # ---------------------------------------------------------------
    # Energy fit:
    # performed on the masked DataFrame.
    # ---------------------------------------------------------------

    energy_fit_function = energy_fit_function_for_surface(
        z_position,
        positions_scoring,
        ripple_filters,
    )

    energy_binning = make_auto_binning_from_dataframe(
        df=df_masked,
        column="delta energy",
    )

    if energy_fit_function is langau_curve:
        x_scale_factor = (
            1.0
            if particle_type == "proton"
            else 0.05
        )

        (
            _,
            _,
            _,
            mu_delta_energy,
            _,
            sigma_delta_energy,
        ) = energy_fit_function(
            surface_df,
            "delta energy",
            binning=energy_binning,
            x_scale_factor=x_scale_factor,
        )

    else:
        (
            _,
            _,
            _,
            mu_delta_energy,
            _,
            sigma_delta_energy,
        ) = energy_fit_function(
            surface_df,
            "delta energy",
            binning=energy_binning,
        )
    # ---------------------------------------------------------------
    # Theta fit:
    # performed on the masked DataFrame.
    # ---------------------------------------------------------------

    (
        _,
        _,
        theta_peak,
        A_theta,
        B_theta,
        sigma_theta,
        _,
        _,
    ) = fit_theta(
        surface_df,
        "theta",
    )

    # ---------------------------------------------------------------
    # Twiss statistical parameters:
    # performed on the masked DataFrame.
    # ---------------------------------------------------------------

    (
        emittance_x,
        alpha_x,
        beta_x,
        gamma_x,
    ) = beam_parameters(
        surface_df,
        "x - coordinate",
        "x_prime",
    )

    (
        emittance_y,
        alpha_y,
        beta_y,
        gamma_y,
    ) = beam_parameters(
        surface_df,
        "y - coordinate",
        "y_prime",
    )

    cov_x_stat = np.cov(
        surface_df["x - coordinate"],
        surface_df["x_prime"],
        ddof=0,
    )[0, 1]

    cov_y_stat = np.cov(
        surface_df["y - coordinate"],
        surface_df["y_prime"],
        ddof=0,
    )[0, 1]

    # ---------------------------------------------------------------
    # Variations:
    #
    # IMPORTANT:
    # calculate_variations receives the complete masked DataFrame,
    # not only the current surface.
    #
    # No percentile cut is applied afterwards.
    # ---------------------------------------------------------------

    (
        df_masked_variations,
        _,
        _,
        _,
        _,
    ) = calculate_variations(
        df_masked,
        z_position,
        label,
    )

    df_masked_variations = df_masked_variations.copy()
    surface_df_masked_variations = df_masked_variations[df_masked_variations['scoring_surface'] == z_position].copy()

    if df_masked_variations.empty:
        raise ValueError(
            f"No masked variation data found at "
            f"z = {z_position}."
        )

    (
        _,
        _,
        _,
        mu_var_xp,
        sigma_var_xp,
        _,
    ) = gaussian_curve(
        surface_df_masked_variations,
        "delta_xp",
    )

    (
        _,
        _,
        _,
        mu_var_yp,
        sigma_var_yp,
        _,
    ) = gaussian_curve(
        surface_df_masked_variations,
        "delta_yp",
    )
    

    variation_energy_fit_function = (
        energy_fit_function_for_surface_diff(
            z_position,
            positions_scoring,
            ripple_filters,
        )
    )

    variation_energy_binning = (
        make_auto_binning_from_dataframe(
            df=df_masked_variations,
            column="delta_energy",
        )
    )

    if variation_energy_fit_function is langau_curve:
        x_scale_factor = (
            1.0
            if particle_type == "proton"
            else 0.05
        )

        (
            _,
            _,
            _,
            mpv,
            _,
            sigma_var_delta_energy,
        ) = variation_energy_fit_function(
            surface_df_masked_variations,
            "delta_energy",
            binning=variation_energy_binning,
            x_scale_factor=x_scale_factor,
        )

    else:
        (
            _,
            _,
            _,
            mpv,
            _,
            sigma_var_delta_energy,
        ) = variation_energy_fit_function(
            surface_df_masked_variations,
            "delta_energy",
            binning=variation_energy_binning,
        )

    (
        _,
        _,
        _,
        _,
        _,
        sigma_var_theta,
        _,
        _,
    ) = fit_theta(
        surface_df_masked_variations,
        "delta_theta",
    )

    statistics = {
        "z_position": z_position,
        "counts": len(surface_df),

        "max delta energy": surface_df[
            "delta energy"
        ].max(),

        "theta": surface_df[
            "theta"
        ].to_numpy(
            dtype=float
        ),

        "min theta": surface_df[
            "theta"
        ].min(),

        "std_xp": surface_df[
            "x_prime"
        ].std(),

        "std_yp": surface_df[
            "y_prime"
        ].std(),

        "emittance_x": emittance_x,
        "alpha_x": alpha_x,
        "beta_x": beta_x,
        "gamma_x": gamma_x,

        "emittance_y": emittance_y,
        "alpha_y": alpha_y,
        "beta_y": beta_y,
        "gamma_y": gamma_y,

        "cov_x": cov_x_stat,
        "cov_y": cov_y_stat,

        "A_x": A_x,
        "mu_x": mu_x,
        "sigma_x": sigma_x,

        "A_y": A_y,
        "mu_y": mu_y,
        "sigma_y": sigma_y,

        "A_xp": A_xp,
        "mu_xp": mu_xp,
        "sigma_xp": sigma_xp,

        "A_yp": A_yp,
        "mu_yp": mu_yp,
        "sigma_yp": sigma_yp,

        "mu_delta_energy": mu_delta_energy,
        "sigma_delta_energy": sigma_delta_energy,

        "theta_peak": theta_peak,
        "Atheta": A_theta,
        "Btheta": B_theta,
        "sigma_theta": sigma_theta,

        "mu_var_xp": mu_var_xp,
        "sigma_var_xp": sigma_var_xp,

        "mu_var_yp": mu_var_yp,
        "sigma_var_yp": sigma_var_yp,

        "mpv": mpv,
        "sigma_var_delta_energy": (
            sigma_var_delta_energy
        ),

        "sigma_var_theta": sigma_var_theta,
    }

    return (
        statistics,
        surface_df,
        surface_df_masked_variations,
    )

def analyze_surface_orig(df_original, df_filtered, z_position, label, ripple_filters, positions_scoring, particle_type):

    surface_df = df_original[df_original['scoring_surface'] == z_position].copy()

    _, _, A_x, mu_x, sigma_x, _ = gaussian_curve(surface_df, 'x - coordinate')
    _, _, A_y,mu_y, sigma_y, _= gaussian_curve(surface_df, 'y - coordinate')
    _, _, A_xp, mu_xp, sigma_xp, _  = gaussian_curve(surface_df, 'x_prime')
    _, _, A_yp, mu_yp, sigma_yp, _= gaussian_curve(surface_df, 'y_prime')
    fit_func = energy_fit_function_for_surface(z_position, positions_scoring, ripple_filters)

    if fit_func is langau_curve:
        x_scale_factor = 1.0 if particle_type == "proton" else 0.1
        _, _, _, mu_energy, _, sigma_delta_energy = fit_func(
            surface_df,
            'delta energy',
            binning=make_auto_binning_from_dataframe(surface_df, column='delta energy'),
            x_scale_factor=x_scale_factor
        )
    else:
        _, _, _, mu_energy, _, sigma_delta_energy = fit_func(
            surface_df,
            'delta energy',
            binning=make_auto_binning_from_dataframe(surface_df, column='delta energy')
        )
    _, _,_, _, _, sigma_theta, _,_ = fit_theta (surface_df, 'theta')
    

    df_filtered_variations, _, _, _, _ = calculate_variations(df_filtered, z_position, label)
    surface_df_filtered_variations = df_filtered_variations[df_filtered_variations['scoring_surface'] == z_position].copy()
    
    value_ranges_orig = {
        'delta_xp': (1, 99),
        'delta_yp': (1, 99),
        'delta_energy': (0, 99),
        'delta_theta': (5, 100)}
    selected_cols_orig = ['delta_xp', 'delta_yp', 'delta_energy', 'delta_theta']
    surface_df_var = percentile_cut(value_ranges_orig, surface_df_filtered_variations, selected_cols_orig)

    _,_, _, mu_var_xp, sigma_var_xp,_= gaussian_curve(surface_df_var,'delta_xp')
    _,_, _, mu_var_yp, sigma_var_yp,_= gaussian_curve(surface_df_var, 'delta_yp')
    fit_func = energy_fit_function_for_surface_diff(z_position, positions_scoring, ripple_filters)

    if fit_func is langau_curve:
        x_scale_factor = 1.0 if particle_type == "proton" else 0.1
        _, _, _, mpv, _, sigma_var_delta_energy = fit_func(
            surface_df_var,
            'delta_energy',
            binning=make_auto_binning_from_dataframe(surface_df_var, column='delta_energy'),
            x_scale_factor=x_scale_factor
        )
    else:
        _, _, _, mpv, _, sigma_var_delta_energy = fit_func(
            surface_df_var,
            'delta_energy',
            binning=make_auto_binning_from_dataframe(surface_df_var, column='delta_energy')
        )
    _, _,_, _, _, sigma_var_theta, _,_ = fit_theta (surface_df_var,'delta_theta')


    statistics = {
        'z_position': z_position,
        'counts': len(surface_df),
        'max delta energy': surface_df ['delta energy'].max(),
        'theta': surface_df['theta'].values,
        'min theta': surface_df['theta'].min(),
        'sigma_theta': sigma_theta,
        #'std_xp':surface_df['x_prime'].std(),
        #'std_yp':surface_df['y_prime'].std(),

        'A_x': A_x,
        'mu_x': mu_x,
        'sigma_x': sigma_x,
        'A_y': A_y,
        'mu_y': mu_y,
        'sigma_y': sigma_y,
        'A_xp': A_xp,
        'mu_xp': mu_xp,
        'sigma_xp': sigma_xp,
        'A_yp': A_yp,
        'mu_yp': mu_yp,
        'sigma_yp': sigma_yp,

        'sigma_delta_energy': sigma_delta_energy,
        
        'sigma_var_xp': sigma_var_xp,
        'sigma_var_yp': sigma_var_yp,
        'sigma_var_delta_energy': sigma_var_delta_energy,
        'mu_var_xp': mu_var_xp,
        'mu_var_yp': mu_var_yp,
        'mpv': mpv,
        'sigma_var_theta': sigma_var_theta
    }

    return statistics, surface_df, surface_df_var


def analyze_surface_filt(df, surface_data_original_cut,  z_position, label, ripple_filters, positions_scoring, particle_type):
    
    surface_df = df[df['scoring_surface'] == z_position].copy()


    print (surface_df.columns)

    
    _, _, A_x, mu_x, sigma_x, _ = gaussian_curve(surface_df, 'x - coordinate')
    _, _, A_y,mu_y, sigma_y, _= gaussian_curve(surface_df, 'y - coordinate')
    _, _, A_xp, mu_xp, sigma_xp, _  = gaussian_curve(surface_df, 'x_prime')
    _, _, A_yp, mu_yp, sigma_yp, _= gaussian_curve(surface_df, 'y_prime')
    fit_func = energy_fit_function_for_surface(z_position, positions_scoring, ripple_filters)
    df_variations, _, _, _, _ = calculate_variations(surface_df, z_position, label)
    surface_df_variations = df_variations[df_variations['scoring_surface'] == z_position].copy()

    if fit_func is langau_curve:
        x_scale_factor = 1.0 if particle_type == "proton" else 0.1
        _, _, _, mu_delta_energy, _, sigma_delta_energy = fit_func(
            surface_df,
            'delta energy',
            binning=make_auto_binning_from_dataframe(surface_df_variations, column='delta energy'),
            x_scale_factor=x_scale_factor
        )
    else:
        _, _, _, mu_delta_energy, _, sigma_delta_energy = fit_func(
            surface_df,
            'delta energy',
            binning=make_auto_binning_from_dataframe(surface_df_variations, column='delta energy')
        )
    _, _, theta_peak, A_theta, B_theta, sigma_theta, _,_ = fit_theta (surface_df, 'theta')

    emittance_x, alpha_x, beta_x, gamma_x = beam_parameters(surface_df, 'x - coordinate', 'x_prime')
    emittance_y, alpha_y, beta_y, gamma_y = beam_parameters(surface_df, 'y - coordinate', 'y_prime')

    cov_x_stat = np.cov(surface_df['x - coordinate'], surface_df['x_prime'])[0, 1]
    cov_y_stat = np.cov(surface_df['y - coordinate'], surface_df['y_prime'])[0, 1]


    #_, _, mut_x, sigmat_x, dft_x = tstudent_fit_on_hist_with_baseline(surface_df, 'x - coordinate')
    #_, _, mut_y, sigmat_y, dft_y = tstudent_fit_on_hist_with_baseline(surface_df, 'y - coordinate')
    #_, _, mut_xp, sigmat_xp, dft_xp  = tstudent_fit_on_hist_with_baseline(surface_df, 'x_prime')
    #_, _, mut_yp, sigmat_yp,dft_yp= tstudent_fit_on_hist_with_baseline(surface_df, 'y_prime')

    
    statistics = {
        'z_position': z_position,
        'counts': len(surface_df),
        'max delta energy': surface_df ['delta energy'].max(),
        'theta': surface_df['theta'].values,
        'min theta': surface_df['theta'].min(),
        'std_xp':surface_df['x_prime'].std(),
        'std_yp':surface_df['y_prime'].std(),

        'emittance_x': emittance_x,
        'alpha_x': alpha_x,
        'beta_x': beta_x,
        'gamma_x': gamma_x,
        'emittance_y': emittance_y,
        'alpha_y': alpha_y,
        'beta_y': beta_y,
        'gamma_y': gamma_y,

        'mu_delta_energy': mu_delta_energy,
        'sigma_delta_energy': sigma_delta_energy,
        'sigma_theta': sigma_theta,
        'theta_peak': theta_peak,
        'Atheta': A_theta,
        'Btheta': B_theta,

        'cov_x': cov_x_stat,
        'cov_y': cov_y_stat,

        'A_x': A_x,
        'mu_x': mu_x,
        'sigma_x': sigma_x,
        'A_y': A_y,
        'mu_y': mu_y,
        'sigma_y': sigma_y,
        'A_xp': A_xp,
        'mu_xp': mu_xp,
        'sigma_xp': sigma_xp,
        'A_yp': A_yp,
        'mu_yp': mu_yp,
        'sigma_yp': sigma_yp,

        #'mut_x': mut_x,
        #'mut_xp': mut_xp,
        #'mut_y': mut_y,
        #'mut_yp': mut_yp, 

        #'dft_x': dft_x,
        #'dft_xp': dft_xp,
        #'dft_y': dft_y,
        #'dft_yp': dft_yp,

        #'sigmat_x': sigmat_x,
        #'sigmat_xp': sigmat_xp,
        #'sigmat_y': sigmat_y,
        #'sigmat_yp': sigmat_yp,

    }
    return statistics, surface_df

def analyze_dataset(
    df_original,
    df_filtered,
    label,
    positions_scoring,
    ripple_filters=False,
    particle_type="proton",
    cleaning_cut_mode="ellipse",
    theta_percentile_min=0.0,
    theta_percentile_max=99.0,
    ellipse_peak_fraction=0.1,
    angle_percentile_min=0.0,
    angle_percentile_max=99.0
):
    """
    Analyze one dataset across all scoring surfaces.

    Pipeline
    --------
    1. Original data:
       statistics are calculated from the unfiltered input dataset.

    2. Filtered data:
       statistics are calculated after build_sequential_particles().

    3. Final cleaned/masked data:
       statistics are calculated after one selectable cleaning cut:

       - cleaning_cut_mode="ellipse":
         a row survives only if it lies inside both phase-space
         ellipses x-x' and y-y';

       - cleaning_cut_mode="theta":
         a row survives only if theta lies within the selected
         percentile interval.

    Notes
    -----
    The return variable names still use "masked" for compatibility
    with the rest of the project. In theta mode, "masked" means
    "theta-cleaned".

    No percentile cut is applied to the final cleaned dataset other
    than the theta selection when cleaning_cut_mode="theta".

    Returns
    -------
    tuple
        (
            results_original,
            results_filtered,
            results_masked,
            df_filtered_complete,
            df_masked_complete,
            surface_data_original,
            surface_data_filtered,
            surface_data_masked,
            stored_surface_plots,
            stored_distribution_plots,
            surface_data_masked_variations,
        )
    """

    if df_original is None:
        raise ValueError(
            "df_original cannot be None."
        )

    if df_filtered is None:
        raise ValueError(
            "df_filtered cannot be None."
        )

    if df_original.empty:
        raise ValueError(
            f"{label}: df_original is empty."
        )

    if df_filtered.empty:
        raise ValueError(
            f"{label}: df_filtered is empty."
        )

    if "scoring_surface" not in df_original.columns:
        raise KeyError(
            "df_original requires column "
            "'scoring_surface'."
        )

    if "scoring_surface" not in df_filtered.columns:
        raise KeyError(
            "df_filtered requires column "
            "'scoring_surface'."
        )

    positions_scoring = [
        float(position)
        for position in positions_scoring
    ]

    if not positions_scoring:
        raise ValueError(
            "positions_scoring cannot be empty."
        )

    cleaning_cut_mode = str(
        cleaning_cut_mode
    ).strip().lower()

    if cleaning_cut_mode not in (
        "ellipse",
        "theta",
    ):
        raise ValueError(
            "cleaning_cut_mode must be "
            "'ellipse' or 'theta'."
        )

    theta_percentile_min = float(
        theta_percentile_min
    )

    theta_percentile_max = float(
        theta_percentile_max
    )

    ellipse_peak_fraction = float(
        ellipse_peak_fraction
    )

    angle_percentile_max = float(angle_percentile_max)
    angle_percentile_min = float(angle_percentile_min)

    if cleaning_cut_mode == "theta":
        if (
            theta_percentile_min < 0.0
            or theta_percentile_max > 100.0
            or theta_percentile_min >= theta_percentile_max
        ):
            raise ValueError(
                "Theta percentile limits must satisfy "
                "0 <= min < max <= 100."
            )

    if cleaning_cut_mode == "ellipse":
        if (
            ellipse_peak_fraction <= 0.0
            or ellipse_peak_fraction >= 1.0
        ):
            raise ValueError(
                "ellipse_peak_fraction must satisfy "
                "0 < ellipse_peak_fraction < 1."
            )

    results_original = []
    results_filtered = []
    results_masked = []

    all_surface_original = []
    all_surface_filtered = []
    all_surface_masked = []

    all_variations_masked = []

    stored_surface_plots = []
    stored_distribution_plots = []

    print(
        f"\n========== ANALYZE DATASET: {label} =========="
    )

    print(
        f"Final cleaning mode: {cleaning_cut_mode}"
    )

    if cleaning_cut_mode == "theta":
        print(
            "Theta percentile range: "
            f"{theta_percentile_min:.2f}% "
            f"to {theta_percentile_max:.2f}%"
        )

    elif cleaning_cut_mode == "ellipse":
        print(
            "Ellipse peak fraction: "
            f"{ellipse_peak_fraction:.6g}"
        )

    print(
        "\nOriginal scoring-surface counts:"
    )

    print(
        df_original[
            "scoring_surface"
        ].value_counts().sort_index()
    )

    # ============================================================
    # FIRST PASS:
    # build the final cleaned dataset independently at every
    # scoring surface.
    # ============================================================

    for position in positions_scoring:
        original_surface_mask = np.isclose(
            pd.to_numeric(
                df_original[
                    "scoring_surface"
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            ),
            position,
            rtol=0.0,
            atol=1e-9,
        )

        filtered_surface_mask = np.isclose(
            pd.to_numeric(
                df_filtered[
                    "scoring_surface"
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            ),
            position,
            rtol=0.0,
            atol=1e-9,
        )

        df_surf_original = df_original.loc[
            original_surface_mask
        ].copy()

        df_surf_filtered = df_filtered.loc[
            filtered_surface_mask
        ].copy()

        if df_surf_original.empty:
            print(
                f"WARNING: {label} | z={position:.3f} cm | "
                "no original rows."
            )

        if df_surf_filtered.empty:
            print(
                f"WARNING: {label} | z={position:.3f} cm | "
                "no filtered rows."
            )

            continue

        df_surf_masked = apply_dataset_cleaning_cut(
            df_surf_filtered,

            cut_mode=cleaning_cut_mode,

            theta_percentile_min=(
                theta_percentile_min
            ),

            theta_percentile_max=(
                theta_percentile_max
            ),

            ellipse_peak_fraction=(
                ellipse_peak_fraction
            ),
            angle_percentile_max=(
                angle_percentile_max
            ),
            angle_percentile_min=(
                angle_percentile_min
            )
        )

        print(
            f"{label} | z={position:.3f} cm | "
            f"mode={cleaning_cut_mode} | "
            f"N original={len(df_surf_original)} | "
            f"N filtered={len(df_surf_filtered)} | "
            f"N cleaned={len(df_surf_masked)}"
        )

        all_surface_original.append(
            df_surf_original
        )

        all_surface_filtered.append(
            df_surf_filtered
        )

        all_surface_masked.append(
            df_surf_masked
        )

    if all_surface_filtered:
        df_filtered_complete = pd.concat(
            all_surface_filtered,
            ignore_index=True,
        )

    else:
        df_filtered_complete = pd.DataFrame(
            columns=df_filtered.columns
        )

    if all_surface_masked:
        df_masked_complete = pd.concat(
            all_surface_masked,
            ignore_index=True,
        )

    else:
        df_masked_complete = pd.DataFrame(
            columns=df_filtered.columns
        )

    all_surface_filtered.clear()
    all_surface_masked.clear()

    del df_surf_original
    del df_surf_filtered
    del df_surf_masked


    # ============================================================
    # SECOND PASS:
    # calculate original, filtered and final-cleaned statistics.
    # ============================================================

    for position in positions_scoring:
        original_surface_mask = np.isclose(
            pd.to_numeric(
                df_original[
                    "scoring_surface"
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            ),
            position,
            rtol=0.0,
            atol=1e-9,
        )

        filtered_surface_mask = np.isclose(
            pd.to_numeric(
                df_filtered_complete[
                    "scoring_surface"
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            ),
            position,
            rtol=0.0,
            atol=1e-9,
        )

        masked_surface_mask = np.isclose(
            pd.to_numeric(
                df_masked_complete[
                    "scoring_surface"
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            ),
            position,
            rtol=0.0,
            atol=1e-9,
        )

        df_surf_original = df_original.loc[
            original_surface_mask
        ].copy()

        df_surf_filtered = df_filtered_complete.loc[
            filtered_surface_mask
        ].copy()

        df_surf_masked = df_masked_complete.loc[
            masked_surface_mask
        ].copy()

        if df_surf_original.empty:
            print(
                f"WARNING: {label} | z={position:.3f} cm | "
                "original surface unavailable; skipped."
            )

            continue

        if df_surf_filtered.empty:
            print(
                f"WARNING: {label} | z={position:.3f} cm | "
                "filtered surface unavailable; skipped."
            )

            continue

        if df_surf_masked.empty:
            print(
                f"WARNING: {label} | z={position:.3f} cm | "
                f"no rows survived the "
                f"'{cleaning_cut_mode}' cleaning cut; skipped."
            )

            continue

        # --------------------------------------------------------
        # Original-data statistics.
        # --------------------------------------------------------

        (
            statistics_original,
            surface_data_original_current,
            surface_data_original_variations,
        ) = analyze_surface_orig(
            df_original,
            df_filtered_complete,
            position,
            label,
            ripple_filters=ripple_filters,
            positions_scoring=positions_scoring,
            particle_type=particle_type,
        )

        # --------------------------------------------------------
        # Sequentially filtered-data statistics.
        # --------------------------------------------------------

        (
            statistics_filtered,
            surface_data_filtered_current,
        ) = analyze_surface_filt(
            df_surf_filtered,
            surface_data_original_variations,
            position,
            label,
            ripple_filters=ripple_filters,
            positions_scoring=positions_scoring,
            particle_type=particle_type,
        )

        # --------------------------------------------------------
        # Statistics after selected cleaning mode:
        # ellipse or theta percentile cut.
        # --------------------------------------------------------

        (
            statistics_masked,
            surface_data_masked_current,
            surface_data_masked_variations_current,
        ) = analyze_surface_masked(
            df_masked=df_masked_complete,
            z_position=position,
            label=label,
            ripple_filters=ripple_filters,
            positions_scoring=positions_scoring,
            particle_type=particle_type,
        )

        # Store the selected cleaning mode for traceability.
        statistics_original[
            "cleaning_cut_mode"
        ] = "original"

        statistics_filtered[
            "cleaning_cut_mode"
        ] = "filtered"

        statistics_masked[
            "cleaning_cut_mode"
        ] = cleaning_cut_mode

        if cleaning_cut_mode == "theta":
            statistics_masked[
                "theta_percentile_min"
            ] = theta_percentile_min

            statistics_masked[
                "theta_percentile_max"
            ] = theta_percentile_max

        elif cleaning_cut_mode == "ellipse":
            statistics_masked[
                "ellipse_peak_fraction"
            ] = ellipse_peak_fraction

        results_original.append(
            statistics_original.copy()
        )

        results_filtered.append(
            statistics_filtered.copy()
        )

        results_masked.append(
            statistics_masked.copy()
        )

        all_variations_masked.append(
            surface_data_masked_variations_current.copy()
        )

        stored_surface_plots.append(
            {
                "surface_df_original": (
                    df_surf_original.copy()
                ),

                "surface_df_filtered": (
                    df_surf_filtered.copy()
                ),

                # This is either ellipse-cleaned or theta-cleaned,
                # depending on cleaning_cut_mode.
                "surface_df_masked": (
                    surface_data_masked_current.copy()
                ),

                "surface_data_masked": (
                    surface_data_masked_variations_current.copy()
                ),

                # Legacy compatibility key used by histogram sliders.
                "surface_data_masked_cut": (
                    surface_data_masked_variations_current.copy()
                ),

                "z_position": float(
                    position
                ),

                "theta_peak": statistics_masked.get(
                    "theta_peak"
                ),

                "A": statistics_masked.get(
                    "Atheta"
                ),

                "B": statistics_masked.get(
                    "Btheta"
                ),

                "sigma": statistics_masked.get(
                    "sigma_theta"
                ),

                "label": label,

                "cleaning_cut_mode": (
                    cleaning_cut_mode
                ),
            }
        )

        stored_distribution_plots.append(
            {
                "surface_df_original": (
                    df_surf_original.copy()
                ),

                "surface_df_filtered": (
                    df_surf_filtered.copy()
                ),

                # This is the final selected dataset.
                "surface_df_masked": (
                    surface_data_masked_current.copy()
                ),

                "z_position": float(
                    position
                ),

                "label": label,

                "cleaning_cut_mode": (
                    cleaning_cut_mode
                ),
            }
        )

    if all_surface_original:
        surface_data_original = pd.concat(
            all_surface_original,
            ignore_index=True,
        )

    else:
        surface_data_original = pd.DataFrame(
            columns=df_original.columns
        )

    surface_data_filtered = df_filtered_complete

    if results_masked:
        surface_data_masked = pd.concat(
            [
                item["surface_df_masked"]
                for item in stored_surface_plots
            ],
            ignore_index=True,
        )

    else:
        surface_data_masked = pd.DataFrame(
            columns=df_filtered.columns
        )

    if all_variations_masked:
        surface_data_masked_variations = pd.concat(
            all_variations_masked,
            ignore_index=True,
        )

    else:
        surface_data_masked_variations = pd.DataFrame()

    print(
        f"\n========== END ANALYSIS: {label} =========="
    )

    print(
        f"Cleaning mode: {cleaning_cut_mode}"
    )

    print(
        f"Filtered rows: {len(df_filtered_complete)}"
    )

    print(
        f"Final cleaned rows: {len(df_masked_complete)}"
    )

    return (
        results_original,
        results_filtered,
        results_masked,
        df_filtered_complete,
        df_masked_complete,
        surface_data_original,
        surface_data_filtered,
        surface_data_masked,
        stored_surface_plots,
        stored_distribution_plots,
        surface_data_masked_variations,
    )

def setup_dataset(file_name, label, E_0, positions_scoring, tolerance, column_names, nRows=None):
    print(f"\nProcessing dataset: {label}")

    df = load_data(file_name, column_names=column_names, nRows=nRows)
    if df is None:
        print("File not found or error in loading.")
        return None, None

    df = conversion(df, E_0 = E_0)
    df = assign_scoring_surface(df, positions_scoring, tolerance)

    if 'scoring position' in df.columns:
        df.drop(columns=['scoring position'], inplace=True)
    print(f"Base dataframe for {label}: {len(df)} rows")

    df_original = df.copy()

    return df_original

def generate_comparison_dataframe(results_original_vacuum, results_original_air, positions_scoring):
    data = {
        "z [cm]": positions_scoring,
        "STD(xp) Vacuum": [r["sigma_xp"] for r in results_original_vacuum],
        "STD(yp) Vacuum": [r["sigma_yp"] for r in results_original_vacuum],
        "STD(xp) Air":    [r["sigma_xp"] for r in results_original_air],
        "STD(yp) Air":    [r["sigma_yp"] for r in results_original_air],
        "STD(x) Vacuum": [r["sigma_x"] for r in results_original_vacuum],
        "STD(y) Vacuum": [r["sigma_y"] for r in results_original_vacuum],
        "STD(x) Air": [r["sigma_x"] for r in results_original_air],
        "STD(y) Air": [r["sigma_y"] for r in results_original_air],
        "STD(delta energy) Vacuum": [r["sigma_delta_energy"] for r in results_original_vacuum],
        "STD(delta energy) Air": [r["sigma_delta_energy"] for r in results_original_air],
        "STD (theta) Vacuum": [r["sigma_theta"] for r in results_original_vacuum],
        "STD (theta) Air": [r["sigma_theta"] for r in results_original_air],

        "STD(delta xp) Vacuum": [r["sigma_var_xp"] for r in results_original_vacuum],
        "STD(delta yp) Vacuum": [r["sigma_var_yp"] for r in results_original_vacuum],
        "STD(delta xp) Air": [r["sigma_var_xp"] for r in results_original_air],
        "STD(delta yp) Air": [r["sigma_var_yp"] for r in results_original_air],
        "MPV (mode energy) Vacuum": [r["mpv"] for r in results_original_vacuum],
        "MPV (mode energy) Air": [r["mpv"] for r in results_original_air],
        "STD(variation delta energy) Vacuum": [r["sigma_var_delta_energy"] for r in results_original_vacuum],
        "STD(variation delta energy) Air": [r["sigma_var_delta_energy"] for r in results_original_air],
        "STD (variation theta) Vacuum": [r["sigma_var_theta"] for r in results_original_vacuum],
        "STD (variation theta) Air": [r["sigma_var_theta"] for r in results_original_air],

    }
    dataframe_std = pd.DataFrame(data)


    return dataframe_std


def normalize_pdf(pdf, xgrid):
    """
    Normalizza una PDF discreta affinché:
        integral pdf(x) dx = 1
    """
    pdf = np.asarray(pdf, dtype=float).copy()
    xgrid = np.asarray(xgrid, dtype=float)

    pdf[~np.isfinite(pdf)] = 0.0
    pdf = np.clip(pdf, 0.0, None)

    area = np.trapezoid(pdf, xgrid)

    if not np.isfinite(area) or area <= 0.0:
        pdf[:] = 0.0
        pdf[np.argmin(np.abs(xgrid))] = 1.0
        area = np.trapezoid(pdf, xgrid)

    return pdf / area

def sigma_from_pdf_norm(x, pdf):
    x = np.asarray(x, float)
    pdf = normalize_pdf(pdf, x)
    mu = np.trapezoid(x * pdf, x)
    var = np.trapezoid((x - mu) ** 2 * pdf, x)
    return float(np.sqrt(max(var, 0.0)))

from scipy.interpolate import interp1d
from scipy.stats import t


def compute_all_stats(dataframe_std, ripple_filters): 

    stats_angle = {}
    print("\033[1mAnalysis in angle.\033[0m")
    for attr in ['xp', 'yp']:

        vacuum_std = dataframe_std[f"STD({attr}) Vacuum"].values
        air_std = dataframe_std[f"STD({attr}) Air"].values
        vacuum_theta_std = dataframe_std[f"STD (theta) Vacuum"].values
        air_theta_std = dataframe_std[f"STD (theta) Air"].values

        vacuum_delta = dataframe_std[f"STD(delta {attr}) Vacuum"].values
        air_delta = dataframe_std[f"STD(delta {attr}) Air"].values
        vacuum_delta_theta = dataframe_std[f"STD (variation theta) Vacuum"].values

        if ripple_filters is False:

            window_angle_std = vacuum_std[0]
            window_theta_std = vacuum_theta_std[0]
            nozzle_angle_std = np.sqrt(vacuum_std[1]**2 - vacuum_std[0]**2)
            nozzle_theta_std = np.sqrt(vacuum_theta_std[1]**2 - vacuum_theta_std[0]**2)
            

            air_133 = np.sqrt(air_std[0]**2 - window_angle_std**2)
            air_160 = np.sqrt(air_std[1]**2 - air_std[0]**2 -nozzle_angle_std**2)
            air_293 = np.sqrt (air_std[1]**2 - nozzle_angle_std**2 - window_angle_std**2)
            air_600  = np.sqrt(air_std[2]**2 - air_std[1]**2)
            air_760 = np.sqrt(air_std[2]**2-air_std[0]**2-nozzle_angle_std**2)
            air_893 = np.sqrt(air_std[2]**2 - nozzle_angle_std**2 - window_angle_std**2)
            air_1000 = np.sqrt(air_std[3]**2 - air_std[2]**2)
            air_1600 = np.sqrt(air_std[3]**2 - air_std[1]**2)
            air_1760= np.sqrt(air_std[3]**2 - air_std[0]**2 - nozzle_angle_std**2)
            air_1893 = np.sqrt(air_std[3]**2 - nozzle_angle_std**2 - window_angle_std**2)


            check_133 = air_133 /np.sqrt(133)
            check_160 = air_160 / np.sqrt(160)
            check_293 = air_293/np.sqrt(293)
            check_600  = air_600  / np.sqrt(600)
            check_760 = air_760 / np.sqrt(760)
            check_893 = air_893 /np.sqrt(893)
            check_1000 = air_1000 / np.sqrt(1000)
            check_1600 = air_1600 / np.sqrt(1600)
            check_1760 = air_1760 / np.sqrt(1760)
            check_1893 = air_1893 / np.sqrt(1893)

            print()
            print(f"\033[1m--From the standard deviations of the fit--\033[0m")
            print()
            print(f"Standard deviations in {attr[0]}':")
            print(f"Vacuum window [mrad]: {window_angle_std:.15e}")
            print(f"Nozzle [mrad]: {nozzle_angle_std:.15e}")
            print(f"Verify Air [mrad/sqrt(mm)]: check on 13,3 cm of air: {check_133:.15e}, check on 16 cm of air: {check_160:.15e}, check on 29,3 cm of air: {check_293:.15e}")
            print(f"Verify Air [mrad/sqrt(mm)]: check on 60 cm of air: {check_600:.15e}, check on 76 cm of air: {check_760:.15e}, check on 89,3 cm of air: {check_893:.15e}")
            print(f"Verify Air [mrad/sqrt(mm)]: check on 100 cm of air: {check_1000:.15e}, check on 160 cm of air: {check_1600:.15e}, check on 1760 cm of air: {check_1760:.15e}")
            
            window_angle_delta = vacuum_delta[0]
            window_angle_delta_theta = vacuum_delta_theta[0]
            nozzle_angle_delta = vacuum_delta[1]
            nozzle_angle_delta_theta = vacuum_delta_theta[1]
            
            air_133_delta = np.sqrt(air_delta[0]**2 - window_angle_delta**2)
            air_160_delta = np.sqrt(air_delta[1]**2 -nozzle_angle_delta**2)
            air_293_delta = np.sqrt (air_delta[1]** 2 + air_delta[0]**2 - nozzle_angle_delta**2 - window_angle_delta**2)
            air_600_delta  = air_delta[2]
            air_760_delta = np.sqrt(air_delta[2]**2 + air_delta[1]**2-nozzle_angle_delta**2)
            air_893_delta = np.sqrt(air_delta[2]**2  + air_delta[1]** 2 + air_delta[0]**2 - nozzle_angle_delta**2 - window_angle_delta**2)
            air_1000_delta = air_delta[3]
            air_1600_delta = np.sqrt(air_delta[3]**2 + air_delta[2]**2)
            air_1760_delta= np.sqrt(air_delta[3]**2 + air_delta[2]**2  + air_delta[1]** 2 - nozzle_angle_delta**2)
            air_1893_delta = np.sqrt(air_delta[3]**2 + air_delta[2]**2  + air_delta[1]** 2 + air_delta[0]**2 - nozzle_angle_delta**2 - window_angle_delta**2)

            check_133_delta = air_133_delta /np.sqrt(133)
            check_160_delta = air_160_delta / np.sqrt(160)
            check_293_delta = air_293_delta/np.sqrt(293)
            check_600_delta  = air_600_delta  / np.sqrt(600)
            check_760_delta = air_760_delta / np.sqrt(760)
            check_893_delta = air_893_delta/np.sqrt(893)
            check_1000_delta = air_1000_delta / np.sqrt(1000)
            check_1600_delta = air_1600_delta / np.sqrt(1600)
            check_1760_delta = air_1760_delta / np.sqrt(1760)
            check_1893_delta = air_1893_delta / np.sqrt(1893)




            print()
            print(f"\033[1m--From the deltas--\033[0m")
            print()
            print(f"Standard deviations in {attr[0]}':")
            print(f"Vacuum window [mrad]: {window_angle_delta:.15e}")
            print(f"Nozzle [mrad]: {nozzle_angle_delta:.15e}")
            print(f"Verify Air [mrad/sqrt(mm)]: check on 13,3 cm of air: {check_133_delta:.15e}, check on 16 cm of air: {check_160_delta:.15e}, check on 29,3 cm of air: {check_293_delta:.15e}")
            print(f"Verify Air [mrad/sqrt(mm)]: check on 60 cm of air: {check_600_delta:.15e}, check on 76 cm of air: {check_760_delta:.15e}, check on 89,3 cm of air: {check_893_delta:.15e}")
            print(f"Verify Air [mrad/sqrt(mm)]: check on 100 cm of air: {check_1000_delta:.15e}, check on 160 cm of air: {check_1600_delta:.15e}, check on 1760 cm of air: {check_1760_delta:.15e}")
            
            stats_angle[attr] = {
                "window_angle_std": window_angle_std,
                "window_theta_std": window_theta_std,
                "window_angle_variation_std": window_angle_delta,
                "window_theta_variation_std": window_angle_delta_theta,
                "nozzle_angle_std": nozzle_angle_std,
                "nozzle_theta_std": nozzle_theta_std,
                "nozzle_angle_variation_std": nozzle_angle_delta,
                "nozzle_theta_variation_std": nozzle_angle_delta_theta,
                "air_133": air_133,
                "air_160": air_160,
                "air_293": air_293,
                "air_600": air_600,
                "air_760": air_760,
                "air_893": air_893,
                "air_1000": air_1000,
                "air_1600": air_1600,
                "air_1760": air_1760,
                "air_1893": air_1893,
                "check_133": check_133,
                "check_160": check_160,
                "check_293": check_293,
                "check_600": check_600,
                "check_760": check_760,
                "check_893": check_893,
                "check_1000": check_1000,
                "check_1600": check_1600,
                "check_1760": check_1760,
                "check_1893": check_1893,
                "air_133_delta": air_133_delta,
                "air_160_delta": air_160_delta,
                "air_293_delta": air_293_delta,
                "air_600_delta": air_600_delta,
                "air_760_delta": air_760_delta,
                "air_893_delta": air_893_delta,
                "air_1000_delta": air_1000_delta,
                "air_1600_delta": air_1600_delta,
                "air_1760_delta": air_1760_delta,
                "air_1893_delta": air_1893_delta,
                "check_133_delta": check_133_delta,
                "check_160_delta": check_160_delta,
                "check_293_delta": check_293_delta,
                "check_600_delta": check_600_delta,
                "check_760_delta": check_760_delta,
                "check_893_delta": check_893_delta,
                "check_1000_delta": check_1000_delta,
                "check_1600_delta": check_1600_delta,
                "check_1760_delta": check_1760_delta,
                "check_1893_delta": check_1893_delta,
            }

        elif ripple_filters is True:

            window_angle_std = vacuum_std[0]
            window_theta_std = vacuum_theta_std[0]
            nozzle_ripple_angle_std = np.sqrt(vacuum_std[1]**2 - vacuum_std[0]**2)
            nozzle_ripple_theta_std = np.sqrt(vacuum_theta_std[1]**2 - vacuum_theta_std[0]**2)
            

            air_133 = np.sqrt(air_std[0]**2 - window_angle_std**2)
            air_160 = np.sqrt(air_std[1]**2 - air_std[0]**2 -nozzle_ripple_angle_std**2)
            air_293 = np.sqrt (air_std[1]**2 - nozzle_ripple_angle_std**2 - window_angle_std**2)
            air_600  = np.sqrt(air_std[2]**2 - air_std[1]**2)
            air_760 = np.sqrt(air_std[2]**2-air_std[0]**2-nozzle_ripple_angle_std**2)
            air_893 = np.sqrt(air_std[2]**2 - nozzle_ripple_angle_std**2 - window_angle_std**2)
            air_1000 = np.sqrt(air_std[3]**2 - air_std[2]**2)
            air_1600 = np.sqrt(air_std[3]**2 - air_std[1]**2)
            air_1760= np.sqrt(air_std[3]**2 - air_std[0]**2 - nozzle_ripple_angle_std**2)
            air_1893 = np.sqrt(air_std[3]**2 - nozzle_ripple_angle_std**2 - window_angle_std**2)


            check_133 = air_133 /np.sqrt(133)
            check_160 = air_160 / np.sqrt(160)
            check_293 = air_293/np.sqrt(293)
            check_600  = air_600  / np.sqrt(600)
            check_760 = air_760 / np.sqrt(760)
            check_893 = air_893 /np.sqrt(893)
            check_1000 = air_1000 / np.sqrt(1000)
            check_1600 = air_1600 / np.sqrt(1600)
            check_1760 = air_1760 / np.sqrt(1760)
            check_1893 = air_1893 / np.sqrt(1893)

            print()
            print(f"\033[1m--From the standard deviations of the fit--\033[0m")
            print()
            print(f"Standard deviations in {attr[0]}':")
            print(f"Vacuum window [mrad]: {window_angle_std:.15e}")
            print(f"Nozzle and Ripple Filters [mrad]: {nozzle_ripple_angle_std:.15e}")
            print(f"Verify Air [mrad/sqrt(mm)]: check on 13,3 cm of air: {check_133:.15e}, check on 16 cm of air: {check_160:.15e}, check on 29,3 cm of air: {check_293:.15e}")
            print(f"Verify Air [mrad/sqrt(mm)]: check on 60 cm of air: {check_600:.15e}, check on 76 cm of air: {check_760:.15e}, check on 89,3 cm of air: {check_893:.15e}")
            print(f"Verify Air [mrad/sqrt(mm)]: check on 100 cm of air: {check_1000:.15e}, check on 160 cm of air: {check_1600:.15e}, check on 1760 cm of air: {check_1760:.15e}")
            
            window_angle_delta = vacuum_delta[0]
            window_angle_delta_theta = vacuum_delta_theta[0]
            nozzle_ripple_angle_delta = vacuum_delta[1]
            nozzle_ripple_angle_delta_theta = vacuum_delta_theta[1]
            
            air_133_delta = np.sqrt(air_delta[0]**2 - window_angle_delta**2)
            air_160_delta = np.sqrt(air_delta[1]**2 -nozzle_ripple_angle_delta**2)
            air_293_delta = np.sqrt (air_delta[1]** 2 + air_delta[0]**2 - nozzle_ripple_angle_delta**2 - window_angle_delta**2)
            air_600_delta  = air_delta[2]
            air_760_delta = np.sqrt(air_delta[2]**2 + air_delta[1]**2-nozzle_ripple_angle_delta**2)
            air_893_delta = np.sqrt(air_delta[2]**2  + air_delta[1]** 2 + air_delta[0]**2 - nozzle_ripple_angle_delta**2 - window_angle_delta**2)
            air_1000_delta = air_delta[3]
            air_1600_delta = np.sqrt(air_delta[3]**2 + air_delta[2]**2)
            air_1760_delta= np.sqrt(air_delta[3]**2 + air_delta[2]**2  + air_delta[1]** 2 - nozzle_ripple_angle_delta**2)
            air_1893_delta = np.sqrt(air_delta[3]**2 + air_delta[2]**2  + air_delta[1]** 2 + air_delta[0]**2 - nozzle_ripple_angle_delta**2 - window_angle_delta**2)

            check_133_delta = air_133_delta /np.sqrt(133)
            check_160_delta = air_160_delta / np.sqrt(160)
            check_293_delta = air_293_delta/np.sqrt(293)
            check_600_delta  = air_600_delta  / np.sqrt(600)
            check_760_delta = air_760_delta / np.sqrt(760)
            check_893_delta = air_893_delta/np.sqrt(893)
            check_1000_delta = air_1000_delta / np.sqrt(1000)
            check_1600_delta = air_1600_delta / np.sqrt(1600)
            check_1760_delta = air_1760_delta / np.sqrt(1760)
            check_1893_delta = air_1893_delta / np.sqrt(1893)




            print()
            print(f"\033[1m--From the deltas--\033[0m")
            print()
            print(f"Standard deviations in {attr[0]}':")
            print(f"Vacuum window [mrad]: {window_angle_delta:.15e}")
            print(f"Nozzle and Ripple filters [mrad]: {nozzle_ripple_angle_delta:.15e}")
            print(f"Verify Air [mrad/sqrt(mm)]: check on 13,3 cm of air: {check_133_delta:.15e}, check on 16 cm of air: {check_160_delta:.15e}, check on 29,3 cm of air: {check_293_delta:.15e}")
            print(f"Verify Air [mrad/sqrt(mm)]: check on 60 cm of air: {check_600_delta:.15e}, check on 76 cm of air: {check_760_delta:.15e}, check on 89,3 cm of air: {check_893_delta:.15e}")
            print(f"Verify Air [mrad/sqrt(mm)]: check on 100 cm of air: {check_1000_delta:.15e}, check on 160 cm of air: {check_1600_delta:.15e}, check on 1760 cm of air: {check_1760_delta:.15e}")
            
            stats_angle[attr] = {
                "window_angle_std": window_angle_std,
                "window_theta_std": window_theta_std,
                "window_angle_variation_std": window_angle_delta,
                "window_theta_variation_std": window_angle_delta_theta,
                "nozzle_ripple_angle_std": nozzle_ripple_angle_std,
                "nozzle_ripple_theta_std": nozzle_ripple_theta_std,
                "nozzle_ripple_angle_variation_std": nozzle_ripple_angle_delta,
                "nozzle_ripple_theta_variation_std": nozzle_ripple_angle_delta_theta,
                "air_133": air_133,
                "air_160": air_160,
                "air_293": air_293,
                "air_600": air_600,
                "air_760": air_760,
                "air_893": air_893,
                "air_1000": air_1000,
                "air_1600": air_1600,
                "air_1760": air_1760,
                "air_1893": air_1893,
                "check_133": check_133,
                "check_160": check_160,
                "check_293": check_293,
                "check_600": check_600,
                "check_760": check_760,
                "check_893": check_893,
                "check_1000": check_1000,
                "check_1600": check_1600,
                "check_1760": check_1760,
                "check_1893": check_1893,
                "air_133_delta": air_133_delta,
                "air_160_delta": air_160_delta,
                "air_293_delta": air_293_delta,
                "air_600_delta": air_600_delta,
                "air_760_delta": air_760_delta,
                "air_893_delta": air_893_delta,
                "air_1000_delta": air_1000_delta,
                "air_1600_delta": air_1600_delta,
                "air_1760_delta": air_1760_delta,
                "air_1893_delta": air_1893_delta,
                "check_133_delta": check_133_delta,
                "check_160_delta": check_160_delta,
                "check_293_delta": check_293_delta,
                "check_600_delta": check_600_delta,
                "check_760_delta": check_760_delta,
                "check_893_delta": check_893_delta,
                "check_1000_delta": check_1000_delta,
                "check_1600_delta": check_1600_delta,
                "check_1760_delta": check_1760_delta,
                "check_1893_delta": check_1893_delta,
            }


    return stats_angle


def get_checks_for_attr(stats_angle, attr):
    s = stats_angle[attr]
    checks = np.array([
        s["check_133"],
        s["check_160"],
        s["check_293"],
        s["check_600"],
        s["check_760"],
        s["check_893"],
        s["check_1000"],
        s["check_1600"],
        s["check_1760"],
        s["check_1893"],
    ])
    checks_delta = np.array([
        s["check_133_delta"],
        s["check_160_delta"],
        s["check_293_delta"],
        s["check_600_delta"],
        s["check_760_delta"],
        s["check_893_delta"],
        s["check_1000_delta"],
        s["check_1600_delta"],
        s["check_1760_delta"],
        s["check_1893_delta"],
    ])
    return checks, checks_delta

def analyze_surface_light(df_original, df_filtered, position, label,
                          use_filtered_for_delta=False):

        df_orig_surf = df_original[df_original['scoring_surface'] == position].copy()
        df_filt_surf = df_filtered[df_filtered['scoring_surface'] == position].copy()

        sigma_xp = df_filt_surf['x_prime'].std()
        sigma_yp = df_filt_surf['y_prime'].std()

        base_df_for_delta = df_filtered

        df_delta, delta_xp, delta_yp, delta_energy, _ = calculate_variations(
            base_df_for_delta,
            position,
            label,
        )

        sigma_var_xp = np.nanstd(delta_xp)
        sigma_var_yp = np.nanstd(delta_yp)

        statistics = {
            "position": position,
            "sigma_xp": sigma_xp,
            "sigma_yp": sigma_yp,
            "sigma_var_xp": sigma_var_xp,
            "sigma_var_yp": sigma_var_yp,
        }

        return statistics, df_filt_surf


def sigma_from_gaussian_fit(x, pdf, fit_window=None):
    """Sigma dal FIT GAUSSIANO del core, robusto a ringing e rumore."""

    try:
        from scipy.optimize import curve_fit
        _HAS_SCIPY = True
    except Exception:
        _HAS_SCIPY = False
    x = np.asarray(x, float)
    pdf = normalize_pdf(pdf, x)
    if not _HAS_SCIPY:
        return sigma_from_pdf_norm(x, pdf)
    mu0 = x[np.argmax(pdf)]
    if fit_window is not None:
        m = np.abs(x - mu0) <= fit_window
        xf, pf = x[m], pdf[m]
    else:
        xf, pf = x, pdf
    A0 = pf.max()
    s0 = max((xf.max() - xf.min()) / 6.0, 1e-3)

    def _g(xx, A, mu, s):
        return A * np.exp(-0.5 * ((xx - mu) / s) ** 2)

    try:
        popt, _ = curve_fit(_g, xf, pf, p0=[A0, mu0, s0], maxfev=10000)
        return float(abs(popt[2]))
    except Exception:
        return sigma_from_pdf_norm(x, pdf)


def measure_sigma(
    x,
    pdf,
    method="fit",
    fit_window=None,
):
    """
    method='fit':
        sigma del fit gaussiano della PDF.

    method='moment':
        deviazione standard dell'intera PDF.

    Non effettua un fallback silenzioso da fit a momento.
    """
    x = np.asarray(x, dtype=float)
    pdf = np.asarray(pdf, dtype=float)

    if x.ndim != 1 or pdf.shape != x.shape:
        raise ValueError(
            "x e pdf devono essere array 1D della stessa lunghezza."
        )

    if (
        x.size < 3
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(pdf))
        or np.any(np.diff(x) <= 0.0)
    ):
        raise ValueError(
            "x e pdf devono essere finiti; x deve essere crescente."
        )

    if method == "moment":
        return sigma_from_pdf_norm(x, pdf)

    if method != "fit":
        raise ValueError(
            "method deve essere 'fit' oppure 'moment'."
        )

    x_fit = x
    pdf_fit = pdf

    if fit_window is not None:
        half_width = float(fit_window)

        if not np.isfinite(half_width) or half_width <= 0.0:
            raise ValueError(
                "fit_window deve essere una semilarghezza positiva."
            )

        peak_position = x[np.argmax(pdf)]
        mask = np.abs(x - peak_position) <= half_width

        if np.count_nonzero(mask) < 5:
            raise ValueError(
                "Troppi pochi punti nella finestra del fit."
            )

        x_fit = x[mask]
        pdf_fit = pdf[mask]

    sigma = float(
        fit_sigma_from_sampled_pdf(
            x_fit,
            pdf_fit,
        )
    )

    if not np.isfinite(sigma) or sigma <= 0.0:
        raise RuntimeError(
            "Il fit gaussiano della PDF non ha restituito "
            "una sigma valida. Nessun fallback al momento."
        )

    return sigma

from scipy.fft import fft, ifft, next_fast_len


def validate_deconvolution_inputs(signal, kernel, xgrid):
    signal = np.asarray(signal, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    xgrid = np.asarray(xgrid, dtype=float)

    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")

    if kernel.ndim != 1:
        raise ValueError("kernel must be one-dimensional.")

    if xgrid.ndim != 1:
        raise ValueError("xgrid must be one-dimensional.")

    if signal.size != kernel.size or signal.size != xgrid.size:
        raise ValueError("signal, kernel and xgrid must have the same length.")

    if signal.size < 3:
        raise ValueError("At least three grid points are required.")

    if not np.all(np.isfinite(xgrid)):
        raise ValueError("xgrid contains non-finite values.")

    dx_values = np.diff(xgrid)

    if np.any(dx_values <= 0.0):
        raise ValueError("xgrid must be strictly increasing.")

    dx = float(np.mean(dx_values))

    if not np.allclose(dx_values, dx, rtol=1e-8, atol=1e-12):
        raise ValueError("xgrid must be uniformly spaced.")

    signal = np.where(np.isfinite(signal), signal, 0.0)
    kernel = np.where(np.isfinite(kernel), kernel, 0.0)

    return signal, kernel, xgrid, dx


def wiener_deconvolution_fixed_lambda(
    signal,
    kernel,
    xgrid,
    lambd,
    return_raw=False
):
    """
    Wiener deconvolution on a centered uniform grid.

    This implementation uses one consistent n-point FFT convention:
    - PDFs are centered in physical x-space;
    - ifftshift moves x=0 to index 0 before FFT;
    - the result is shifted back once after IFFT;
    - no zero-padding/cropping is applied inside the inverse.

    The function returns a non-negative normalized PDF on xgrid.
    """
    signal, kernel, xgrid, dx = validate_deconvolution_inputs(
        signal,
        kernel,
        xgrid,
    )

    lambd = float(lambd)

    if not np.isfinite(lambd) or lambd <= 0.0:
        raise ValueError(
            "lambd must be finite and positive."
        )

    signal = normalize_pdf(
        signal,
        xgrid,
    )

    kernel = normalize_pdf(
        kernel,
        xgrid,
    )

    # Move physical x=0 from the centre of the arrays to index zero.
    signal_zero_lag = np.fft.ifftshift(
        signal
    )

    kernel_zero_lag = np.fft.ifftshift(
        kernel
    )

    signal_fft = fft(
        signal_zero_lag,
    )

    kernel_fft = fft(
        kernel_zero_lag,
    )

    kernel_power = np.abs(
        kernel_fft
    ) ** 2

    # Discrete Wiener inverse.
    #
    # dx is NOT used here because both input PDFs are represented as
    # sampled normalized arrays and the final result is normalized again.
    #
    # The key requirement is that signal/kernel use the same n-point
    # convention and no padded/cropped array is mixed into this inverse.
    wiener_filter = np.conj(
        kernel_fft
    ) / (
        kernel_power
        + lambd
    )

    recovered_zero_lag = np.real(
        ifft(
            signal_fft * wiener_filter
        )
    )


    recovered = np.fft.fftshift(recovered_zero_lag)
    recovered = np.where(np.isfinite(recovered), recovered, 0.0)

    raw_area = np.trapezoid(recovered, xgrid)

    if not np.isfinite(raw_area) or raw_area <= 0.0:
        raise ValueError(
            "La soluzione Wiener ha area non positiva o non finita."
        )

    raw_pdf = recovered / raw_area

    negative_mass = np.trapezoid(
        np.clip(-raw_pdf, 0.0, None),
        xgrid,
    )

    physical_pdf = normalize_pdf(
        np.clip(raw_pdf, 0.0, None),
        xgrid,
    )

    if return_raw:
        return physical_pdf, {
            "raw_pdf": raw_pdf.copy(),
            "negative_mass": float(negative_mass),
        }

    return physical_pdf


def validate_deconvolution_inputs(
    signal,
    kernel,
    xgrid,
):
    signal = np.asarray(
        signal,
        dtype=float,
    )

    kernel = np.asarray(
        kernel,
        dtype=float,
    )

    xgrid = np.asarray(
        xgrid,
        dtype=float,
    )

    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")

    if kernel.ndim != 1:
        raise ValueError("kernel must be one-dimensional.")

    if xgrid.ndim != 1:
        raise ValueError("xgrid must be one-dimensional.")

    if signal.size != kernel.size:
        raise ValueError(
            "signal and kernel must have the same size."
        )

    if signal.size != xgrid.size:
        raise ValueError(
            "signal and xgrid must have the same size."
        )

    if signal.size < 3:
        raise ValueError(
            "At least three grid points are required."
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            "signal contains non-finite values."
        )

    if not np.all(np.isfinite(kernel)):
        raise ValueError(
            "kernel contains non-finite values."
        )

    if not np.all(np.isfinite(xgrid)):
        raise ValueError(
            "xgrid contains non-finite values."
        )

    dx_values = np.diff(xgrid)

    if np.any(dx_values <= 0.0):
        raise ValueError(
            "xgrid must be strictly increasing."
        )

    dx = float(np.mean(dx_values))

    if not np.allclose(
        dx_values,
        dx,
        rtol=1.0e-8,
        atol=1.0e-12,
    ):
        raise ValueError(
            "xgrid must be uniformly spaced."
        )

    return signal, kernel, xgrid, dx

def relative_forward_error(
    measured_pdf,
    reconvolved_pdf,
    xgrid,
):
    measured_pdf = normalize_pdf(
        measured_pdf,
        xgrid,
    )

    reconvolved_pdf = normalize_pdf(
        reconvolved_pdf,
        xgrid,
    )

    numerator = np.trapezoid(
        (reconvolved_pdf - measured_pdf) ** 2,
        xgrid,
    )

    denominator = np.trapezoid(
        measured_pdf ** 2,
        xgrid,
    )

    return float(
        numerator / max(denominator, 1.0e-30)
    )

def relative_curvature_penalty(
    pdf,
    xgrid,
):
    pdf = normalize_pdf(
        pdf,
        xgrid,
    )

    dx = float(xgrid[1] - xgrid[0])

    first_derivative = np.gradient(
        pdf,
        dx,
    )

    second_derivative = np.gradient(
        first_derivative,
        dx,
    )

    curvature = np.trapezoid(
        second_derivative ** 2,
        xgrid,
    )

    norm = np.trapezoid(
        pdf ** 2,
        xgrid,
    )

    return float(
        (dx ** 4) * curvature
        / max(norm, 1.0e-30)
    )



def deconvolution_forward_objective(
    lambd,
    signal,
    kernel,
    xgrid,
    roughness_weight=0.0,
    core_weight=1.0,
    cdf_weight=1.0,
    edge_weight=1.0,
    edge_fraction=0.05,
):
    """
    Objective basata esclusivamente sul residuo relativo quadratico
    tra PDF misurata e PDF riconvoluta.

    Gli argomenti dei vecchi pesi sono mantenuti per compatibilità,
    ma non vengono utilizzati.
    """
    del (
        roughness_weight,
        core_weight,
        cdf_weight,
        edge_weight,
        edge_fraction,
    )

    signal, kernel, xgrid, _ = validate_deconvolution_inputs(
        signal,
        kernel,
        xgrid,
    )

    lambd = float(lambd)

    if not np.isfinite(lambd) or lambd <= 0.0:
        raise ValueError("lambd deve essere finita e positiva.")

    measured_pdf = normalize_pdf(signal, xgrid)
    kernel_pdf = normalize_pdf(kernel, xgrid)

    candidate_pdf, raw_info = wiener_deconvolution_fixed_lambda(
        signal=measured_pdf,
        kernel=kernel_pdf,
        xgrid=xgrid,
        lambd=lambd,
        return_raw=True,
    )

    reconvolved_pdf = reconvolve_same_grid(
        candidate_pdf,
        kernel_pdf,
        xgrid,
    )

    forward_error = float(
        relative_forward_error(
            measured_pdf=measured_pdf,
            reconvolved_pdf=reconvolved_pdf,
            xgrid=xgrid,
        )
    )

    raw_pdf = raw_info["raw_pdf"]

    negative_area = float(
        np.trapezoid(
            np.maximum(-raw_pdf, 0.0),
            xgrid,
        )
    )

    absolute_area = float(
        np.trapezoid(np.abs(raw_pdf), xgrid)
    )

    negative_fraction = (
        negative_area / max(absolute_area, 1.0e-30)
    )

    return {
        "objective": forward_error,
        "forward_error": forward_error,
        "forward_residual": float(
            np.sqrt(max(forward_error, 0.0))
        ),
        "negative_fraction_raw": float(negative_fraction),
        "roughness": 0.0,
        "candidate": candidate_pdf,
        "reconvolved": reconvolved_pdf,
        "raw_candidate": raw_pdf,
    }

def find_optimal_lambda_forward(
    signal,
    kernel,
    xgrid,
    lambda_min=1.0e-20,
    lambda_max=1.0e8,
    n_grid=41,
    n_iter=6,
    shrink_factor=0.50,
    roughness_weight=0.0,
    verbose=False,
):
    """
    Seleziona lambda minimizzando esclusivamente il residuo forward.
    Nessun criterio obbligatorio di accettazione.
    """
    del roughness_weight

    signal, kernel, xgrid, _ = validate_deconvolution_inputs(
        signal,
        kernel,
        xgrid,
    )

    lambda_min = float(lambda_min)
    lambda_max = float(lambda_max)

    if (
        not np.isfinite(lambda_min)
        or not np.isfinite(lambda_max)
        or not 0.0 < lambda_min < lambda_max
    ):
        raise ValueError(
            "Richiesto: 0 < lambda_min < lambda_max."
        )

    n_grid = int(n_grid)
    n_iter = int(n_iter)
    shrink_factor = float(shrink_factor)

    if n_grid < 3 or n_iter < 1:
        raise ValueError("n_grid >= 3 e n_iter >= 1.")

    if (
        not np.isfinite(shrink_factor)
        or not 0.0 < shrink_factor <= 1.0
    ):
        raise ValueError(
            "shrink_factor deve essere tra 0 e 1."
        )

    measured_pdf = normalize_pdf(signal, xgrid)
    kernel_pdf = normalize_pdf(kernel, xgrid)

    lo = lambda_min
    hi = lambda_max
    best = None
    history = []

    for iteration in range(n_iter):
        points = max(n_grid, 81) if iteration == 0 else n_grid

        lambda_grid = np.logspace(
            np.log10(lo),
            np.log10(hi),
            points,
        )

        errors = np.full(points, np.inf)

        for index, lambd in enumerate(lambda_grid):
            result = deconvolution_forward_objective(
                lambd=float(lambd),
                signal=measured_pdf,
                kernel=kernel_pdf,
                xgrid=xgrid,
            )

            error = float(result["forward_error"])

            if not np.isfinite(error):
                continue

            errors[index] = error

            if best is None or error < best["error_opt"]:
                best = {
                    "lambda_opt": float(lambd),
                    "error_opt": error,
                    "forward_error_opt": error,
                    "forward_residual_opt": float(
                        result["forward_residual"]
                    ),
                    "negative_fraction_raw_opt": float(
                        result["negative_fraction_raw"]
                    ),
                    "roughness_opt": 0.0,
                    "deconvolved_opt": (
                        result["candidate"].copy()
                    ),
                    "reconvolved_opt": (
                        result["reconvolved"].copy()
                    ),
                }

        finite_indices = np.flatnonzero(
            np.isfinite(errors)
        )

        if finite_indices.size == 0:
            if best is None:
                raise RuntimeError(
                    "Nessuna candidata ha prodotto "
                    "un residuo forward finito."
                )
            break

        best_index = int(
            finite_indices[
                np.argmin(errors[finite_indices])
            ]
        )

        history.append({
            "iteration": iteration,
            "lambda_grid": lambda_grid.copy(),
            "objective_values": errors.copy(),
            "best_lambda_iteration": float(
                lambda_grid[best_index]
            ),
            "best_objective_iteration": float(
                errors[best_index]
            ),
        })

        if verbose:
            print(
                f"[forward-only {iteration}] "
                f"lambda={lambda_grid[best_index]:.8e} | "
                f"errore quadratico={errors[best_index]:.8e} | "
                f"residuo L2="
                f"{np.sqrt(max(errors[best_index], 0.0)):.8e}"
            )

        left = max(best_index - 1, 0)
        right = min(best_index + 1, points - 1)

        log_center = np.log10(
            lambda_grid[best_index]
        )

        half_width = max(
            0.5
            * (
                np.log10(lambda_grid[right])
                - np.log10(lambda_grid[left])
            )
            * shrink_factor,
            0.05,
        )

        lo = max(
            10.0 ** (log_center - half_width),
            lambda_min,
        )

        hi = min(
            10.0 ** (log_center + half_width),
            lambda_max,
        )

        if hi <= lo * (1.0 + 1.0e-10):
            break

    best["history"] = history
    best["status"] = "forward_minimum"

    best["at_lower_bound"] = bool(
        np.isclose(
            best["lambda_opt"],
            lambda_min,
            rtol=1.0e-3,
            atol=0.0,
        )
    )

    best["at_upper_bound"] = bool(
        np.isclose(
            best["lambda_opt"],
            lambda_max,
            rtol=1.0e-3,
            atol=0.0,
        )
    )

    if verbose:
        print("\n--- LAMBDA SELEZIONATA: SOLO FORWARD ---")
        print(f"Lambda = {best['lambda_opt']:.8e}")
        print(
            f"Residuo relativo L2 = "
            f"{best['forward_residual_opt']:.8e}"
        )
        print(
            f"Negatività raw, solo diagnostica = "
            f"{best['negative_fraction_raw_opt']:.8e}"
        )

        if best["at_lower_bound"] or best["at_upper_bound"]:
            print(
                "WARNING: lambda è a un limite della ricerca; "
                "il minimo non è necessariamente interno."
            )

    return best

def _objective(lambd, signal, kernel, xgrid, w_forward=1.0, w_rough=0.0,
               sigma_method="fit", fit_window=None):
    cand = wiener_deconvolution_fixed_lambda(signal, kernel, xgrid, lambd)
    s_cand = measure_sigma(xgrid, cand, method=sigma_method, fit_window=fit_window)
    if w_forward > 0.0:
        reco = reconvolve_same_grid(cand, kernel, xgrid)
        sig_n = normalize_pdf(signal, xgrid)
        num = np.trapezoid((reco - sig_n) ** 2, xgrid)
        den = np.trapezoid(sig_n ** 2, xgrid) + 1e-30
        forward_term = num / den
    else:
        forward_term = 0.0
    if w_rough > 0.0:
        dx = float(xgrid[1] - xgrid[0])
        grad = np.gradient(cand, dx)
        rough_term = np.trapezoid(grad ** 2, xgrid)
    else:
        rough_term = 0.0
    total = w_forward * forward_term + w_rough * rough_term
    return total, cand, s_cand


def find_optimal_lambda(signal, kernel, xgrid,
                        lambda_min=1e-20, lambda_max=1e5,
                        n_grid=21, n_iter=6, shrink=0.35,
                        w_forward=1.0, w_rough=0.0,
                        sigma_method="fit", fit_window=None,
                        verbose=False):
    lo, hi = float(lambda_min), float(lambda_max)
    best = (np.inf, None, None, None)
    for it in range(n_iter):
        lambdas = np.logspace(np.log10(lo), np.log10(hi), n_grid)
        errs = []
        for lam in lambdas:
            err, cand, s = _objective(lam, signal, kernel, xgrid, w_forward, w_rough,
                                      sigma_method=sigma_method,
                                      fit_window=fit_window)
            errs.append(err)
            if err < best[0]:
                best = (err, float(lam), cand.copy(), float(s))
        errs = np.asarray(errs)
        idx = int(np.argmin(errs))
        if verbose:
            print(f"[iter {it}] lambda*={lambdas[idx]:.3e} err={errs[idx]:.4e} "
                  f"sigma_cand={best[3]:.4f}")
        l_idx = max(idx - 1, 0)
        r_idx = min(idx + 1, len(lambdas) - 1)
        logc = np.log10(lambdas[idx])
        half = max(0.5 * (np.log10(lambdas[r_idx]) - np.log10(lambdas[l_idx])) * shrink, 0.05)
        lo = max(10 ** (logc - half), lambda_min)
        hi = min(10 ** (logc + half), lambda_max)
        if hi <= lo * 1.0000001:
            break
    return {"lambda_opt": best[1], "deconvolved_opt": best[2],
            "sigma_opt": best[3], "error_opt": best[0]}


def deconvolve_wiener(
    signal,
    kernel,
    xgrid,
    lambd=None,
    auto_lambda=True,
    lambda_min=1.0e-20,
    lambda_max=1.0e8,
    roughness_weight=0.0,
    return_info=False,
    verbose=False,
):
    """
    Wiener con selezione di lambda basata soltanto
    sulla riconvoluzione.

    Per usare una lambda fissata:
        lambd=valore, auto_lambda=False
    """
    del roughness_weight

    if auto_lambda or lambd is None:
        optimization = find_optimal_lambda_forward(
            signal=signal,
            kernel=kernel,
            xgrid=xgrid,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            verbose=verbose,
        )

        output_pdf = optimization["deconvolved_opt"]

    else:
        result = deconvolution_forward_objective(
            lambd=float(lambd),
            signal=signal,
            kernel=kernel,
            xgrid=xgrid,
        )

        output_pdf = result["candidate"]

        optimization = {
            "lambda_opt": float(lambd),
            "error_opt": float(result["forward_error"]),
            "forward_error_opt": float(
                result["forward_error"]
            ),
            "forward_residual_opt": float(
                result["forward_residual"]
            ),
            "negative_fraction_raw_opt": float(
                result["negative_fraction_raw"]
            ),
            "roughness_opt": 0.0,
            "deconvolved_opt": output_pdf.copy(),
            "reconvolved_opt": result["reconvolved"].copy(),
            "history": [],
            "status": "fixed_lambda",
        }

        if verbose:
            print(
                f"[lambda fissata] lambda={float(lambd):.8e} | "
                f"residuo L2={result['forward_residual']:.8e}"
            )

    if return_info:
        return output_pdf, optimization

    return output_pdf

# Somme in quadaratura
def sumQuadProp(z_scatt, sigma_scatt_angle, sigma0_space, sigma0_angle, z0, z_scorings):
    Zout = np.concatenate(([z0], np.asarray(z_scorings, float)))
    
    sigmas_space = np.zeros(len(Zout))
    sigmas_angle = np.zeros(len(Zout))

    sigmas_space[0] = sigma0_space
    sigmas_angle[0] = sigma0_angle

    for i in range(1, len(Zout)):
        var_scatt = sigma0_angle**2
        z_current = Zout[i]
        for k in range(len(z_scatt)):
            distance = z_current - z_scatt[k]
            if distance > 0:
                var_scatt += sigma_scatt_angle[k]**2
        sigmas_angle[i] = np.sqrt(var_scatt)
        
        term_initial = (sigma0_angle * 1e-3 * (z_current - Zout[0]))**2
        term_scatt = 0
        for j in range(i):
            dist = z_current - z_scatt[j]
            if dist > 0:
                term_scatt += (sigma_scatt_angle[j] * 1e-3 * dist)**2
        
        sigmas_space[i] = np.sqrt(sigma0_space**2 + term_initial + term_scatt)
        
    return Zout, sigmas_space, sigmas_angle



# Convoluzioni


def state_space_propagation(
    z_scatt,
    sigma_scatt_angle,
    sigma0_space,
    sigma0_angle,
    z0,
    z_scorings,
):
    z_scatt = np.asarray(z_scatt, dtype=float)
    sigma_scatt_angle = np.asarray(sigma_scatt_angle, dtype=float)

    Zout = np.concatenate(([z0], np.asarray(z_scorings, dtype=float)))
    nZ = len(Zout)

    sigmas_space = np.zeros(nZ, dtype=float)
    sigmas_angle = np.zeros(nZ, dtype=float)
    covariance = np.zeros(nZ, dtype=float)

    P0 = np.array([
        [sigma0_space**2, 0.0],
        [0.0, sigma0_angle**2]
    ], dtype=float)

    sigmas_space[0] = sigma0_space
    sigmas_angle[0] = sigma0_angle
    covariance[0] = 0.0

    for i in range(1, nZ):
        z_curr = Zout[i]
        dz0 = z_curr - z0

        A0 = np.array([
            [1.0, dz0 * 1e-3],
            [0.0, 1.0]
        ], dtype=float)

        P = A0 @ P0 @ A0.T

        for z_s, sigma_s in zip(z_scatt, sigma_scatt_angle):
            if z_s < z_curr:
                dzs = z_curr - z_s

                Qs = np.array([
                    [0.0, 0.0],
                    [0.0, sigma_s**2]
                ], dtype=float)

                As = np.array([
                    [1.0, dzs * 1e-3],
                    [0.0, 1.0]
                ], dtype=float)

                P += As @ Qs @ As.T

        sigmas_space[i] = np.sqrt(max(P[0, 0], 0.0))
        sigmas_angle[i] = np.sqrt(max(P[1, 1], 0.0))
        covariance[i] = P[0, 1]

    return Zout, sigmas_space, sigmas_angle, covariance

def fit_sigma_from_sampled_pdf(x, pdf):
    x = np.asarray(x, dtype=float)
    pdf = np.asarray(pdf, dtype=float)

    if len(x) < 3 or np.all(pdf <= 0):
        return np.nan

    bin_width = float(np.mean(np.diff(x)))

    N0 = np.sum(pdf)
    mu0 = x[np.argmax(pdf)]

    if np.sum(pdf) > 0:
        sigma0 = np.sqrt(np.sum(pdf * (x - mu0)**2) / np.sum(pdf))
    else:
        sigma0 = bin_width

    baseline0 = max(np.min(pdf), 0.0)

    def model(x, N_eff, mu, sigma, baseline):
        return gaussian_model_counts(x, N_eff, mu, sigma, baseline, bin_width)

    try:
        popt, _ = curve_fit(
            model,
            x,
            pdf,
            p0=[N0, mu0, sigma0, baseline0],
            bounds=([0.0, x.min(), 1e-9, 0.0],
                    [np.inf, x.max(), np.inf, np.inf]),
            maxfev=50000
        )
        sigmafit = float(abs(popt[2]))
        return sigmafit
    except Exception:
        return np.nan
    

def diffQuad(z_scatt, sigma_scatt_angle, sigma4_space, sigma4_angle, z0, z_scorings):
    Zout = np.concatenate(([z0], np.asarray(z_scorings, float)))
    sigmas_space = np.zeros(len(Zout))
    sigmas_angle = np.zeros(len(Zout))

    sigmas_space[-1] = sigma4_space
    sigmas_angle[-1] = sigma4_angle

    for i in range(len(Zout)-1, 0, -1):
        d1 = z_scatt[i-1] - Zout[i-1]
        d2 = Zout[i] - z_scatt[i-1]
        sigmas_angle[i-1] = np.sqrt(sigmas_angle[i]**2 - sigma_scatt_angle[i-1]**2)
        sigmas_space[i-1] = np.sqrt(sigmas_space[i]**2 - (sigmas_angle[i-1]*1e-3*(d1))**2-(sigmas_angle[i]*1e-3*(d2))**2)
        
    return Zout, sigmas_space, sigmas_angle

def diffQuad_prop(z_scatt, sigma_scatt_angle, z0, z_scorings, sigmas_angle_measured, sigmas_space_measured):    

    Zout = np.concatenate(([z0], np.asarray(z_scorings, float)))
    sigmas_space = np.zeros(len(Zout))
    sigmas_angle = np.zeros(len(Zout))

    print (sigmas_angle_measured[0])

    sigmas_angle[0] = sigmas_angle_measured[0]
    sigmas_space[0] = sigmas_space_measured[0]

    for i in range(1, len(Zout)):
        z_current = Zout[i]

        term_angle_total = sigmas_angle_measured[i]**2
        term_angle_scatt = 0

        for j in range(len(z_scatt)):
            if z_scatt[j] < z_current:
                term_angle_scatt += sigma_scatt_angle[j]**2

        sigmas_angle[i] = np.sqrt(term_angle_total - term_angle_scatt)

        term_space_total = sigmas_space_measured[i]**2  
        term_space_scatt = 0
        for j in range(len(z_scatt)):
            dist_scatt_to_surf = z_current - z_scatt[j]
            if dist_scatt_to_surf > 0:
                term_space_scatt += (sigma_scatt_angle[j] * 1e-3 * dist_scatt_to_surf)**2

        sigmas_space[i] = np.sqrt(term_space_total - term_space_scatt)

        print(f"Z: {z_current}, Total: {term_angle_total:.6f}, Scatt: {term_angle_scatt:.6f}")
        
    return Zout, sigmas_space, sigmas_angle


def state_space_diffquad_prop(
    z_scatt,
    sigma_scatt_angle,
    z0,
    z_scorings,
    sigmas_angle_measured,
    sigmas_space_measured,
    covariance_measured=None,
):
    z_scatt = np.asarray(z_scatt, dtype=float)
    sigma_scatt_angle = np.asarray(sigma_scatt_angle, dtype=float)

    Zout = np.concatenate(([z0], np.asarray(z_scorings, dtype=float)))
    nZ = len(Zout)

    sigmas_space = np.zeros(nZ, dtype=float)
    sigmas_angle = np.zeros(nZ, dtype=float)
    covariance = np.zeros(nZ, dtype=float)

    if covariance_measured is None:
        covariance_measured = np.zeros(nZ, dtype=float)
    else:
        covariance_measured = np.asarray(covariance_measured, dtype=float)

    sigmas_space[0] = sigmas_space_measured[0]
    sigmas_angle[0] = sigmas_angle_measured[0]
    covariance[0] = covariance_measured[0]

    for i in range(1, nZ):
        z_curr = Zout[i]
        dz = z_curr - z0

        # matrice misurata alla scoring surface i
        P_meas = np.array([
            [sigmas_space_measured[i]**2, covariance_measured[i]],
            [covariance_measured[i],      sigmas_angle_measured[i]**2]
        ], dtype=float)

        # somma dei contributi degli scatterer propagati fino a z_curr
        P_scatt = np.zeros((2, 2), dtype=float)

        for z_s, sigma_s_mrad in zip(z_scatt, sigma_scatt_angle):
            if z_s < z_curr:
                dz_s = z_curr - z_s

                Q_s = np.array([
                    [0.0, 0.0],
                    [0.0, sigma_s_mrad**2]
                ], dtype=float)

                A_s = np.array([
                    [1.0, dz_s * 1e-3],
                    [0.0, 1.0]
                ], dtype=float)

                P_scatt += A_s @ Q_s @ A_s.T

        # rimozione contributi scatterer
        P_clean = P_meas - P_scatt

        # inversione del drift dalla sorgente a z_curr
        A0 = np.array([
            [1.0, dz * 1e-3],
            [0.0, 1.0]
        ], dtype=float)

        A0_inv = np.array([
            [1.0, -dz * 1e-3],
            [0.0, 1.0]
        ], dtype=float)

        P0_est = A0_inv @ P_clean @ A0_inv.T

        # protezione numerica
        P0_est[0, 0] = max(P0_est[0, 0], 0.0)
        P0_est[1, 1] = max(P0_est[1, 1], 0.0)

        if P0_est[0, 0] * P0_est[1, 1] < P0_est[0, 1]**2:
            P0_est[0, 1] = np.sign(P0_est[0, 1]) * np.sqrt(
                max(P0_est[0, 0] * P0_est[1, 1], 0.0)
            )

        sigmas_space[i] = np.sqrt(P0_est[0, 0])
        sigmas_angle[i] = np.sqrt(P0_est[1, 1])
        covariance[i] = P0_est[0, 1]

    return Zout, sigmas_space, sigmas_angle, covariance


def fit_sigma_from_sampled_pdf(x, pdf):
    x = np.asarray(x, dtype=float)
    pdf = np.asarray(pdf, dtype=float)

    if len(x) < 3 or np.all(pdf <= 0):
        return np.nan

    bin_width = float(np.mean(np.diff(x)))

    N0 = np.sum(pdf)
    mu0 = x[np.argmax(pdf)]

    if np.sum(pdf) > 0:
        sigma0 = np.sqrt(np.sum(pdf * (x - mu0)**2) / np.sum(pdf))
    else:
        sigma0 = bin_width

    baseline0 = max(np.min(pdf), 0.0)

    def model(x, N_eff, mu, sigma, baseline):
        return gaussian_model_counts(x, N_eff, mu, sigma, baseline, bin_width)

    try:
        popt, _ = curve_fit(
            model,
            x,
            pdf,
            p0=[N0, mu0, sigma0, baseline0],
            bounds=([0.0, x.min(), 1e-9, 0.0],
                    [np.inf, x.max(), np.inf, np.inf]),
            maxfev=50000
        )
        sigmafit = float(abs(popt[2]))
        return sigmafit
    except Exception:
        return np.nan
    
def clean_data(df, col):
    data = df[col].to_numpy()
    return data[np.isfinite(data)]


def gaussian_model_counts(x, N_eff, mu, sigma, baseline, bin_width):
    sigma = max(abs(float(sigma)), 1e-12)
    pdf = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return baseline + N_eff * pdf * bin_width

def deconvolutionFinal_with_air(
    z_scatt,
    sigma_scatt_angle,
    pdf_meas_space,
    sigma_meas_space,
    sigma_meas_angle,
    z0,
    z_scorings,
    xbin,
    air_segments=None,
    lambd=None,
    lambda_method="forward",
    target_relative_residual=1.0e-3,
    lambda_min=1.0e-20,
    lambda_max=1.0e8,
    roughness_weight=1.0e-2,
    sigma_method="fit",
    fit_window=None,
    verbose=False,
):
    """
    Deconvolve the transverse spatial PDF at every scoring plane.

    Parameters
    ----------
    z_scatt : array-like
        Positions of point-like scattering kicks [mm].

    sigma_scatt_angle : array-like
        RMS angular kick of each point-like scatterer [mrad].

    pdf_meas_space : sequence of 1D arrays
        PDFs measured on xbin. Index 0 corresponds to z0; the following
        PDFs correspond, in order, to z_scorings.

    sigma_meas_space : array-like
        Measured spatial RMS values [mm], including the source plane.

    sigma_meas_angle : array-like
        Measured angular RMS values [mrad], including the source plane.

    z0 : float
        Source/reference longitudinal position [mm].

    z_scorings : array-like
        Scoring-plane positions [mm].

    xbin : 1D array
        Uniform increasing spatial grid [mm].

    air_segments : sequence or None
        Distributed-air segments accepted by _distributed_scattering_moments.

    lambd : float or None
        If a finite positive value is supplied, use that fixed Wiener
        regularization parameter. If None, optimize lambda.

    lambda_method : str
        Retained for compatibility/documentation. The actual lambda choice
        is delegated to deconvolve_wiener(auto_lambda=True).

    target_relative_residual : float
        Retained for compatibility with previous calls. Not used directly
        here if lambda is selected by forward reconstruction.

    lambda_min, lambda_max : float
        Bounds for automatic lambda search.

    roughness_weight : float
        Curvature regularization weight passed to deconvolve_wiener if the
        latter supports it.

    sigma_method : {"fit", "moment"}
        Method used to calculate reported spatial RMS.

    fit_window : float or None
        Optional fit window for sigma_method="fit".

    verbose : bool
        Print numerical diagnostics.

    Returns
    -------
    Zout : ndarray
        z0 followed by z_scorings.

    sigmas_angle : ndarray
        Angular RMS after removing modeled point scatterers and air [mrad].

    pdf_space : ndarray, shape (nplanes, len(xbin))
        Physical deconvolved PDFs after non-negativity clipping and
        renormalization.

    diagnostics : dict
        Full diagnostics required to inspect measured PDF, scattering kernel,
        deconvolved PDF, reconvolved PDF and forward error at every plane.
    """
    del lambda_method, target_relative_residual

    z_scatt = np.asarray(z_scatt, dtype=float)
    sigma_scatt_angle = np.asarray(
        sigma_scatt_angle,
        dtype=float,
    )
    sigma_meas_space = np.asarray(
        sigma_meas_space,
        dtype=float,
    )
    sigma_meas_angle = np.asarray(
        sigma_meas_angle,
        dtype=float,
    )
    z_scorings = np.asarray(
        z_scorings,
        dtype=float,
    )
    xbin = np.asarray(xbin, dtype=float)

    if z_scatt.ndim != 1:
        raise ValueError("z_scatt must be one-dimensional.")

    if sigma_scatt_angle.ndim != 1:
        raise ValueError(
            "sigma_scatt_angle must be one-dimensional."
        )

    if z_scatt.shape != sigma_scatt_angle.shape:
        raise ValueError(
            "z_scatt and sigma_scatt_angle "
            "must have the same shape."
        )

    if (
        xbin.ndim != 1
        or xbin.size < 3
        or not np.all(np.isfinite(xbin))
        or np.any(np.diff(xbin) <= 0.0)
    ):
        raise ValueError(
            "xbin must be a finite, strictly increasing "
            "one-dimensional grid."
        )

    dx_values = np.diff(xbin)
    dx = float(np.mean(dx_values))

    if not np.allclose(
        dx_values,
        dx,
        rtol=1.0e-8,
        atol=1.0e-12,
    ):
        raise ValueError(
            "xbin must be uniformly spaced for FFT/Wiener "
            "deconvolution."
        )

    if lambda_min <= 0.0 or lambda_max <= lambda_min:
        raise ValueError(
            "Require 0 < lambda_min < lambda_max."
        )

    if lambd is not None:
        lambd = float(lambd)

        if not np.isfinite(lambd) or lambd <= 0.0:
            raise ValueError(
                "lambd must be None or a finite positive number."
            )

    Zout = np.concatenate(
        (
            np.array([float(z0)], dtype=float),
            z_scorings,
        )
    )

    nplanes = Zout.size

    if len(pdf_meas_space) != nplanes:
        raise ValueError(
            "pdf_meas_space must contain one PDF at z0 and "
            "one PDF for each scoring plane. "
            f"Received {len(pdf_meas_space)} PDFs; "
            f"expected {nplanes}."
        )

    if sigma_meas_space.size != nplanes:
        raise ValueError(
            "sigma_meas_space must contain one value at z0 "
            "and one value for each scoring plane."
        )

    if sigma_meas_angle.size != nplanes:
        raise ValueError(
            "sigma_meas_angle must contain one value at z0 "
            "and one value for each scoring plane."
        )

    sigmas_angle = np.full(
        nplanes,
        np.nan,
        dtype=float,
    )

    pdf_space = np.zeros(
        (nplanes, xbin.size),
        dtype=float,
    )

    pdf_measured_values = np.zeros(
        (nplanes, xbin.size),
        dtype=float,
    )

    pdf_kernel_values = np.zeros(
        (nplanes, xbin.size),
        dtype=float,
    )

    pdf_reconvolved_values = np.zeros(
        (nplanes, xbin.size),
        dtype=float,
    )

    lambda_values = np.full(
        nplanes,
        np.nan,
        dtype=float,
    )

    objective_errors = np.full(
        nplanes,
        np.nan,
        dtype=float,
    )

    forward_errors = np.full(
        nplanes,
        np.nan,
        dtype=float,
    )

    roughness_values = np.full(
        nplanes,
        np.nan,
        dtype=float,
    )

    sigma_measured_from_pdf = np.full(
        nplanes,
        np.nan,
        dtype=float,
    )

    sigma_kernel_values = np.full(
        nplanes,
        np.nan,
        dtype=float,
    )

    sigma_deconvolved = np.full(
        nplanes,
        np.nan,
        dtype=float,
    )

    sigma_reconvolved = np.full(
        nplanes,
        np.nan,
        dtype=float,
    )

    variance_measured = np.full(
        nplanes,
        np.nan,
        dtype=float,
    )

    variance_kernel = np.full(
        nplanes,
        np.nan,
        dtype=float,
    )

    variance_intrinsic_quadrature = np.full(
        nplanes,
        np.nan,
        dtype=float,
    )

    var_space_point_values = np.full(
        nplanes,
        np.nan,
        dtype=float,
    )

    var_space_air_values = np.full(
        nplanes,
        np.nan,
        dtype=float,
    )

    var_angle_point_values = np.full(
        nplanes,
        np.nan,
        dtype=float,
    )

    var_angle_air_values = np.full(
        nplanes,
        np.nan,
        dtype=float,
    )

    negative_fraction_raw = np.full(
        nplanes,
        np.nan,
        dtype=float,
    )

    pdf_measured_values[0] = normalize_pdf(
        pdf_meas_space[0],
        xbin,
    )

    pdf_space[0] = pdf_measured_values[0].copy()

    # At source there is no scattering kernel to remove:
    # use a delta kernel centered at x=0.
    pdf_kernel_values[0] = _gaussian_pdf_or_delta(
        xbin,
        0.0,
    )

    pdf_reconvolved_values[0] = pdf_space[0].copy()

    sigmas_angle[0] = float(sigma_meas_angle[0])

    sigma_measured_from_pdf[0] = measure_sigma(
        xbin,
        pdf_measured_values[0],
        method=sigma_method,
        fit_window=fit_window,
    )

    sigma_deconvolved[0] = measure_sigma(
        xbin,
        pdf_space[0],
        method=sigma_method,
        fit_window=fit_window,
    )

    sigma_reconvolved[0] = sigma_deconvolved[0]

    variance_measured[0] = (
        sigma_measured_from_pdf[0] ** 2
    )

    variance_kernel[0] = 0.0
    variance_intrinsic_quadrature[0] = (
        variance_measured[0]
    )

    for i in range(1, nplanes):
        z_current = float(Zout[i])

        # Scatterers located exactly on the current scoring plane do not
        # contribute to spatial broadening because distance = 0 anyway.
        # <= is clearer and physically includes their angular contribution.
        active = z_scatt <= z_current

        z_active = z_scatt[active]
        sigma_active = sigma_scatt_angle[active]

        distances_mm = z_current - z_active

        var_angle_point = float(
            np.sum(sigma_active**2)
        )

        var_space_point = float(
            np.sum(
                (
                    sigma_active
                    * 1.0e-3
                    * distances_mm
                ) ** 2
            )
        )

        (
            var_space_air,
            _cov_space_angle_air,
            var_angle_air,
        ) = _distributed_scattering_moments(
            z_current,
            air_segments,
        )

        var_space_air = float(var_space_air)
        var_angle_air = float(var_angle_air)

        var_space_kernel = max(
            var_space_point + var_space_air,
            0.0,
        )
        var_space_point_values[i] = var_space_point
        var_space_air_values[i] = var_space_air
        var_angle_point_values[i] = var_angle_point
        var_angle_air_values[i] = var_angle_air

        sigma_spatial_kernel = float(
            np.sqrt(var_space_kernel)
        )

        var_angle_clean = max(
            sigma_meas_angle[i] ** 2
            - var_angle_point
            - var_angle_air,
            0.0,
        )

        sigmas_angle[i] = float(
            np.sqrt(var_angle_clean)
        )

        pdf_measured = normalize_pdf(
            pdf_meas_space[i],
            xbin,
        )

        pdf_scattering_kernel = _gaussian_pdf_or_delta(
            xbin,
            sigma_spatial_kernel,
        )

        # Store every object used in the actual deconvolution.
        pdf_measured_values[i] = pdf_measured
        pdf_kernel_values[i] = pdf_scattering_kernel

        sigma_measured_pdf_i = measure_sigma(
            xbin,
            pdf_measured,
            method=sigma_method,
            fit_window=fit_window,
        )

        sigma_kernel_pdf_i = measure_sigma(
            xbin,
            pdf_scattering_kernel,
            method="fit",
            fit_window=None,
        )

        sigma_measured_from_pdf[i] = sigma_measured_pdf_i
        sigma_kernel_values[i] = sigma_kernel_pdf_i

        variance_measured[i] = sigma_measured_pdf_i**2
        variance_kernel[i] = sigma_kernel_pdf_i**2

        intrinsic_variance_i = max(
            variance_measured[i] - variance_kernel[i],
            0.0,
        )

        variance_intrinsic_quadrature[i] = (
            intrinsic_variance_i
        )

        # IMPORTANT:
        # If lambd is None, automatic lambda selection is enabled.
        # If lambd is supplied, exactly that lambda is used.
        #
        # Keep the keyword names aligned with YOUR deconvolve_wiener
        # implementation. The expected convention is auto_lambda,
        # lambda_min, lambda_max, roughness_weight, return_info.
        deconv_kwargs = {
            "signal": pdf_measured,
            "kernel": pdf_scattering_kernel,
            "xgrid": xbin,
            "lambd": lambd,
            "auto_lambda": lambd is None,
            "lambda_min": lambda_min,
            "lambda_max": lambda_max,
            "roughness_weight": roughness_weight,
            "return_info": True,
            "verbose": verbose,
        }

        # Only pass roughness_weight if the current implementation accepts it.
        # If your deconvolve_wiener does not expose this parameter, remove this
        # block and keep the rest of the function unchanged.
        pdf_raw, optimization = deconvolve_wiener(
            signal=pdf_measured,
            kernel=pdf_scattering_kernel,
            xgrid=xbin,
            lambd=lambd,
            auto_lambda=(lambd is None),
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            roughness_weight=roughness_weight,
            return_info=True,
            verbose=verbose,
        )

        # pdf_raw restituita da deconvolve_wiener è già clippata.
        pdf_raw = np.asarray(pdf_raw, dtype=float)
        pdf_raw = np.where(np.isfinite(pdf_raw), pdf_raw, 0.0)

        # Ricalcola la soluzione con la lambda selezionata,
        # conservando anche il risultato precedente al clipping.
        selected_lambda = float(optimization["lambda_opt"])

        _, raw_info = wiener_deconvolution_fixed_lambda(
            signal=pdf_measured,
            kernel=pdf_scattering_kernel,
            xgrid=xbin,
            lambd=selected_lambda,
            return_raw=True,
        )

        signed_pdf = raw_info["raw_pdf"]

        negative_area = float(raw_info["negative_mass"])
        absolute_area = float(
            np.trapezoid(np.abs(signed_pdf), xbin)
        )

        negative_fraction_raw[i] = (
            negative_area / max(absolute_area, 1.0e-30)
        )

        if verbose:
            signed_mean = np.trapezoid(
                xbin * signed_pdf,
                xbin,
            )

            signed_second_moment = np.trapezoid(
                (xbin - signed_mean) ** 2 * signed_pdf,
                xbin,
            )

            print(
                f"Massa negativa Wiener prima del clipping = "
                f"{negative_area:.8e}"
            )
            print(
                f"Secondo momento Wiener con segno = "
                f"{signed_second_moment:.8f} mm²"
            )

        # A physical transverse PDF must be non-negative and normalized.
        pdf_physical = normalize_pdf(
            np.clip(pdf_raw, 0.0, None),
            xbin,
        )

        # This is the decisive validation:
        # recovered PDF (*) same kernel must reproduce measured PDF.
        pdf_reconvolved = reconvolve_same_grid(
            pdf_physical,
            pdf_scattering_kernel,
            xbin,
        )

        forward_error_direct = relative_forward_error(
            measured_pdf=pdf_measured,
            reconvolved_pdf=pdf_reconvolved,
            xgrid=xbin,
        )

        if roughness_weight > 0.0:
            roughness_direct = relative_curvature_penalty(
                pdf_physical,
                xbin,
            )
        else:
            roughness_direct = 0.0

        # Accept variants from old/new optimizer dictionaries, but do not rely
        # on them for the actual forward error used in diagnostics.
        if optimization is None:
            optimization = {}

        lambda_opt = float(
            optimization.get(
                "lambda_opt",
                lambd if lambd is not None else np.nan,
            )
        )

        objective_error = float(
            optimization.get(
                "error_opt",
                forward_error_direct
                + roughness_weight * roughness_direct,
            )
        )

        sigma_out = measure_sigma(
            xbin,
            pdf_physical,
            method=sigma_method,
            fit_window=fit_window,
        )

        sigma_moment_check = measure_sigma(
            xbin,
            pdf_physical,
            method="fit",
            fit_window=None,
        )

        sigma_fit_check = fit_sigma_from_sampled_pdf(
            xbin,
            pdf_physical,
        )

        if verbose:
            print("\n--- SIGMA DELLA STESSA PDF DECONVOLUTA ---")
            print(f"z = {z_current:.1f} mm")
            print(
                f"Sigma da momento = "
                f"{sigma_moment_check:.8f} mm"
            )
            print(
                f"Sigma da fit gaussiano = "
                f"{sigma_fit_check:.8f} mm"
            )

        sigma_reconv_out = measure_sigma(
            xbin,
            pdf_reconvolved,
            method=sigma_method,
            fit_window=fit_window,
        )

        pdf_space[i] = pdf_physical
        pdf_reconvolved_values[i] = pdf_reconvolved

        lambda_values[i] = lambda_opt
        objective_errors[i] = objective_error
        forward_errors[i] = forward_error_direct
        roughness_values[i] = roughness_direct
        sigma_deconvolved[i] = sigma_out
        sigma_reconvolved[i] = sigma_reconv_out

        at_lower_bound = (
            np.isfinite(lambda_opt)
            and np.isclose(
                lambda_opt,
                lambda_min,
                rtol=1.0e-3,
                atol=0.0,
            )
        )

        at_upper_bound = (
            np.isfinite(lambda_opt)
            and np.isclose(
                lambda_opt,
                lambda_max,
                rtol=1.0e-3,
                atol=0.0,
            )
        )

        if verbose:
            sigma_quadrature = float(
                np.sqrt(
                    variance_intrinsic_quadrature[i]
                )
            )

            print(
                "\n"
                + "=" * 72
            )

            print(
                f"DECONVOLUTION DIAGNOSTIC: "
                f"z = {z_current:.1f} mm"
            )

            print(
                "=" * 72
            )

            print(
                f"Grid dx = {dx:.8f} mm"
            )

            print(
                f"Active point scatterers = "
                f"{z_active.size}"
            )

            print(
                f"lambda = {lambda_opt:.8e}"
            )

            print(
                f"objective error = "
                f"{objective_error:.8e}"
            )

            print(
                f"forward reconvolution error = "
                f"{forward_error_direct:.8e}"
            )

            print(
                f"roughness penalty = "
                f"{roughness_direct:.8e}"
            )

            print(
                f"negative raw-PDF fraction = "
                f"{negative_fraction_raw[i]:.8e}"
            )

            print(
                f"sigma measured array = "
                f"{sigma_meas_space[i]:.6f} mm"
            )

            print(
                f"sigma measured PDF = "
                f"{sigma_measured_pdf_i:.6f} mm"
            )

            print(
                f"sigma scattering kernel = "
                f"{sigma_kernel_pdf_i:.6f} mm"
            )

            print(
                f"sigma intrinsic "
                f"quadrature expectation = "
                f"{sigma_quadrature:.6f} mm"
            )

            print(
                f"sigma deconvolved = "
                f"{sigma_out:.6f} mm"
            )

            print(
                f"sigma reconvolved = "
                f"{sigma_reconv_out:.6f} mm"
            )

            print(
                f"lambda at lower bound = "
                f"{at_lower_bound}"
            )

            print(
                f"lambda at upper bound = "
                f"{at_upper_bound}"
            )

        if (
            np.isfinite(sigma_out)
            and sigma_out > 1.01 * sigma_measured_pdf_i
        ):
            print(
                "WARNING: deconvolved sigma is larger "
                "than the measured-PDF sigma. Check kernel, "
                "grid, shifts, or regularization."
            )

        if negative_fraction_raw[i] > 1.0e-3:
            print(
                "WARNING: the raw Wiener result contains a "
                "non-negligible negative component before clipping. "
                "The recovered PDF may be unstable."
            )

        if forward_error_direct > 5.0e-2:
            print(
                "WARNING: reconvolved recovered PDF does not "
                "reproduce the measured PDF closely. Inspect "
                "measured/kernel/deconvolved/reconvolved PDFs."
            )

        if at_lower_bound or at_upper_bound:
            print(
                "WARNING: lambda reached a search boundary. "
                "The automatic optimum may not be reliable."
            )

    diagnostics = {
        "zout_mm": Zout.copy(),
        "dx_mm": dx,
        "lambda_values": lambda_values.copy(),
        "objective_errors": objective_errors.copy(),
        "forward_errors": forward_errors.copy(),
        "roughness_values": roughness_values.copy(),
        "sigma_measured_array_mm": sigma_meas_space.copy(),
        "sigma_measured_from_pdf_mm": (
            sigma_measured_from_pdf.copy()
        ),
        "sigma_kernel_mm": sigma_kernel_values.copy(),
        "sigma_intrinsic_quadrature_mm": np.sqrt(
            np.maximum(
                variance_intrinsic_quadrature,
                0.0,
            )
        ),
        "sigma_deconvolved_mm": sigma_deconvolved.copy(),
        "sigma_reconvolved_mm": sigma_reconvolved.copy(),
        "sigmas_angle_clean_mrad": sigmas_angle.copy(),
        "var_space_point_mm2": var_space_point_values.copy(),
        "var_space_air_mm2": var_space_air_values.copy(),
        "var_angle_point_mrad2": var_angle_point_values.copy(),
        "var_angle_air_mrad2": var_angle_air_values.copy(),
        "negative_fraction_raw": negative_fraction_raw.copy(),
        "pdf_measured": pdf_measured_values.copy(),
        "pdf_kernels": pdf_kernel_values.copy(),
        "pdf_deconvolved": pdf_space.copy(),
        "pdf_reconvolved": pdf_reconvolved_values.copy(),
    }

    if verbose:
        print(
            "\nSelected lambdas "
            "(source then scoring surfaces):"
        )
        print(lambda_values)

        print(
            "\nForward reconvolution errors "
            "(source then scoring surfaces):"
        )
        print(forward_errors)

        print(
            "\nKernel sigmas [mm] "
            "(source then scoring surfaces):"
        )
        print(sigma_kernel_values)

        print(
            "\nQuadrature expectations [mm] "
            "(source then scoring surfaces):"
        )
        print(
            diagnostics[
                "sigma_intrinsic_quadrature_mm"
            ]
        )

        print(
            "\nDeconvolved sigmas [mm] "
            "(source then scoring surfaces):"
        )
        print(sigma_deconvolved)

    return (
        Zout,
        sigmas_angle,
        pdf_space,
        diagnostics,
    )
def build_measured_space_pdfs(xbin, sigma_meas_space):
    """
    Costruisce PDF gaussiane normalizzate dalle sigma misurate.
    La sequenza delle sigma deve comprendere:
    sorgente + superfici di scoring.
    """
    xbin = np.asarray(xbin, dtype=float)
    sigmas = np.asarray(sigma_meas_space, dtype=float)

    if (
        xbin.ndim != 1
        or xbin.size < 3
        or not np.all(np.isfinite(xbin))
        or np.any(np.diff(xbin) <= 0.0)
    ):
        raise ValueError("xbin deve essere finita e crescente.")

    if (
        sigmas.ndim != 1
        or not np.all(np.isfinite(sigmas))
        or np.any(sigmas < 0.0)
    ):
        raise ValueError(
            "Le sigma devono essere finite e non negative."
        )

    pdfs = []

    for sigma in sigmas:
        if sigma == 0.0:
            pdf = np.zeros_like(xbin)
            pdf[np.argmin(np.abs(xbin))] = 1.0
        else:
            pdf = np.exp(
                -0.5 * (xbin / sigma) ** 2
            )

        pdfs.append(
            normalize_pdf(pdf, xbin)
        )

    return pdfs



def run_analysis(
    results_masked_vacuum,
    results_masked_air,
    df_original_vacuum,
    stats_angle,
    xbin,
    xpbin,
    z0,
    sigma_0_angle,
    sigma_0_space,
    sigma_0_theta,
    z_nozzle_start,
    z_nozzle_end,
    ripple_filters=False,
):

    positions_scoring_mm = np.array([-960.0, -600.0, 0.0, 1000.0])

    def _prepend_source(results):
        sigma_xp = np.asarray([r["sigma_xp"] for r in results], dtype=float)
        sigma_yp = np.asarray([r["sigma_yp"] for r in results], dtype=float)
        sigma_x = np.asarray([r["sigma_x"] for r in results], dtype=float)
        sigma_y = np.asarray([r["sigma_y"] for r in results], dtype=float)
        sigma_theta = np.asarray([r["sigma_theta"] for r in results], dtype=float)
        return (
            np.insert(sigma_xp, 0, sigma_0_angle),
            np.insert(sigma_yp, 0, sigma_0_angle),
            np.insert(sigma_x, 0, sigma_0_space),
            np.insert(sigma_y, 0, sigma_0_space),
            np.insert(sigma_theta, 0, sigma_0_theta),
        )

    def pdf_from_surface(df, column, grid):
        grid = np.asarray(grid, dtype=float)

        if grid.ndim != 1 or grid.size < 3:
            raise ValueError(
                "grid deve essere un array 1D con almeno tre punti."
            )

        steps = np.diff(grid)
        dx = float(np.mean(steps))

        if (
            dx <= 0.0
            or not np.allclose(
                steps,
                dx,
                rtol=1.0e-8,
                atol=1.0e-12,
            )
        ):
            raise ValueError(
                "grid deve essere uniforme e crescente."
            )

        values = np.asarray(
            get_clean_data(df, column),
            dtype=float,
        )
        values = values[np.isfinite(values)]

        if values.size == 0:
            raise ValueError(
                f"Nessun valore finito per {column!r}."
            )

        # Ogni punto di grid è il centro di un bin.
        edges = np.concatenate((
            [grid[0] - 0.5 * dx],
            grid + 0.5 * dx,
        ))

        counts, _ = np.histogram(
            values,
            bins=edges,
            density=False,
        )

        n_inside = int(np.sum(counts))

        if n_inside == 0:
            raise ValueError(
                f"Nessun valore di {column!r} dentro la griglia."
            )

        pdf = normalize_pdf(
            counts.astype(float) / dx,
            grid,
        )

        mean_pdf = np.trapezoid(
            grid * pdf,
            grid,
        )

        variance_pdf = np.trapezoid(
            (grid - mean_pdf)**2 * pdf,
            grid,
        )

        values_inside = values[
            (values >= edges[0])
            & (values <= edges[-1])
        ]

        print(
            f"[PDF {column}] "
            f"N={values.size}, "
            f"N dentro={n_inside}, "
            f"fuori={100.0 * (1.0 - n_inside / values.size):.6f}%, "
            f"dx={dx:.8f}, "
            f"sigma dati dentro={np.std(values_inside):.6f} mm, "
            f"sigma PDF={np.sqrt(max(variance_pdf, 0.0)):.6f} mm"
        )

        return pdf

    (
        sigma_meas_xp_vacuum,
        sigma_meas_yp_vacuum,
        sigma_meas_x_vacuum,
        sigma_meas_y_vacuum,
        sigma_meas_theta_vacuum,
    ) = _prepend_source(results_masked_vacuum)

    (
        sigma_meas_xp_air,
        sigma_meas_yp_air,
        sigma_meas_x_air,
        sigma_meas_y_air,
        sigma_meas_theta_air,
    ) = _prepend_source(results_masked_air)

    sigma_window_attr = {
        "xp": float(stats_angle["xp"]["window_angle_std"]),
        "yp": float(stats_angle["yp"]["window_angle_std"]),
    }
    sigma_nozzle_attr = {
        "xp": float(
            stats_angle["xp"]["nozzle_angle_std"]
            if not ripple_filters
            else stats_angle["xp"]["nozzle_ripple_angle_std"]
        ),
        "yp": float(
            stats_angle["yp"]["nozzle_angle_std"]
            if not ripple_filters
            else stats_angle["yp"]["nozzle_ripple_angle_std"]
        ),
    }

    z_scatt = np.array([-1093.0], dtype=float)
    sigma_scatt_attr = {
        "xp": np.array([sigma_window_attr["xp"]], dtype=float),
        "yp": np.array([sigma_window_attr["yp"]], dtype=float),
    }
    distributed_segments = {
        "xp": [
            _build_distributed_segment_from_total_sigma(
                z_start=z_nozzle_start,
                z_end=z_nozzle_end,
                sigma_total=sigma_nozzle_attr["xp"],
            )
        ],
        "yp": [
            _build_distributed_segment_from_total_sigma(
                z_start=z_nozzle_start,
                z_end=z_nozzle_end,
                sigma_total=sigma_nozzle_attr["yp"],
            )
        ],
    }

    Zout_prop, sigmas_space_prop, sigmas_angle_prop = {}, {}, {}
    Zout_conv_final, sigmas_angle_conv_final, pdf_space_conv_final = {}, {}, {}
    Z_ss, sigmas_space_ss, sigmas_theta_ss, cov_ss = {}, {}, {}, {}

    for attr in ("xp", "yp"):
        sigma_scatt_angle = sigma_scatt_attr[attr]
        (
            Zout_prop[attr],
            sigmas_space_prop[attr],
            sigmas_angle_prop[attr],
        ) = sumQuadProp_with_air(
            z_scatt=z_scatt,
            sigma_scatt_angle=sigma_scatt_angle,
            sigma0_space=sigma_0_space,
            sigma0_angle=sigma_0_angle,
            z0=z0,
            z_scorings=positions_scoring_mm,
            air_segments=distributed_segments[attr]
        )
        (
            Zout_conv_final[attr],
            sigmas_angle_conv_final[attr],
            pdf_space_conv_final[attr],
        ) = ConvFinal_with_air(
            z_scatt=z_scatt,
            sigma_scatt_angle=sigma_scatt_angle,
            sigma0_space=sigma_0_space,
            sigma0_angle=sigma_0_angle,
            z0=z0,
            z_scorings=positions_scoring_mm,
            xbin=xbin,
            air_segments=distributed_segments[attr],
        )
        (
            Z_ss[attr],
            sigmas_space_ss[attr],
            sigmas_theta_ss[attr],
            cov_ss[attr],
        ) = state_space_propagation_with_air(
            z_scatt=z_scatt,
            sigma_scatt_angle=sigma_scatt_angle,
            sigma0_space=sigma_0_space,
            sigma0_angle=sigma_0_angle,
            z0=z0,
            z_scorings=positions_scoring_mm,
            air_segments = distributed_segments[attr],
        )

    Zout_dq_prop, sigmas_space_dq_prop, sigmas_angle_dq_prop = {}, {}, {}
    Zout_deconvolution, sigmas_angle_deconvolution, pdf_space_deconvolution, deconvolution_diagnostics = {}, {}, {}, {}
    Zout_ss_inv, sigmas_space_ss_inv, sigmas_angle_ss_inv, covariance_inverse = {}, {}, {}, {}

    for attr in ("xp", "yp"):
        sigma_scatt_angle = sigma_scatt_attr[attr]
        sigma_angle_measured = (
            sigma_meas_xp_vacuum if attr == "xp" else sigma_meas_yp_vacuum
        )
        sigma_space_measured = (
            sigma_meas_x_vacuum if attr == "xp" else sigma_meas_y_vacuum
        )
        space_column = "x - coordinate" if attr == "xp" else "y - coordinate"

        (
            Zout_ss_inv[attr],
            sigmas_space_ss_inv[attr],
            sigmas_angle_ss_inv[attr],
            covariance_inverse[attr],
        ) = state_space_diffquad_prop_with_air(
            z_scatt=z_scatt,
            sigma_scatt_angle=sigma_scatt_angle,
            z0=z0,
            z_scorings=positions_scoring_mm,
            sigmas_angle_measured=sigma_angle_measured,
            sigmas_space_measured=sigma_space_measured,
            covariance_measured=None,
            air_segments=distributed_segments[attr]
        )
        (
            Zout_dq_prop[attr],
            sigmas_space_dq_prop[attr],
            sigmas_angle_dq_prop[attr],
        ) = diffQuad_prop_with_air(
            z_scatt=z_scatt,
            sigma_scatt_angle=sigma_scatt_angle,
            z0=z0,
            z_scorings=positions_scoring_mm,
            sigmas_angle_measured=sigma_angle_measured,
            sigmas_space_measured=sigma_space_measured,
            air_segments= distributed_segments[attr],

        )

        pdf_meas_space = build_measured_space_pdfs(
            xbin,
            sigma_space_measured,
        )
        expected_npdf = len(positions_scoring_mm) + 1
        if len(pdf_meas_space) != expected_npdf:
            raise RuntimeError(
                f"{attr}: received {len(pdf_meas_space)} PDFs; expected {expected_npdf}."
            )

        print(
            f"\n[{attr}] Deconvoluzione di PDF gaussiane: "
            f"{len(pdf_meas_space)} PDF "
            f"(sorgente + {len(positions_scoring_mm)} superfici)."
        )

        (
            Zout_deconvolution[attr],
            sigmas_angle_deconvolution[attr],
            pdf_space_deconvolution[attr],
            deconvolution_diagnostics[attr]
        ) = deconvolutionFinal_with_air(
            z_scatt=z_scatt,
            sigma_scatt_angle=sigma_scatt_angle,
            pdf_meas_space=pdf_meas_space,
            sigma_meas_space=sigma_space_measured,
            sigma_meas_angle=sigma_angle_measured,
            z0=z0,
            z_scorings=positions_scoring_mm,
            xbin=xbin,
            air_segments=distributed_segments[attr],
            lambd=None,
            verbose=True,
        )

    import inspect

    print("\n" + "=" * 80)
    print("ACTIVE WIENER FUNCTIONS")
    print("=" * 80)

    print("\n--- wiener_deconvolution_fixed_lambda ---")
    print(inspect.getsource(wiener_deconvolution_fixed_lambda))

    print("\n--- deconvolve_wiener ---")
    print(inspect.getsource(deconvolve_wiener))

    print("\n--- find_optimal_lambda_forward ---")
    print(inspect.getsource(find_optimal_lambda_forward))

    print("\n--- reconvolve_same_grid ---")
    print(inspect.getsource(reconvolve_same_grid))

    print("\n--- deconvolutionFinal_with_air ---")
    print(inspect.getsource(deconvolutionFinal_with_air))


    labels_surfaces = [
        f"{z0 / 10.0:.1f} cm",
        *[f"{z / 10.0:.1f} cm" for z in positions_scoring_mm],
    ]

    Zout_deconvolution_xp = Zout_deconvolution["xp"]
    Zout_deconvolution_yp = Zout_deconvolution["yp"]
    prop_xp_angle = sigmas_angle_prop["xp"]
    prop_yp_angle = sigmas_angle_prop["yp"]
    prop_x_space = sigmas_space_prop["xp"]
    prop_y_space = sigmas_space_prop["yp"]
    ss_x = sigmas_space_ss["xp"]
    ss_y = sigmas_space_ss["yp"]
    ss_xp = sigmas_theta_ss["xp"]
    ss_yp = sigmas_theta_ss["yp"]

    conv_final_x = np.array([
        fit_sigma_from_sampled_pdf(xbin, pdf)
        for pdf in pdf_space_conv_final["xp"]
    ])
    conv_final_y = np.array([
        fit_sigma_from_sampled_pdf(xbin, pdf)
        for pdf in pdf_space_conv_final["yp"]
    ])
    conv_final_xp = sigmas_angle_conv_final["xp"]
    conv_final_yp = sigmas_angle_conv_final["yp"]

    dq_prop_xp_angle = sigmas_angle_dq_prop["xp"]
    dq_prop_yp_angle = sigmas_angle_dq_prop["yp"]
    dq_prop_x_space = sigmas_space_dq_prop["xp"]
    dq_prop_y_space = sigmas_space_dq_prop["yp"]

    deconvolution_diagnostics_x = deconvolution_diagnostics["xp"]
    deconvolution_diagnostics_y = deconvolution_diagnostics["yp"]

    deconvolution_final_x = np.asarray(
        deconvolution_diagnostics_x["sigma_deconvolved_mm"],
        dtype=float,
    )

    deconvolution_final_y = np.asarray(
        deconvolution_diagnostics_y["sigma_deconvolved_mm"],
        dtype=float,
    )

    print("\n[DECONVOLUTION SIGMAS ACTUALLY USED FOR PLOT — X]")
    for z_value, sigma_value in zip(
        Zout_deconvolution_xp,
        deconvolution_final_x
    ):
        print(f"z = {z_value:9.3f} mm | sigma_plot = {sigma_value:12.6f} mm")

    print("\n[DECONVOLUTION SIGMAS ACTUALLY USED FOR PLOT — Y]")
    for z_value, sigma_value in zip(
        Zout_deconvolution_yp,
        deconvolution_final_y
    ):
        print(f"z = {z_value:9.3f} mm | sigma_plot = {sigma_value:12.6f} mm")


    deconvolution_final_xp = sigmas_angle_deconvolution["xp"]
    deconvolution_final_yp = sigmas_angle_deconvolution["yp"]
    ss_inverse_x = sigmas_space_ss_inv["xp"]
    ss_inverse_y = sigmas_space_ss_inv["yp"]

    for i, z_surf in enumerate(positions_scoring_mm):
        print(f"\n=== Surface loop i={i}, z_surf={z_surf} mm ===")
        z_surf_cm = z_surf / 10.0
        surface_df = df_original_vacuum.loc[
            np.isclose(
                df_original_vacuum["scoring_surface"].to_numpy(dtype=float),
                z_surf_cm,
                rtol=0.0,
                atol=1e-9,
            )
        ].copy()
        data_dict = {
            "xp": clean_data(surface_df, "x_prime"),
            "yp": clean_data(surface_df, "y_prime"),
            "x": clean_data(surface_df, "x - coordinate"),
            "y": clean_data(surface_df, "y - coordinate"),
        }
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.ravel()
        idx = i + 1
        configs = [
            ("xp", "x' [mrad]", xpbin, prop_xp_angle, ss_xp, conv_final_xp),
            ("yp", "y' [mrad]", xpbin, prop_yp_angle, ss_yp, conv_final_yp),
            ("x", "x [mm]", xbin, prop_x_space, ss_x, conv_final_x),
            ("y", "y [mm]", xbin, prop_y_space, ss_y, conv_final_y),
        ]
        for ax_idx, (key, unit, grid_coord, prop_list, ss_list, conv_list) in enumerate(configs):
            ax = axes[ax_idx]
            current_data = data_dict[key]
            hist_vals, bin_edges = np.histogram(current_data, bins=grid_coord, density=False)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_width = float(bin_edges[1] - bin_edges[0])
            n_total = len(current_data)
            mu_plot = 0.0 if key in ("xp", "yp") else float(np.mean(current_data))
            sigma_prop = float(prop_list[idx])
            sigma_ss = float(ss_list[idx])
            sigma_conv = float(conv_list[idx])
            ax.bar(bin_centers, hist_vals, width=bin_width, alpha=0.25, color="grey", edgecolor="none", label=f"Data (original) {key}")
            ax.step(bin_centers, gaussian_model_counts(bin_centers, n_total, mu_plot, sigma_prop, 0.0, bin_width), where="mid", color="g", lw=1.6, linestyle="--", label=f"Sigma Prop: {sigma_prop:.3f}")
            ax.step(bin_centers, gaussian_model_counts(bin_centers, n_total, mu_plot, sigma_ss, 0.0, bin_width), where="mid", color="orange", lw=1.8, linestyle="--", label=f"State Space sigma: {sigma_ss:.3f}")
            ax.step(bin_centers, gaussian_model_counts(bin_centers, n_total, mu_plot, sigma_conv, 0.0, bin_width), where="mid", color="purple", lw=1.8, linestyle="--", label=f"ConvFinal sigma: {sigma_conv:.3f}")
            ax.set_title(f"{key} at {z_surf_cm:.1f} cm")
            ax.set_xlabel(unit)
            ax.set_ylabel("Counts")
            ax.legend(fontsize="x-small", loc="upper right")
            ax.grid(alpha=0.2)
        plt.suptitle(f"Surface z = {z_surf:.1f} mm (Original data)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()

    window_angle_std_xp = stats_angle["xp"]["window_angle_std"]
    window_angle_std_yp = stats_angle["yp"]["window_angle_std"]
    window_theta_std_xp = stats_angle["xp"]["window_theta_std"]
    window_theta_std_yp = stats_angle["yp"]["window_theta_std"]
    window_angle_variation_std_xp = stats_angle["xp"]["window_angle_variation_std"]
    window_angle_variation_std_yp = stats_angle["yp"]["window_angle_variation_std"]

    if not ripple_filters:
        nozzle_label = "Nozzle"
        nozzle_theta_std_xp = stats_angle["xp"]["nozzle_theta_std"]
        nozzle_theta_std_yp = stats_angle["yp"]["nozzle_theta_std"]
        nozzle_angle_variation_std_xp = stats_angle["xp"]["nozzle_angle_variation_std"]
        nozzle_angle_variation_std_yp = stats_angle["yp"]["nozzle_angle_variation_std"]
    else:
        nozzle_label = "Nozzle and Ripple filters"
        nozzle_theta_std_xp = stats_angle["xp"]["nozzle_ripple_theta_std"]
        nozzle_theta_std_yp = stats_angle["yp"]["nozzle_ripple_theta_std"]
        nozzle_angle_variation_std_xp = stats_angle["xp"]["nozzle_ripple_angle_variation_std"]
        nozzle_angle_variation_std_yp = stats_angle["yp"]["nozzle_ripple_angle_variation_std"]

    scatterers_comparison = pd.DataFrame({
        ("", "Scatterers"): ["Vacuum Window", nozzle_label],
        ("FROM INTEGRAL DATASET (variation of STDs)", "In x' (mrad)"): [window_angle_std_xp, sigma_nozzle_attr["xp"]],
        ("FROM INTEGRAL DATASET (variation of STDs)", "In y' (mrad)"): [window_angle_std_yp, sigma_nozzle_attr["yp"]],
        ("FROM INTEGRAL DATASET (variation of STDs)", "In theta (mrad)"): [window_theta_std_xp, nozzle_theta_std_xp],
        ("FROM DIFFERENTIAL DATASET (STD of variations)", "In x' (mrad)"): [window_angle_variation_std_xp, nozzle_angle_variation_std_xp],
        ("FROM DIFFERENTIAL DATASET (STD of variations)", "In y' (mrad)"): [window_angle_variation_std_yp, nozzle_angle_variation_std_yp],
    })
    scatterers_comparison.columns = pd.MultiIndex.from_tuples(scatterers_comparison.columns)

    direct_methods_comparison = pd.DataFrame({
        ("", "Scoring surfaces"): labels_surfaces,
        ("FROM INTEGRAL DATASET", "STD in x (mm)"): sigma_meas_x_vacuum,
        ("FROM INTEGRAL DATASET", "STD in y (mm)"): sigma_meas_y_vacuum,
        ("FROM SUM OF SQUARES", "STD in x (mm)"): prop_x_space,
        ("FROM SUM OF SQUARES", "STD in y (mm)"): prop_y_space,
        ("FROM CONVOLUTION", "STD in x (mm)"): conv_final_x,
        ("FROM CONVOLUTION", "STD in y (mm)"): conv_final_y,
        ("FROM STATE SPACE", "STD in x (mm)"): ss_x,
        ("FROM STATE SPACE", "STD in y (mm)"): ss_y,
    })
    direct_methods_comparison.columns = pd.MultiIndex.from_tuples(direct_methods_comparison.columns)

    inverse_methods_comparison = pd.DataFrame({
        ("", "Scoring surfaces"): labels_surfaces,
        ("FROM DIFFERENCE OF SQUARES", "STD in x (mm)"): dq_prop_x_space,
        ("FROM DIFFERENCE OF SQUARES", "STD in y (mm)"): dq_prop_y_space,
        ("FROM DECONVOLUTION", "STD in x (mm)"): deconvolution_final_x,
        ("FROM DECONVOLUTION", "STD in y (mm)"): deconvolution_final_y,
        ("FROM STATE SPACE", "STD in x (mm)"): ss_inverse_x,
        ("FROM STATE SPACE", "STD in y (mm)"): ss_inverse_y,
    })
    inverse_methods_comparison.columns = pd.MultiIndex.from_tuples(inverse_methods_comparison.columns)

    return (
        scatterers_comparison,
        direct_methods_comparison,
        inverse_methods_comparison,
        sigma_meas_xp_vacuum,
        sigma_meas_yp_vacuum,
        sigma_meas_x_vacuum,
        sigma_meas_y_vacuum,
        sigma_meas_theta_vacuum,
        sigma_meas_xp_air,
        sigma_meas_yp_air,
        sigma_meas_x_air,
        sigma_meas_y_air,
        sigma_meas_theta_air,
    )


def calc_equivalent_x(
    angle_xp_mrad,
    angle_yp_mrad,
    angle_theta_mrad,
    pc,
    beta,
    z,
    X0,
    delta_energy,
    K,
    Z_A_water,
    rest_energy_electron,
    I,
    j,
    gamma,
    n_iter=8,
    return_history=False,
    landau_output_unit="mm" 
):
    angle_xp_mrad = float(angle_xp_mrad)
    angle_yp_mrad = float(angle_yp_mrad)
    angle_theta_mrad = float(angle_theta_mrad)
    pc = float(pc)
    beta = float(beta)
    z = float(z)
    X0 = float(X0)
    delta_energy = float(delta_energy)
    K = float(K)
    Z_A_water = float(Z_A_water)
    rest_energy_electron = float(rest_energy_electron)
    I = float(I)
    j = float(j)
    gamma = float(gamma)


    theta_xp = angle_xp_mrad * 1e-3
    theta_yp = angle_yp_mrad * 1e-3
    theta = angle_theta_mrad * 1e-3

    def solve_equivalent_x_mcs(theta_rad):
        if theta_rad < 0:
            theta_rad = abs(theta_rad)

        x_eq = ((theta_rad * beta * pc) / (13.6 * abs(z)))**2 * X0
        history = [x_eq]

        for _ in range(int(n_iter)):
            arg = max((x_eq * z**2) / (X0 * beta**2), 1e-30)
            corr = 1.0 + 0.038 * np.log(arg)

            if corr <= 1e-12:
                corr = 1e-12

            x_eq = ((theta_rad * beta * pc) / (13.6 * abs(z) * corr))**2 * X0
            history.append(x_eq)

        return x_eq, history

    equivalent_x_MCS_xp, history_MCS_xp = solve_equivalent_x_mcs(theta_xp)
    equivalent_x_MCS_yp, history_MCS_yp = solve_equivalent_x_mcs(theta_yp)
    equivalent_x_MCS_theta, history_MCS_theta = solve_equivalent_x_mcs(theta)

    csi = (K / 2.0) * Z_A_water * (z**2 / beta**2)

    term_const = (
        np.log((2.0 * rest_energy_electron * beta**2 * gamma**2) / I)
        + j
        - beta**2
    )

    x_landau_cm = delta_energy / (csi * term_const)
    history_Landau = [x_landau_cm]

    for _ in range(int(n_iter)):
        xi = csi * x_landau_cm
        log_arg = max(xi / I, 1e-30)

        denom = csi * (term_const + np.log(log_arg))
        if denom <= 1e-30:
            denom = 1e-30

        x_landau_cm = delta_energy / denom
        history_Landau.append(x_landau_cm)

    if landau_output_unit.lower() == "mm":
        x_equivalent_landau = x_landau_cm * 10.0
    elif landau_output_unit.lower() == "cm":
        x_equivalent_landau = x_landau_cm
    else:
        raise ValueError("landau_output_unit deve essere 'mm' oppure 'cm'")

    if return_history:
        return {
            "equivalent_x_MCS_xp": equivalent_x_MCS_xp,
            "equivalent_x_MCS_yp": equivalent_x_MCS_yp,
            "equivalent_x_MCS_theta": equivalent_x_MCS_theta,
            "x_equivalent_landau": x_equivalent_landau,
            "history_MCS_xp": history_MCS_xp,
            "history_MCS_yp": history_MCS_yp,
            "history_Landau": history_Landau,
            "x_landau_cm": x_landau_cm,
        }

    return equivalent_x_MCS_xp, equivalent_x_MCS_yp, equivalent_x_MCS_theta, x_equivalent_landau


def _build_air_segments(
    sigma_air_permm,
    positions_scoring_mm,
    z_window=-1093.0,
    z_air2_end=-925.5,
    z_air3_start=-705.5,
):

    z_scoring = np.asarray(positions_scoring_mm, dtype=float)
    if z_scoring.size != 4:
        raise ValueError("positions_scoring_mm must contain exactly four surfaces.")

    s0, s1, s2, s3 = z_scoring
    segments = [
        (z_window, s0, sigma_air_permm),
        (s0, z_air2_end, sigma_air_permm),
        (z_air3_start, s1, sigma_air_permm),
        (s1, s2, sigma_air_permm),
        (s2, s3, sigma_air_permm),
    ]

    for z_start, z_end, _ in segments:
        if z_end <= z_start:
            raise ValueError(
                f"Invalid air segment: z_end={z_end} must be greater than z_start={z_start}."
            )

    return segments




def _delta_pdf(xbin):
    pdf = np.zeros_like(xbin, dtype=float)
    pdf[np.argmin(np.abs(xbin))] = 1.0
    return normalize_pdf(pdf, xbin)


def _gaussian_pdf_or_delta(xbin, sigma):
    sigma = float(sigma)

    if sigma <= 1e-12:
        return _delta_pdf(xbin)

    return normalize_pdf(
        gaussian_model_counts(
            xbin,
            1.0,
            0.0,
            sigma,
            0.0,
            1.0,
        ),
        xbin,
    )


def sumQuadProp_with_air(
    z_scatt,
    sigma_scatt_angle,
    sigma0_space,
    sigma0_angle,
    z0,
    z_scorings,
    air_segments=None,
):

    z_scatt = np.asarray(z_scatt, dtype=float)
    sigma_scatt_angle = np.asarray(
        sigma_scatt_angle,
        dtype=float,
    )

    if z_scatt.shape != sigma_scatt_angle.shape:
        raise ValueError(
            "z_scatt and sigma_scatt_angle must have the same shape."
        )

    Zout = np.concatenate((
        [float(z0)],
        np.asarray(z_scorings, dtype=float),
    ))

    sigmas_space = np.zeros(len(Zout), dtype=float)
    sigmas_angle = np.zeros(len(Zout), dtype=float)

    sigmas_space[0] = float(sigma0_space)
    sigmas_angle[0] = float(sigma0_angle)

    for i in range(1, len(Zout)):
        z_current = Zout[i]

        # Source divergence propagated from z0 to z_current.
        var_space = sigma0_space**2 + (
            sigma0_angle * 1e-3 * (z_current - z0)
        )**2

        var_angle = sigma0_angle**2

        # Window and nozzle: point-like scattering contributions.
        for z_kick, sigma_kick in zip(
            z_scatt,
            sigma_scatt_angle,
        ):
            distance = z_current - z_kick

            if distance <= 0.0:
                continue

            var_angle += sigma_kick**2
            var_space += (
                sigma_kick * 1e-3 * distance
            )**2

        # Air: exact distributed-medium integrals.
        var_x_air, _, var_theta_air = _distributed_scattering_moments(
            z_current,
            air_segments,
        )

        var_space += var_x_air
        var_angle += var_theta_air

        sigmas_space[i] = np.sqrt(max(var_space, 0.0))
        sigmas_angle[i] = np.sqrt(max(var_angle, 0.0))

    return Zout, sigmas_space, sigmas_angle


def ConvFinal_with_air(
    z_scatt,
    sigma_scatt_angle,
    sigma0_space,
    sigma0_angle,
    z0,
    z_scorings,
    xbin,
    air_segments=None,
):
    z_scatt = np.asarray(z_scatt, dtype=float)
    sigma_scatt_angle = np.asarray(
        sigma_scatt_angle,
        dtype=float,
    )
    xbin = np.asarray(xbin, dtype=float)

    if z_scatt.shape != sigma_scatt_angle.shape:
        raise ValueError(
            "z_scatt and sigma_scatt_angle must have the same shape."
        )

    Zout = np.concatenate((
        [float(z0)],
        np.asarray(z_scorings, dtype=float),
    ))

    sigmas_angle = np.zeros(len(Zout), dtype=float)
    sigmas_angle[0] = float(sigma0_angle)

    pdf_space = np.zeros(
        (len(Zout), len(xbin)),
        dtype=float,
    )

    pdf_space_0 = _gaussian_pdf_or_delta(
        xbin,
        sigma0_space,
    )
    pdf_space[0] = pdf_space_0

    for i in range(1, len(Zout)):
        z_current = Zout[i]

        var_angle = sigma0_angle**2

        source_distance_mm = (
            z_current
            - z0
        )

        varspace_from_source_divergence = (
            sigma0_angle
            * 1.0e-3
            * source_distance_mm
        ) ** 2

        var_space_scatt = (
            varspace_from_source_divergence
        )

        # Point-like window/nozzle contributions.
        for z_kick, sigma_kick in zip(
            z_scatt,
            sigma_scatt_angle,
        ):
            distance = z_current - z_kick

            if distance <= 0.0:
                continue

            var_angle += sigma_kick**2

            sigma_x_kick = (
                sigma_kick
                * 1e-3
                * distance
            )
            var_space_scatt += sigma_x_kick**2

        # Exact distributed-air contributions.
        var_x_air, _, var_theta_air = _distributed_scattering_moments(
            z_current,
            air_segments,
        )

        var_angle += var_theta_air
        var_space_scatt += var_x_air

        sigmas_angle[i] = np.sqrt(max(var_angle, 0.0))

        sigma_space_scatt = np.sqrt(
            max(var_space_scatt, 0.0)
        )

        pdf_scatt_kernel = _gaussian_pdf_or_delta(
            xbin,
            sigma_space_scatt,
        )

        #pdf_space[i] = normalize_pdf(
        #    np.convolve(
        #        pdf_space[i-1],
        #        pdf_scatt_kernel,
        #        mode="same",
        #    ),
        #    xbin,
        #)
        pdf_space[i] = reconvolve_same_grid(
            pdf_space_0,
            pdf_scatt_kernel,
            xbin,
        )

    return Zout, sigmas_angle, pdf_space


def state_space_propagation_with_air(
    z_scatt,
    sigma_scatt_angle,
    sigma0_space,
    sigma0_angle,
    z0,
    z_scorings,
    air_segments=None,
):
    z_scatt = np.asarray(z_scatt, dtype=float)
    sigma_scatt_angle = np.asarray(
        sigma_scatt_angle,
        dtype=float,
    )

    if z_scatt.shape != sigma_scatt_angle.shape:
        raise ValueError(
            "z_scatt and sigma_scatt_angle must have the same shape."
        )

    Zout = np.concatenate((
        [float(z0)],
        np.asarray(z_scorings, dtype=float),
    ))
    nZ = len(Zout)

    sigmas_space = np.zeros(nZ, dtype=float)
    sigmas_angle = np.zeros(nZ, dtype=float)
    covariance = np.zeros(nZ, dtype=float)

    P0 = np.array([
        [sigma0_space**2, 0.0],
        [0.0, sigma0_angle**2],
    ], dtype=float)

    sigmas_space[0] = float(sigma0_space)
    sigmas_angle[0] = float(sigma0_angle)
    covariance[0] = 0.0

    for i in range(1, nZ):
        z_current = Zout[i]
        dz0 = z_current - z0

        # Propagation from the source to z_current.
        A0 = np.array([
            [1.0, dz0 * 1e-3],
            [0.0, 1.0],
        ], dtype=float)

        P = A0 @ P0 @ A0.T

        # Window/nozzle point-like scatterers.
        for z_kick, sigma_kick in zip(
            z_scatt,
            sigma_scatt_angle,
        ):
            if z_kick >= z_current:
                continue

            distance = z_current - z_kick

            Q_kick = np.array([
                [0.0, 0.0],
                [0.0, sigma_kick**2],
            ], dtype=float)

            A_kick = np.array([
                [1.0, distance * 1e-3],
                [0.0, 1.0],
            ], dtype=float)

            P += A_kick @ Q_kick @ A_kick.T

        # Air: exact distributed covariance contribution.
        var_x_air, cov_air, var_theta_air = (
            _distributed_scattering_moments(
                z_current,
                air_segments,
            )
        )

        P_air = np.array([
            [var_x_air, cov_air],
            [cov_air, var_theta_air],
        ], dtype=float)

        P += P_air

        sigmas_space[i] = np.sqrt(max(P[0, 0], 0.0))
        sigmas_angle[i] = np.sqrt(max(P[1, 1], 0.0))
        covariance[i] = P[0, 1]

    return Zout, sigmas_space, sigmas_angle, covariance


def diffQuad_prop_with_air(
    z_scatt,
    sigma_scatt_angle,
    z0,
    z_scorings,
    sigmas_angle_measured,
    sigmas_space_measured,
    air_segments=None,
):

    z_scatt = np.asarray(z_scatt, dtype=float)
    sigma_scatt_angle = np.asarray(
        sigma_scatt_angle,
        dtype=float,
    )
    sigmas_angle_measured = np.asarray(
        sigmas_angle_measured,
        dtype=float,
    )
    sigmas_space_measured = np.asarray(
        sigmas_space_measured,
        dtype=float,
    )

    if z_scatt.shape != sigma_scatt_angle.shape:
        raise ValueError(
            "z_scatt and sigma_scatt_angle must have the same shape."
        )

    Zout = np.concatenate((
        [float(z0)],
        np.asarray(z_scorings, dtype=float),
    ))

    if (
        sigmas_angle_measured.size != len(Zout)
        or sigmas_space_measured.size != len(Zout)
    ):
        raise ValueError(
            "Measured sigma arrays must contain z0 and every "
            "scoring surface."
        )

    sigmas_space = np.zeros(len(Zout), dtype=float)
    sigmas_angle = np.zeros(len(Zout), dtype=float)

    sigmas_space[0] = sigmas_space_measured[0]
    sigmas_angle[0] = sigmas_angle_measured[0]

    for i in range(1, len(Zout)):
        z_current = Zout[i]

        var_angle_scatt = 0.0
        var_space_scatt = 0.0

        # Window/nozzle point-like contributions to remove.
        for z_kick, sigma_kick in zip(
            z_scatt,
            sigma_scatt_angle,
        ):
            distance = z_current - z_kick

            if distance <= 0.0:
                continue

            var_angle_scatt += sigma_kick**2
            var_space_scatt += (
                sigma_kick * 1e-3 * distance
            )**2

        # Air contributions to remove.
        var_x_air, _, var_theta_air = _distributed_scattering_moments(
            z_current,
            air_segments,
        )

        var_angle_scatt += var_theta_air
        var_space_scatt += var_x_air

        var_angle_clean = (
            sigmas_angle_measured[i]**2
            - var_angle_scatt
        )

        var_space_clean = (
            sigmas_space_measured[i]**2
            - var_space_scatt
        )

        sigmas_angle[i] = np.sqrt(
            max(var_angle_clean, 0.0)
        )

        sigmas_space[i] = np.sqrt(
            max(var_space_clean, 0.0)
        )

    return Zout, sigmas_space, sigmas_angle


def state_space_diffquad_prop_with_air(
    z_scatt,
    sigma_scatt_angle,
    z0,
    z_scorings,
    sigmas_angle_measured,
    sigmas_space_measured,
    covariance_measured=None,
    air_segments=None,
):

    z_scatt = np.asarray(z_scatt, dtype=float)
    sigma_scatt_angle = np.asarray(
        sigma_scatt_angle,
        dtype=float,
    )
    sigmas_angle_measured = np.asarray(
        sigmas_angle_measured,
        dtype=float,
    )
    sigmas_space_measured = np.asarray(
        sigmas_space_measured,
        dtype=float,
    )

    if z_scatt.shape != sigma_scatt_angle.shape:
        raise ValueError(
            "z_scatt and sigma_scatt_angle must have the same shape."
        )

    Zout = np.concatenate((
        [float(z0)],
        np.asarray(z_scorings, dtype=float),
    ))
    nZ = len(Zout)

    if (
        sigmas_angle_measured.size != nZ
        or sigmas_space_measured.size != nZ
    ):
        raise ValueError(
            "Measured sigma arrays must contain z0 and every "
            "scoring surface."
        )

    if covariance_measured is None:
        covariance_measured = np.zeros(nZ, dtype=float)
    else:
        covariance_measured = np.asarray(
            covariance_measured,
            dtype=float,
        )

        if covariance_measured.size != nZ:
            raise ValueError(
                "covariance_measured must contain z0 and every "
                "scoring surface."
            )

    sigmas_space = np.zeros(nZ, dtype=float)
    sigmas_angle = np.zeros(nZ, dtype=float)
    covariance = np.zeros(nZ, dtype=float)

    sigmas_space[0] = sigmas_space_measured[0]
    sigmas_angle[0] = sigmas_angle_measured[0]
    covariance[0] = covariance_measured[0]

    for i in range(1, nZ):
        z_current = Zout[i]
        dz_source = z_current - z0

        # Measured covariance matrix at the scoring surface.
        P_meas = np.array([
            [
                sigmas_space_measured[i]**2,
                covariance_measured[i],
            ],
            [
                covariance_measured[i],
                sigmas_angle_measured[i]**2,
            ],
        ], dtype=float)

        # Point-like window/nozzle contribution.
        P_scatt = np.zeros((2, 2), dtype=float)

        for z_kick, sigma_kick in zip(
            z_scatt,
            sigma_scatt_angle,
        ):
            if z_kick >= z_current:
                continue

            distance = z_current - z_kick

            Q_kick = np.array([
                [0.0, 0.0],
                [0.0, sigma_kick**2],
            ], dtype=float)

            A_kick = np.array([
                [1.0, distance * 1e-3],
                [0.0, 1.0],
            ], dtype=float)

            P_scatt += A_kick @ Q_kick @ A_kick.T

        # Exact covariance matrix of distributed air.
        var_x_air, cov_air, var_theta_air = (
            _distributed_scattering_moments(
                z_current,
                air_segments,
            )
        )

        P_air = np.array([
            [var_x_air, cov_air],
            [cov_air, var_theta_air],
        ], dtype=float)

        # Remove materials and air from the measured covariance.
        P_clean = P_meas - P_scatt - P_air

        # Numerical protection: keep diagonal variances non-negative.
        P_clean[0, 0] = max(P_clean[0, 0], 0.0)
        P_clean[1, 1] = max(P_clean[1, 1], 0.0)

        # Enforce positive semidefinite 2x2 covariance matrix.
        max_covariance = np.sqrt(
            max(P_clean[0, 0] * P_clean[1, 1], 0.0)
        )

        P_clean[0, 1] = np.clip(
            P_clean[0, 1],
            -max_covariance,
            max_covariance,
        )
        P_clean[1, 0] = P_clean[0, 1]

        # This is retained for diagnostics and consistency with your
        # original inverse-state-space method.
        A0_inv = np.array([
            [1.0, -dz_source * 1e-3],
            [0.0, 1.0],
        ], dtype=float)

        P0_est = A0_inv @ P_clean @ A0_inv.T

        # Avoid an unused-variable warning and preserve diagnostic
        # information if you inspect this function interactively.
        _ = P0_est

        sigmas_space[i] = np.sqrt(
            max(P_clean[0, 0], 0.0)
        )
        sigmas_angle[i] = np.sqrt(
            max(P_clean[1, 1], 0.0)
        )
        covariance[i] = P_clean[0, 1]

    return Zout, sigmas_space, sigmas_angle, covariance

def _distributed_scattering_moments(
    z_current,
    distributed_segments,
):

    var_x = 0.0
    cov_x_theta = 0.0
    var_theta = 0.0

    if distributed_segments is None:
        return var_x, cov_x_theta, var_theta

    z_current = float(z_current)

    for z_start, z_end, sigma_per_sqrt_mm in distributed_segments:
        z_start = float(z_start)
        z_end = float(z_end)
        sigma_per_sqrt_mm = float(sigma_per_sqrt_mm)

        if z_end <= z_start:
            raise ValueError(
                "Each distributed segment must satisfy "
                "z_end > z_start."
            )

        z_end_eff = min(
            z_end,
            z_current,
        )

        if z_end_eff <= z_start:
            continue

        q = sigma_per_sqrt_mm ** 2

        d_start = z_current - z_start
        d_end = z_current - z_end_eff

        var_theta += q * (
            z_end_eff - z_start
        )

        cov_x_theta += (
            q
            * 1e-3
            * (d_start**2 - d_end**2)
            / 2.0
        )

        var_x += (
            q
            * 1e-6
            * (d_start**3 - d_end**3)
            / 3.0
        )

    return var_x, cov_x_theta, var_theta

def _subdivide_air_segments(air_segments, dz=None):

    if dz is None:
        return [
            (float(z_start), float(z_end), float(sigma_air_permm))
            for z_start, z_end, sigma_air_permm in air_segments
        ]

    dz = float(dz)

    if dz <= 0.0:
        raise ValueError("dz must be positive or None.")

    subsegments = []

    for z_start, z_end, sigma_air_permm in air_segments:
        z_start = float(z_start)
        z_end = float(z_end)
        sigma_air_permm = float(sigma_air_permm)

        length = z_end - z_start

        if length <= 0.0:
            raise ValueError("Each air segment must have positive length.")

        n_parts = max(1, int(np.ceil(length / dz)))
        dz_eff = length / n_parts

        edges = z_start + np.arange(n_parts + 1, dtype=float) * dz_eff

        for z_left, z_right in zip(edges[:-1], edges[1:]):
            subsegments.append((
                float(z_left),
                float(z_right),
                sigma_air_permm,
            ))

    return subsegments


def _run_air_methods(
    z_kicks,
    sigma_kicks_angle,
    air_segments,
    sigma_0_space,
    sigma_0_angle,
    sigma_meas_angle,
    sigma_meas_space,
    xbin,
    z0,
    positions_scoring_mm,
    deconvolution_verbose=False,
):

    result = {}

    (
        result["Zout_prop"],
        result["space_prop"],
        result["angle_prop"],
    ) = sumQuadProp_with_air(
        z_scatt=z_kicks,
        sigma_scatt_angle=sigma_kicks_angle,
        sigma0_space=sigma_0_space,
        sigma0_angle=sigma_0_angle,
        z0=z0,
        z_scorings=positions_scoring_mm,
        air_segments=air_segments,
    )

    (
        result["Zout_conv"],
        result["angle_conv"],
        result["pdf_conv"],
    ) = ConvFinal_with_air(
        z_scatt=z_kicks,
        sigma_scatt_angle=sigma_kicks_angle,
        sigma0_space=sigma_0_space,
        sigma0_angle=sigma_0_angle,
        z0=z0,
        z_scorings=positions_scoring_mm,
        xbin=xbin,
        air_segments=air_segments,
    )

    (
        result["Zout_ss"],
        result["space_ss"],
        result["angle_ss"],
        result["cov_ss"],
    ) = state_space_propagation_with_air(
        z_scatt=z_kicks,
        sigma_scatt_angle=sigma_kicks_angle,
        sigma0_space=sigma_0_space,
        sigma0_angle=sigma_0_angle,
        z0=z0,
        z_scorings=positions_scoring_mm,
        air_segments=air_segments,
    )

    (
        result["Zout_dq"],
        result["space_dq"],
        result["angle_dq"],
    ) = diffQuad_prop_with_air(
        z_scatt=z_kicks,
        sigma_scatt_angle=sigma_kicks_angle,
        z0=z0,
        z_scorings=positions_scoring_mm,
        sigmas_angle_measured=sigma_meas_angle,
        sigmas_space_measured=sigma_meas_space,
        air_segments=air_segments,
    )

    (
        result["Zout_deconv"],
        result["angle_deconv"],
        result["pdf_deconv"],
        result["deconvolution_diagnostics"],
    ) = deconvolutionFinal_with_air(
        z_scatt=z_kicks,
        sigma_scatt_angle=sigma_kicks_angle,
        pdf_meas_space=_build_measured_space_pdfs(
                xbin,
                sigma_meas_space,
            ),
        sigma_meas_space=sigma_meas_space,
        sigma_meas_angle=sigma_meas_angle,
        z0=z0,
        z_scorings=positions_scoring_mm,
        xbin=xbin,
        air_segments=air_segments,
        lambd=None,
        sigma_method="fit",
        verbose=deconvolution_verbose,
    )

    (
        result["Zout_ss_inv"],
        result["space_ss_inv"],
        result["angle_ss_inv"],
        result["cov_ss_inv"],
    ) = state_space_diffquad_prop_with_air(
        z_scatt=z_kicks,
        sigma_scatt_angle=sigma_kicks_angle,
        z0=z0,
        z_scorings=positions_scoring_mm,
        sigmas_angle_measured=sigma_meas_angle,
        sigmas_space_measured=sigma_meas_space,
        covariance_measured=None,
        air_segments=air_segments,
    )

    result["space_conv"] = np.array([
        measure_sigma(
            xbin,
            pdf,
            method="fit",
            fit_window=None,
        )
        for pdf in result["pdf_conv"]
    ], dtype=float)

    # Sigma del fit: è la quantità inviata al plot inverso.
    result["space_deconv"] = np.asarray(
        [
            measure_sigma(
                xbin,
                pdf,
                method="fit",
                fit_window=None,
            )
            for pdf in result["pdf_deconv"]
        ],
        dtype=float,
    )

    result["space_deconv"][0] = float(
        sigma_meas_space[0]
    )

    # RMS globale: conservato separatamente per diagnostica.
    result["space_deconv_moment"] = np.asarray(
        [
            measure_sigma(
                xbin,
                pdf,
                method="fit",
                fit_window=None,
            )
            for pdf in result["pdf_deconv"]
        ],
        dtype=float,
    )

    if deconvolution_verbose:
        print("\n" + "=" * 78)
        print("CONTROLLO SIGMA DECONVOLUTA IN ARIA")
        print("=" * 78)

        diagnostics = result["deconvolution_diagnostics"]

        print(
            f"{'z [mm]':>10} "
            f"{'RMS [mm]':>14} "
            f"{'Fit [mm]':>14} "
            f"{'Plot [mm]':>14} "
            f"{'Lambda':>14} "
            f"{'Neg. raw':>14}"
        )

        for index, (z_value, pdf) in enumerate(
            zip(
                result["Zout_deconv"],
                result["pdf_deconv"],
            )
        ):
            sigma_fit = fit_sigma_from_sampled_pdf(
                xbin,
                pdf,
            )

            print(
                f"{z_value:10.1f} "
                f"{result['space_deconv_moment'][index]:14.8f} "
                f"{sigma_fit:14.8f} "
                f"{result['space_deconv'][index]:14.8f} "
                f"{diagnostics['lambda_values'][index]:14.6e} "
                f"{diagnostics['negative_fraction_raw'][index]:14.6e}"
            )

    return result


def _build_measured_space_pdfs(xbin, sigma_meas_space):


    return [
        normalize_pdf(
            gaussian_model_counts(
                xbin,
                1.0,
                0.0,
                float(sigma),
                0.0,
                1.0,
            ),
            xbin,
        )
        for sigma in sigma_meas_space
    ]

def build_thin_scatterers(
    z_window,
    sigma_window,
):


    z_kicks = np.array(
        [z_window],
        dtype=float,
    )

    sigma_kicks = np.array(
        [sigma_window],
        dtype=float,
    )

    return z_kicks, sigma_kicks

def run_one_distributed_air_model(
    stats_angle,
    physical_air_segments,
    air_dz,
    sigma0_space,
    sigma0_angle,
    sigma_meas_xp,
    sigma_meas_yp,
    sigma_meas_x,
    sigma_meas_y,
    xbin,
    z0,
    z_scoring,
    z_nozzle_start,
    z_nozzle_end,
    ripple_filters=False,
    deconvolution_verbose=False,
):
    results = {}

    for attr in ("xp", "yp"):

        sigma_window = float(
            stats_angle[attr]["window_angle_std"]
        )

        sigma_nozzle = float(
            stats_angle[attr]["nozzle_angle_std"]
            if not ripple_filters
            else stats_angle[attr][
                "nozzle_ripple_angle_std"
            ]
        )

        sigma_meas_angle = (
            sigma_meas_xp
            if attr == "xp"
            else sigma_meas_yp
        )

        sigma_meas_space = (
            sigma_meas_x
            if attr == "xp"
            else sigma_meas_y
        )

        # Vacuum window: scatterer puntiforme
        z_kicks, sigma_kicks = build_thin_scatterers(
            z_window=-1093.0,
            sigma_window=sigma_window,
        )

        # Nozzle: scatterer distribuito
        nozzle_segment = (
            _build_distributed_segment_from_total_sigma(
                z_start=z_nozzle_start,
                z_end=z_nozzle_end,
                sigma_total=sigma_nozzle,
            )
        )

        # Aria + nozzle: tutti gli scatterer distribuiti
        distributed_segments = [
            nozzle_segment,
            *physical_air_segments[attr],
        ]

        # Questo è opzionale:
        # con gli integrali esatti non cambia il risultato fisico
        distributed_segments = _subdivide_air_segments(
            distributed_segments,
            dz=air_dz,
        )

        results[attr] = _run_air_methods(
            z_kicks=z_kicks,
            sigma_kicks_angle=sigma_kicks,
            air_segments=distributed_segments,
            sigma_0_space=sigma0_space,
            sigma_0_angle=sigma0_angle,
            sigma_meas_angle=sigma_meas_angle,
            sigma_meas_space=sigma_meas_space,
            xbin=xbin,
            z0=z0,
            positions_scoring_mm=z_scoring,
            deconvolution_verbose=deconvolution_verbose,
        )

    return results

def run_analysis_with_air_unique_block(
    stats_angle,
    sigma_air_permm_xp,
    sigma_air_permm_yp,
    xbin,
    xpbin,
    z0,
    sigma_0_space,
    sigma_0_angle,
    sigma_meas_xp,
    sigma_meas_yp,
    sigma_meas_x,
    sigma_meas_y,
    positions_scoring_mm,
    results_original_air,
    ripple_filters=False,
):

    del xpbin, results_original_air

    z_scoring = np.asarray(positions_scoring_mm, dtype=float)

    physical_air_segments = {
        "xp": _build_air_segments(
            sigma_air_permm_xp,
            z_scoring,
        ),
        "yp": _build_air_segments(
            sigma_air_permm_yp,
            z_scoring,
        ),
    }
    z_nozzle_start = -925.5 #mm
    z_nozzle_end = -705.5 #mm
    results = run_one_distributed_air_model(
        stats_angle=stats_angle,
        physical_air_segments=physical_air_segments,
        air_dz=None,
        sigma0_space=sigma_0_space,
        sigma0_angle=sigma_0_angle,
        sigma_meas_xp=sigma_meas_xp,
        sigma_meas_yp=sigma_meas_yp,
        sigma_meas_x=sigma_meas_x,
        sigma_meas_y=sigma_meas_y,
        xbin=xbin,
        z0=z0,
        z_scoring=z_scoring,
        z_nozzle_start=z_nozzle_start,
        z_nozzle_end=z_nozzle_end,
        ripple_filters=ripple_filters,
        deconvolution_verbose=False,
    )

    Zout = results["xp"]["Zout_prop"]

    sigma_scatter_attr_withair = {}

    for attr in ("xp", "yp"):
        sigma_window = float(stats_angle[attr]["window_angle_std"])

        sigma_nozzle = float(
            stats_angle[attr]["nozzle_angle_std"]
            if not ripple_filters
            else stats_angle[attr]["nozzle_ripple_angle_std"]
        )

        _, sigma_kicks = build_thin_scatterers(
            z_window=-1093.0,
            sigma_window=sigma_window,
        )

        sigma_scatter_attr_withair[attr] = sigma_kicks

    direct_methods_comparison_withair = pd.DataFrame({
        ("IN SPACE", "", "Scoring surfaces"): Zout,
        ("IN SPACE", "FROM INTEGRAL DATASET", "STD in x (mm)"): sigma_meas_x,
        ("IN SPACE", "FROM INTEGRAL DATASET", "STD in y (mm)"): sigma_meas_y,
        ("IN SPACE", "FROM SUM OF SQUARES", "STD in x (mm)"): results["xp"]["space_prop"],
        ("IN SPACE", "FROM SUM OF SQUARES", "STD in y (mm)"): results["yp"]["space_prop"],
        ("IN SPACE", "FROM CONVOLUTION", "STD in x (mm)"): results["xp"]["space_conv"],
        ("IN SPACE", "FROM CONVOLUTION", "STD in y (mm)"): results["yp"]["space_conv"],
        ("IN SPACE", "FROM STATE SPACE", "STD in x (mm)"): results["xp"]["space_ss"],
        ("IN SPACE", "FROM STATE SPACE", "STD in y (mm)"): results["yp"]["space_ss"],
    })
    direct_methods_comparison_withair.columns = pd.MultiIndex.from_tuples(
        direct_methods_comparison_withair.columns
    )

    inverse_methods_comparison_withair = pd.DataFrame({
        ("IN SPACE", "", "Scoring surfaces"): Zout,
        ("IN SPACE", "FROM DIFFERENCE OF SQUARES", "STD in x (mm)"): results["xp"]["space_dq"],
        ("IN SPACE", "FROM DIFFERENCE OF SQUARES", "STD in y (mm)"): results["yp"]["space_dq"],
        ("IN SPACE", "FROM DECONVOLUTION", "STD in x (mm)"): results["xp"]["space_deconv"],
        ("IN SPACE", "FROM DECONVOLUTION", "STD in y (mm)"): results["yp"]["space_deconv"],
        ("IN SPACE", "FROM INVERSE STATE SPACE", "STD in x (mm)"): results["xp"]["space_ss_inv"],
        ("IN SPACE", "FROM INVERSE STATE SPACE", "STD in y (mm)"): results["yp"]["space_ss_inv"],
    })
    inverse_methods_comparison_withair.columns = pd.MultiIndex.from_tuples(
        inverse_methods_comparison_withair.columns
    )

    direct_methods_comparison_withair_angle = pd.DataFrame({
        ("IN ANGLE", "", "Scoring surfaces"): Zout,
        ("IN ANGLE", "FROM INTEGRAL DATASET", "STD in x' (mrad)"): sigma_meas_xp,
        ("IN ANGLE", "FROM INTEGRAL DATASET", "STD in y' (mrad)"): sigma_meas_yp,
        ("IN ANGLE", "FROM SUM OF SQUARES", "STD in x' (mrad)"): results["xp"]["angle_prop"],
        ("IN ANGLE", "FROM SUM OF SQUARES", "STD in y' (mrad)"): results["yp"]["angle_prop"],
        ("IN ANGLE", "FROM CONVOLUTION", "STD in x' (mrad)"): results["xp"]["angle_conv"],
        ("IN ANGLE", "FROM CONVOLUTION", "STD in y' (mrad)"): results["yp"]["angle_conv"],
    })
    direct_methods_comparison_withair_angle.columns = pd.MultiIndex.from_tuples(
        direct_methods_comparison_withair_angle.columns
    )

    return (
        direct_methods_comparison_withair,
        direct_methods_comparison_withair_angle,
        inverse_methods_comparison_withair,
        sigma_scatter_attr_withair,
    )

def _build_distributed_segment_from_total_sigma(
    z_start,
    z_end,
    sigma_total,
):
    """
    Converte una sigma angolare totale [mrad]
    in un segmento distribuito uniforme.

    Restituisce:
        (z_start, z_end, sigma_per_sqrt_mm)
    """

    z_start = float(z_start)
    z_end = float(z_end)
    sigma_total = float(sigma_total)

    length = z_end - z_start

    if length <= 0.0:
        raise ValueError(
            "z_end must be greater than z_start."
        )

    if sigma_total < 0.0:
        raise ValueError(
            "sigma_total must be non-negative."
        )

    sigma_per_sqrt_mm = sigma_total / np.sqrt(length)

    return (
        z_start,
        z_end,
        sigma_per_sqrt_mm,
    )

def run_analysis_with_air_small_components(
    positions_scoring_mm,
    sigma_air_permm_xp,
    sigma_air_permm_yp,
    stats_angle,
    sigma_scatter_attr_withair,
    sigma_0_space,
    sigma_0_angle,
    sigma_meas_xp,
    sigma_meas_yp,
    sigma_meas_x,
    sigma_meas_y,
    xbin,
    xpbin,
    z0,
    direct_methods_comparison_withair,
    direct_methods_comparison_withair_angle,
    ripple_filters=False,
):

    del sigma_scatter_attr_withair
    del xpbin

    z_scoring = np.asarray(positions_scoring_mm, dtype=float)

    z_nozzle_start = -925.5 #mm
    z_nozzle_end = -705.5 #mm
    
    physical_air_segments = {
        "xp": _build_air_segments(
            sigma_air_permm_xp,
            z_scoring,
        ),
        "yp": _build_air_segments(
            sigma_air_permm_yp,
            z_scoring,
        ),
    }
    one_mm = run_one_distributed_air_model(
        stats_angle=stats_angle,
        physical_air_segments=physical_air_segments,
        air_dz=1.0,
        sigma0_space=sigma_0_space,
        sigma0_angle=sigma_0_angle,
        sigma_meas_xp=sigma_meas_xp,
        sigma_meas_yp=sigma_meas_yp,
        sigma_meas_x=sigma_meas_x,
        sigma_meas_y=sigma_meas_y,
        xbin=xbin,
        z0=z0,
        z_scoring=z_scoring,
        z_nozzle_start=z_nozzle_start,
        z_nozzle_end=z_nozzle_end,
        ripple_filters=ripple_filters,
        deconvolution_verbose=False,
    )

    ten_mm = run_one_distributed_air_model(
        stats_angle=stats_angle,
        physical_air_segments=physical_air_segments,
        air_dz=10.0,
        sigma0_space=sigma_0_space,
        sigma0_angle=sigma_0_angle,
        sigma_meas_xp=sigma_meas_xp,
        sigma_meas_yp=sigma_meas_yp,
        sigma_meas_x=sigma_meas_x,
        sigma_meas_y=sigma_meas_y,
        xbin=xbin,
        z0=z0,
        z_scoring=z_scoring,
        z_nozzle_start=z_nozzle_start,
        z_nozzle_end=z_nozzle_end,
        ripple_filters=ripple_filters,
        deconvolution_verbose=False,
    )

    Zout = one_mm["xp"]["Zout_prop"]

    direct_methods_comparison_withair_total = pd.DataFrame({
        ("IN SPACE", "", "Scoring surfaces"): Zout,
        ("IN SPACE", "FROM INTEGRAL DATASET", "STD in x (mm)"): sigma_meas_x,
        ("IN SPACE", "FROM INTEGRAL DATASET", "STD in y (mm)"): sigma_meas_y,

        ("IN SPACE", "FROM SUM OF SQUARES (1 mm)", "STD in x (mm)"): one_mm["xp"]["space_prop"],
        ("IN SPACE", "FROM SUM OF SQUARES (1 mm)", "STD in y (mm)"): one_mm["yp"]["space_prop"],
        ("IN SPACE", "FROM SUM OF SQUARES (10 mm)", "STD in x (mm)"): ten_mm["xp"]["space_prop"],
        ("IN SPACE", "FROM SUM OF SQUARES (10 mm)", "STD in y (mm)"): ten_mm["yp"]["space_prop"],

        ("IN SPACE", "FROM CONVOLUTION (1 mm)", "STD in x (mm)"): one_mm["xp"]["space_conv"],
        ("IN SPACE", "FROM CONVOLUTION (1 mm)", "STD in y (mm)"): one_mm["yp"]["space_conv"],
        ("IN SPACE", "FROM CONVOLUTION (10 mm)", "STD in x (mm)"): ten_mm["xp"]["space_conv"],
        ("IN SPACE", "FROM CONVOLUTION (10 mm)", "STD in y (mm)"): ten_mm["yp"]["space_conv"],

        ("IN SPACE", "FROM STATE SPACE (1 mm)", "STD in x (mm)"): one_mm["xp"]["space_ss"],
        ("IN SPACE", "FROM STATE SPACE (1 mm)", "STD in y (mm)"): one_mm["yp"]["space_ss"],
        ("IN SPACE", "FROM STATE SPACE (10 mm)", "STD in x (mm)"): ten_mm["xp"]["space_ss"],
        ("IN SPACE", "FROM STATE SPACE (10 mm)", "STD in y (mm)"): ten_mm["yp"]["space_ss"],
    })
    direct_methods_comparison_withair_total.columns = pd.MultiIndex.from_tuples(
        direct_methods_comparison_withair_total.columns
    )

    inverse_methods_comparison_withair_total = pd.DataFrame({
        ("IN SPACE", "", "Scoring surfaces"): Zout,

        ("IN SPACE", "FROM DIFFERENCE OF SQUARES (1 mm)", "STD in x (mm)"): one_mm["xp"]["space_dq"],
        ("IN SPACE", "FROM DIFFERENCE OF SQUARES (1 mm)", "STD in y (mm)"): one_mm["yp"]["space_dq"],
        ("IN SPACE", "FROM DIFFERENCE OF SQUARES (10 mm)", "STD in x (mm)"): ten_mm["xp"]["space_dq"],
        ("IN SPACE", "FROM DIFFERENCE OF SQUARES (10 mm)", "STD in y (mm)"): ten_mm["yp"]["space_dq"],

        ("IN SPACE", "FROM DECONVOLUTION (1 mm)", "STD in x (mm)"): one_mm["xp"]["space_deconv"],
        ("IN SPACE", "FROM DECONVOLUTION (1 mm)", "STD in y (mm)"): one_mm["yp"]["space_deconv"],
        ("IN SPACE", "FROM DECONVOLUTION (10 mm)", "STD in x (mm)"): ten_mm["xp"]["space_deconv"],
        ("IN SPACE", "FROM DECONVOLUTION (10 mm)", "STD in y (mm)"): ten_mm["yp"]["space_deconv"],

        ("IN SPACE", "FROM INVERSE STATE SPACE (1 mm)", "STD in x (mm)"): one_mm["xp"]["space_ss_inv"],
        ("IN SPACE", "FROM INVERSE STATE SPACE (1 mm)", "STD in y (mm)"): one_mm["yp"]["space_ss_inv"],
        ("IN SPACE", "FROM INVERSE STATE SPACE (10 mm)", "STD in x (mm)"): ten_mm["xp"]["space_ss_inv"],
        ("IN SPACE", "FROM INVERSE STATE SPACE (10 mm)", "STD in y (mm)"): ten_mm["yp"]["space_ss_inv"],
    })
    inverse_methods_comparison_withair_total.columns = pd.MultiIndex.from_tuples(
        inverse_methods_comparison_withair_total.columns
    )

    direct_methods_comparison_withair_angle_total = pd.DataFrame({
        ("IN ANGLE", "", "Scoring surfaces"): Zout,
        ("IN ANGLE", "FROM INTEGRAL DATASET", "STD in x' (mrad)"): sigma_meas_xp,
        ("IN ANGLE", "FROM INTEGRAL DATASET", "STD in y' (mrad)"): sigma_meas_yp,

        ("IN ANGLE", "FROM SUM OF SQUARES (1 mm)", "STD in x' (mrad)"): one_mm["xp"]["angle_prop"],
        ("IN ANGLE", "FROM SUM OF SQUARES (1 mm)", "STD in y' (mrad)"): one_mm["yp"]["angle_prop"],
        ("IN ANGLE", "FROM SUM OF SQUARES (10 mm)", "STD in x' (mrad)"): ten_mm["xp"]["angle_prop"],
        ("IN ANGLE", "FROM SUM OF SQUARES (10 mm)", "STD in y' (mrad)"): ten_mm["yp"]["angle_prop"],

        ("IN ANGLE", "FROM CONVOLUTION (1 mm)", "STD in x' (mrad)"): one_mm["xp"]["angle_conv"],
        ("IN ANGLE", "FROM CONVOLUTION (1 mm)", "STD in y' (mrad)"): one_mm["yp"]["angle_conv"],
        ("IN ANGLE", "FROM CONVOLUTION (10 mm)", "STD in x' (mrad)"): ten_mm["xp"]["angle_conv"],
        ("IN ANGLE", "FROM CONVOLUTION (10 mm)", "STD in y' (mrad)"): ten_mm["yp"]["angle_conv"],
    })
    direct_methods_comparison_withair_angle_total.columns = pd.MultiIndex.from_tuples(
        direct_methods_comparison_withair_angle_total.columns
    )

    unique_conv_x = direct_methods_comparison_withair[
        ("IN SPACE", "FROM CONVOLUTION", "STD in x (mm)")
    ]

    unique_conv_y = direct_methods_comparison_withair[
        ("IN SPACE", "FROM CONVOLUTION", "STD in y (mm)")
    ]

    comparison_convolution = pd.DataFrame({
        "Scoring surfaces": Zout,
        "Measured in x": sigma_meas_x,
        "Measured in y": sigma_meas_y,
        "1 mm in x": one_mm["xp"]["space_conv"],
        "1 mm in y": one_mm["yp"]["space_conv"],
        "10 mm in x": ten_mm["xp"]["space_conv"],
        "10 mm in y": ten_mm["yp"]["space_conv"],
        "Unique block in x": unique_conv_x.to_numpy(),
        "Unique block in y": unique_conv_y.to_numpy(),
    })

    return (
        direct_methods_comparison_withair_total,
        direct_methods_comparison_withair_angle_total,
        inverse_methods_comparison_withair_total,
        comparison_convolution,
    )

import matplotlib.colors as mcolors


def plot_comparison_of_dataframes(
    direct_df,
    inverse_df,
    inverse_df_unique,
    inverse_df_small,
    withair_df_unique,
    withair_df_small,
    figsize=(18, 14)
):
    def lighten_color(color, amount=0.35):
        c = np.array(mcolors.to_rgb(color))
        white = np.array([1.0, 1.0, 1.0])
        return tuple((1 - amount) * c + amount * white)

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    axes = axes.flatten()
    
    # DIRECT
    z_direct = direct_df[("", "Scoring surfaces")].values
    meas_x = direct_df[("FROM INTEGRAL DATASET", "STD in x (mm)")].values
    meas_y = direct_df[("FROM INTEGRAL DATASET", "STD in y (mm)")].values
    quad_x = direct_df[("FROM SUM OF SQUARES", "STD in x (mm)")].values
    quad_y = direct_df[("FROM SUM OF SQUARES", "STD in y (mm)")].values
    conv_x = direct_df[("FROM CONVOLUTION", "STD in x (mm)")].values
    conv_y = direct_df[("FROM CONVOLUTION", "STD in y (mm)")].values
    ss_x = direct_df[("FROM STATE SPACE", "STD in x (mm)")].values
    ss_y = direct_df[("FROM STATE SPACE", "STD in y (mm)")].values

    c_meas = "#1f77b4"
    c_quad = "#d62728"
    c_conv = "#2ca02c"
    c_ss = "#ff7f0e"  

    axes[0].plot(z_direct, meas_x, "-o", color=c_meas, lw=2, ms=5, label="Measured x")
    axes[0].plot(z_direct, meas_y, "-o", color=lighten_color(c_meas, 0.45), lw=2, ms=5, label="Measured y")

    axes[0].plot(z_direct, quad_x, "-s", color=c_quad, lw=2, ms=5, label="Sum of squares x")
    axes[0].plot(z_direct, quad_y, "-s", color=lighten_color(c_quad, 0.45), lw=2, ms=5, label="Sum of squares sum y")

    axes[0].plot(z_direct, conv_x, "-^", color=c_conv, lw=2, ms=5, label="Convolution x")
    axes[0].plot(z_direct, conv_y, "-^", color=lighten_color(c_conv, 0.45), lw=2, ms=5, label="Convolution y")

    axes[0].plot(z_direct, ss_x, "-v", color=c_ss, lw=2, ms=5, label="State space x")
    axes[0].plot(z_direct, ss_y, "-v", color=lighten_color(c_ss, 0.45), lw=2, ms=5, label="State space y")

    axes[0].set_title("Direct methods comparison in vacuum")
    axes[0].set_xlabel("Scoring surfaces")
    axes[0].set_ylabel("STD [mm]")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    # INVERSE
    "IN SPACE", "", "Scoring surfaces"
    z_inverse = inverse_df[("", "Scoring surfaces")].values

    theoretical_x = direct_df[("FROM INTEGRAL DATASET", "STD in x (mm)")].iloc[0]
    theoretical_y = direct_df[("FROM INTEGRAL DATASET", "STD in y (mm)")].iloc[0]

    theoretical_x_line = np.full(len(z_inverse), theoretical_x)
    theoretical_y_line = np.full(len(z_inverse), theoretical_y)

    diffq_x = inverse_df[("FROM DIFFERENCE OF SQUARES", "STD in x (mm)")].values
    diffq_y = inverse_df[("FROM DIFFERENCE OF SQUARES", "STD in y (mm)")].values

    deconv_x = inverse_df[("FROM DECONVOLUTION", "STD in x (mm)")].values
    deconv_y = inverse_df[("FROM DECONVOLUTION", "STD in y (mm)")].values
    
    inverse_ss_x = inverse_df[("FROM STATE SPACE", "STD in x (mm)")].values
    inverse_ss_y = inverse_df[("FROM STATE SPACE", "STD in y (mm)")].values


    c_diffq = "#9467bd"
    c_deconv = "#ff7f0e"
    c_theory = "#222222"
    c_ss_inverse = "#2ca02c"


    axes[1].plot(z_inverse, diffq_x, "-s", color=c_diffq, lw=2, ms=5, label="Difference of squares x")
    axes[1].plot(z_inverse, diffq_y, "-s", color=lighten_color(c_diffq, 0.45), lw=2, ms=5, label="Difference of squares y")

    axes[1].plot(z_inverse, deconv_x, "-^", color=c_deconv, lw=2, ms=5, label="Deconvolution x")
    axes[1].plot(z_inverse, deconv_y, "-^", color=lighten_color(c_deconv, 0.45), lw=2, ms=5, label="Deconvolution y")

    axes[1].plot(z_inverse, theoretical_x_line, "--", color=c_theory, lw=2.5, label="Theoretical value at s0 (x)")
    axes[1].plot(z_inverse, theoretical_y_line, "--", color=lighten_color(c_theory, 0.45), lw=2.5, label="Theoretical value at s0 (y)")

    axes[1].plot(z_inverse, inverse_ss_x, "-v", color=c_ss_inverse, lw=2, ms=5, label="State space x")
    axes[1].plot(z_inverse, inverse_ss_y, "-v", color=lighten_color(c_ss_inverse, 0.45), lw=2, ms=5, label="State space y")

    axes[1].set_title("Inverse methods comparison in vacuum")
    axes[1].set_xlabel("Scoring surfaces")
    axes[1].set_ylabel("STD [mm]")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    # AIR X
    z_air = withair_df_small[("IN SPACE", "", "Scoring surfaces")].values
    
    meas_air_1mm_x= withair_df_small[("IN SPACE", "FROM INTEGRAL DATASET", "STD in x (mm)")].values
    quad_air_unique_x = withair_df_unique[("IN SPACE", "FROM SUM OF SQUARES", "STD in x (mm)")].values
    quad_air_1mm_x = withair_df_small[("IN SPACE", "FROM SUM OF SQUARES (1 mm)", "STD in x (mm)")].values
    quad_air_10mm_x = withair_df_small[("IN SPACE", "FROM SUM OF SQUARES (10 mm)", "STD in x (mm)")].values
    conv_air_unique_x = withair_df_unique[("IN SPACE", "FROM CONVOLUTION", "STD in x (mm)")].values
    conv_air_1mm_x = withair_df_small[("IN SPACE", "FROM CONVOLUTION (1 mm)", "STD in x (mm)")].values
    conv_air_10mm_x = withair_df_small[("IN SPACE", "FROM CONVOLUTION (10 mm)", "STD in x (mm)")].values
    ss_air_unique_x = withair_df_unique[("IN SPACE", "FROM STATE SPACE", "STD in x (mm)")].values
    ss_air_1mm_x = withair_df_small[("IN SPACE", "FROM STATE SPACE (1 mm)", "STD in x (mm)")].values
    ss_air_10mm_x = withair_df_small[("IN SPACE", "FROM STATE SPACE (10 mm)", "STD in x (mm)")].values

    c_quad_air = "#d62728"
    c_conv_air = "#2ca02c"
    c_theory_air = "#222222"
    c_ss_air = "#ff7f0e"
    
    axes[2].plot(z_air, meas_air_1mm_x, "-o", color=c_theory_air, lw=2, ms=5, label="Measured x")
    axes[2].plot(z_air, quad_air_unique_x, "-o", color=lighten_color(c_quad_air, 0.20), lw=2, ms=5, label="Sum of squares, air blocks unique")
    axes[2].plot(z_air, quad_air_1mm_x, "-o", color=c_quad_air, lw=2, ms=5, label="Sum of squares, air blocks 1 mm")
    axes[2].plot(z_air, quad_air_10mm_x, "-o", color=lighten_color(c_quad_air, 0.40), lw=2, ms=5, label="Sum of squares, air blocks 10 mm")
    axes[2].plot(z_air, conv_air_unique_x, "-^", color=lighten_color(c_conv_air, 0.20), lw=2, ms=5, label="Convolution, air blocks unique")
    axes[2].plot(z_air, conv_air_1mm_x, "-^", color=c_conv_air, lw=2, ms=5, label="Convolution, air blocks 1 mm")
    axes[2].plot(z_air, conv_air_10mm_x, "-^", color=lighten_color(c_conv_air, 0.40), lw=2, ms=5, label="Convolution, air blocks 10 mm")
    axes[2].plot(z_air, ss_air_unique_x, "-v", color=lighten_color(c_ss_air, 0.20), lw=2, ms=5, label="State space, air blocks unique")
    axes[2].plot(z_air, ss_air_1mm_x, "-v", color=c_ss_air, lw=2, ms=5, label="State space, air blocks 1 mm")
    axes[2].plot(z_air, ss_air_10mm_x, "-v", color=lighten_color(c_ss_air, 0.40), lw=2, ms=5, label="State space, air blocks 10 mm")

    axes[2].set_title("Direct methods comparison in air in x")
    axes[2].set_xlabel("Scoring surfaces")
    axes[2].set_ylabel("STD in x [mm]")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)


    # AIR Y
    meas_air_1mm_y= withair_df_small[("IN SPACE", "FROM INTEGRAL DATASET", "STD in y (mm)")].values
    quad_air_unique_y = withair_df_unique[("IN SPACE", "FROM SUM OF SQUARES", "STD in y (mm)")].values
    conv_air_unique_y = withair_df_unique[("IN SPACE", "FROM CONVOLUTION", "STD in y (mm)")].values
    ss_air_unique_y = withair_df_unique[("IN SPACE", "FROM STATE SPACE", "STD in y (mm)")].values
    quad_air_1mm_y = withair_df_small[("IN SPACE", "FROM SUM OF SQUARES (1 mm)", "STD in y (mm)")].values
    quad_air_10mm_y = withair_df_small[("IN SPACE", "FROM SUM OF SQUARES (10 mm)", "STD in y (mm)")].values
    conv_air_1mm_y = withair_df_small[("IN SPACE", "FROM CONVOLUTION (1 mm)", "STD in y (mm)")].values
    conv_air_10mm_y = withair_df_small[("IN SPACE", "FROM CONVOLUTION (10 mm)", "STD in y (mm)")].values
    ss_air_1mm_y = withair_df_small[("IN SPACE", "FROM STATE SPACE (1 mm)", "STD in y (mm)")].values
    ss_air_10mm_y = withair_df_small[("IN SPACE", "FROM STATE SPACE (10 mm)", "STD in y (mm)")].values
    c_quad_air_y = "#1f77b4"
    c_conv_air_y = "#ff7f0e"
    c_ss_air_y = "#2ca02c"

    axes[3].plot(z_air, meas_air_1mm_y, "-o", color=c_theory_air, lw=2, ms=5, label="Measured y")
    axes[3].plot(z_air, quad_air_unique_y, "-o", color=lighten_color(c_quad_air_y, 0.20), lw=2, ms=5, label="Sum of squares, air blocks unique")
    axes[3].plot(z_air, quad_air_1mm_y, "-o", color=c_quad_air_y, lw=2, ms=5, label="Sum of squares, air blocks 1 mm")
    axes[3].plot(z_air, quad_air_10mm_y, "-o", color=lighten_color(c_quad_air_y, 0.40), lw=2, ms=5, label="Sum of squares, air blocks 10 mm")
    axes[3].plot(z_air, conv_air_unique_y, "-^", color=lighten_color(c_conv_air_y, 0.20), lw=2, ms=5, label="Convolution, air blocks unique")
    axes[3].plot(z_air, conv_air_1mm_y, "-^", color=c_conv_air_y, lw=2, ms=5, label="Convolution, air blocks 1 mm")
    axes[3].plot(z_air, conv_air_10mm_y, "-^", color=lighten_color(c_conv_air_y, 0.40), lw=2, ms=5, label="Convolution, air blocks 10 mm")
    axes[3].plot(z_air, ss_air_1mm_y, "-v", color=c_ss_air_y, lw=2, ms=5, label="State space, air blocks 1 mm")
    axes[3].plot(z_air, ss_air_unique_y, "-v", color=lighten_color(c_ss_air_y, 0.20), lw=2, ms=5, label="State space, air blocks unique")
    axes[3].plot(z_air, ss_air_10mm_y, "-v", color=lighten_color(c_ss_air_y, 0.40), lw=2, ms=5, label="State space, air blocks 10 mm")

    axes[3].set_title("Direct methods comparison in air in y")
    axes[3].set_xlabel("Scoring surfaces")
    axes[3].set_ylabel("STD in y [mm]")
    axes[3].grid(alpha=0.3)
    axes[3].legend(fontsize=8)


    #AIR INVERSE IN X
    meas_air_1mm_x= withair_df_small[("IN SPACE", "FROM INTEGRAL DATASET", "STD in x (mm)")].iloc[0]
    theoretical_x_line_air = np.full(len(z_air), meas_air_1mm_x)
    diff_quad_air_unique_x = inverse_df_unique[("IN SPACE", "FROM DIFFERENCE OF SQUARES", "STD in x (mm)")].values
    deconv_air_unique_x = inverse_df_unique[("IN SPACE", "FROM DECONVOLUTION", "STD in x (mm)")].values
    inverse_ss_air_unique_x = inverse_df_unique[("IN SPACE", "FROM INVERSE STATE SPACE", "STD in x (mm)")].values
    diff_quad_air_1mm_x = inverse_df_small[("IN SPACE", "FROM DIFFERENCE OF SQUARES (1 mm)", "STD in x (mm)")].values
    diff_quad_air_10mm_x = inverse_df_small[("IN SPACE", "FROM DIFFERENCE OF SQUARES (10 mm)", "STD in x (mm)")].values
    deconv_air_1mm_x = inverse_df_small[("IN SPACE", "FROM DECONVOLUTION (1 mm)", "STD in x (mm)")].values
    deconv_air_10mm_x = inverse_df_small[("IN SPACE", "FROM DECONVOLUTION (10 mm)", "STD in x (mm)")].values
    inverse_ss_air_1mm_x = inverse_df_small[("IN SPACE", "FROM INVERSE STATE SPACE (1 mm)", "STD in x (mm)")].values
    inverse_ss_air_10mm_x = inverse_df_small[("IN SPACE", "FROM INVERSE STATE SPACE (10 mm)", "STD in x (mm)")].values

    c_diff_quad_air = "#d62728"
    c_deconv_air = "#2ca02c"
    c_theory_air = "#222222"
    c_inverse_ss_air = "#ff7f0e"
    
    axes[4].plot(z_air, theoretical_x_line_air, "-o", color=c_theory_air, lw=2, ms=5, label="Measured x")
    axes[4].plot(z_air, diff_quad_air_unique_x, "-o", color=lighten_color(c_diff_quad_air, 0.20), lw=2, ms=5, label="Difference of squares, air blocks unique")
    axes[4].plot(z_air, diff_quad_air_1mm_x, "-o", color=c_diff_quad_air, lw=2, ms=5, label="Difference of squares, air blocks 1 mm")
    axes[4].plot(z_air, diff_quad_air_10mm_x, "-o", color=lighten_color(c_diff_quad_air, 0.40), lw=2, ms=5, label="Difference of squares, air blocks 10 mm")
    axes[4].plot(z_air, deconv_air_unique_x, "-^", color=lighten_color(c_deconv_air, 0.20), lw=2, ms=5, label="Deconvolution, air blocks unique")
    axes[4].plot(z_air, deconv_air_1mm_x, "-^", color=c_deconv_air, lw=2, ms=5, label="Deconvolution, air blocks 1 mm")
    axes[4].plot(z_air, deconv_air_10mm_x, "-^", color=lighten_color(c_deconv_air, 0.40), lw=2, ms=5, label="Deconvolution, air blocks 10 mm")
    axes[4].plot(z_air, inverse_ss_air_unique_x, "-v", color=lighten_color(c_inverse_ss_air, 0.20), lw=2, ms=5, label="Inverse state space, air blocks unique")
    axes[4].plot(z_air, inverse_ss_air_1mm_x, "-v", color=c_inverse_ss_air, lw=2, ms=5, label="Inverse state space, air blocks 1 mm")
    axes[4].plot(z_air, inverse_ss_air_10mm_x, "-v", color=lighten_color(c_inverse_ss_air, 0.40), lw=2, ms=5, label="Inverse state space, air blocks 10 mm")

    axes[4].set_title("Inverse methods comparison in air in x")
    axes[4].set_xlabel("Scoring surfaces")
    axes[4].set_ylabel("STD in x [mm]")
    axes[4].grid(alpha=0.3)
    axes[4].legend(fontsize=8)


    #AIR INVERSE IN Y
    meas_air_1mm_y= withair_df_small[("IN SPACE", "FROM INTEGRAL DATASET", "STD in y (mm)")].iloc[0]
    theoretical_y_line_air = np.full(len(z_air), meas_air_1mm_y)
    diff_quad_air_unique_y = inverse_df_unique[("IN SPACE", "FROM DIFFERENCE OF SQUARES", "STD in y (mm)")].values
    deconv_air_unique_y = inverse_df_unique[("IN SPACE", "FROM DECONVOLUTION", "STD in y (mm)")].values
    inverse_ss_air_unique_y = inverse_df_unique[("IN SPACE", "FROM INVERSE STATE SPACE", "STD in y (mm)")].values
    diff_quad_air_1mm_y = inverse_df_small[("IN SPACE", "FROM DIFFERENCE OF SQUARES (1 mm)", "STD in y (mm)")].values
    diff_quad_air_10mm_y = inverse_df_small[("IN SPACE", "FROM DIFFERENCE OF SQUARES (10 mm)", "STD in y (mm)")].values
    deconv_air_1mm_y = inverse_df_small[("IN SPACE", "FROM DECONVOLUTION (1 mm)", "STD in y (mm)")].values
    deconv_air_10mm_y = inverse_df_small[("IN SPACE", "FROM DECONVOLUTION (10 mm)", "STD in y (mm)")].values
    inverse_ss_air_1mm_y = inverse_df_small[("IN SPACE", "FROM INVERSE STATE SPACE (1 mm)", "STD in y (mm)")].values
    inverse_ss_air_10mm_y = inverse_df_small[("IN SPACE", "FROM INVERSE STATE SPACE (10 mm)", "STD in y (mm)")].values

    c_diff_quad_air_y = "#d62728"
    c_deconv_air_y = "#2ca02c"
    c_theory_air_y = "#222222"
    c_inverse_ss_air_y = "#ff7f0e"

    axes[5].plot(z_air, theoretical_y_line_air, "-o", color=c_theory_air_y, lw=2, ms=5, label="Measured y")
    axes[5].plot(z_air, diff_quad_air_unique_y, "-o", color=lighten_color(c_diff_quad_air_y, 0.20), lw=2, ms=5, label="Difference of squares, air blocks unique")
    axes[5].plot(z_air, diff_quad_air_1mm_y, "-o", color=c_diff_quad_air_y, lw=2, ms=5, label="Difference of squares, air blocks 1 mm")
    axes[5].plot(z_air, diff_quad_air_10mm_y, "-o", color=lighten_color(c_diff_quad_air_y, 0.40), lw=2, ms=5, label="Difference of squares, air blocks 10 mm")
    axes[5].plot(z_air, deconv_air_unique_y, "-^", color=lighten_color(c_deconv_air_y, 0.20), lw=2, ms=5, label="Deconvolution, air blocks unique")
    axes[5].plot(z_air, deconv_air_1mm_y, "-^", color=c_deconv_air_y, lw=2, ms=5, label="Deconvolution, air blocks 1 mm")
    axes[5].plot(z_air, deconv_air_10mm_y, "-^", color=lighten_color(c_deconv_air_y, 0.40), lw=2, ms=5, label="Deconvolution, air blocks 10 mm")
    axes[5].plot(z_air, inverse_ss_air_unique_y, "-v", color=lighten_color(c_inverse_ss_air_y, 0.20), lw=2, ms=5, label="Inverse state space, air blocks unique")
    axes[5].plot(z_air, inverse_ss_air_1mm_y, "-v", color=c_inverse_ss_air_y, lw=2, ms=5, label="Inverse state space, air blocks 1 mm")
    axes[5].plot(z_air, inverse_ss_air_10mm_y, "-v", color=lighten_color(c_inverse_ss_air_y, 0.40), lw=2, ms=5, label="Inverse state space, air blocks 10 mm")

    axes[5].set_title("Inverse methods comparison in air in y")
    axes[5].set_xlabel("Scoring surfaces")
    axes[5].set_ylabel("STD in y [mm]")
    axes[5].grid(alpha=0.3)
    axes[5].legend(fontsize=8)

    plt.tight_layout()
    return fig

def plot_final_comparison_all_methods(
    direct_df,
    inverse_df,
    inverse_df_unique,
    inverse_df_small,
    withair_df_unique,
    withair_df_small,
    figsize=(12, 10),
):
    
    def lighten_color(color, amount=0.35):
        c = np.array(mcolors.to_rgb(color))
        white = np.array([1.0, 1.0, 1.0])
        return tuple((1 - amount) * c + amount * white)

    # Font settings
    title_size = 18
    label_size = 16
    tick_size = 14
    line_width = 2.5
    marker_size = 6

    fig, axes = plt.subplots(2, 1, figsize=figsize)
    axes = axes.flatten()

    # ============================================================
    # AIR X
    # ============================================================
    z_air = withair_df_small[
        ("IN SPACE", "", "Scoring surfaces")
    ].values

    meas_air_1mm_x_dir = withair_df_small[
        ("IN SPACE", "FROM INTEGRAL DATASET", "STD in x (mm)")
    ].values

    meas_air_1mm_x = withair_df_small[
        ("IN SPACE", "FROM INTEGRAL DATASET", "STD in x (mm)")
    ].iloc[0]

    theoretical_x_line_air = np.full(
        len(z_air),
        meas_air_1mm_x,
    )

    quad_air_unique_x = withair_df_unique[
        ("IN SPACE", "FROM SUM OF SQUARES", "STD in x (mm)")
    ].values

    conv_air_unique_x = withair_df_unique[
        ("IN SPACE", "FROM CONVOLUTION", "STD in x (mm)")
    ].values

    ss_air_unique_x = withair_df_unique[
        ("IN SPACE", "FROM STATE SPACE", "STD in x (mm)")
    ].values

    diff_quad_air_unique_x = inverse_df_unique[
        ("IN SPACE", "FROM DIFFERENCE OF SQUARES", "STD in x (mm)")
    ].values

    deconv_air_unique_x = inverse_df_unique[
        ("IN SPACE", "FROM DECONVOLUTION", "STD in x (mm)")
    ].values

    inverse_ss_air_unique_x = inverse_df_unique[
        ("IN SPACE", "FROM INVERSE STATE SPACE", "STD in x (mm)")
    ].values

    c_quad_air = "#d62728"
    c_conv_air = "#2ca02c"
    c_theory_air = "#222222"
    c_ss_air = "#3d50f7"

    axes[0].plot(
        z_air,
        meas_air_1mm_x_dir,
        "-o",
        color=c_theory_air,
        lw=line_width,
        ms=marker_size,
    )

    axes[0].plot(
        z_air,
        quad_air_unique_x,
        "-o",
        color=c_quad_air,
        lw=line_width,
        ms=marker_size,
    )

    axes[0].plot(
        z_air,
        conv_air_unique_x,
        "-^",
        color=c_conv_air,
        lw=line_width,
        ms=marker_size,
    )

    axes[0].plot(
        z_air,
        ss_air_unique_x,
        "-v",
        color=c_ss_air,
        lw=line_width,
        ms=marker_size,
    )

    axes[0].plot(
        z_air,
        theoretical_x_line_air,
        "--",
        color=lighten_color(c_theory_air, 0.40),
        lw=line_width,
    )

    axes[0].plot(
        z_air,
        diff_quad_air_unique_x,
        "-o",
        color=lighten_color(c_quad_air, 0.40),
        lw=line_width,
        ms=marker_size,
    )

    axes[0].plot(
        z_air,
        deconv_air_unique_x,
        "-^",
        color=lighten_color(c_conv_air, 0.40),
        lw=line_width,
        ms=marker_size,
    )

    axes[0].plot(
        z_air,
        inverse_ss_air_unique_x,
        "-v",
        color=lighten_color(c_ss_air, 0.40),
        lw=line_width,
        ms=marker_size,
    )

    axes[0].set_title(
        "Direct and inverse methods comparison in air — x direction",
        fontsize=title_size,
        fontweight="bold",
        pad=12,
    )

    axes[0].set_xlabel(
        "Scoring surfaces",
        fontsize=label_size,
        fontweight="bold",
        labelpad=10,
    )

    axes[0].set_ylabel(
        "STD in x [mm]",
        fontsize=label_size,
        fontweight="bold",
        labelpad=10,
    )

    axes[0].tick_params(
        axis="both",
        which="major",
        labelsize=tick_size,
        width=1.5,
        length=6,
    )

    for tick in axes[0].get_xticklabels():
        tick.set_fontweight("bold")

    for tick in axes[0].get_yticklabels():
        tick.set_fontweight("bold")

    axes[0].grid(alpha=0.3, linewidth=1.0)

    # No legend:
    # axes[0].legend(...)

    # ============================================================
    # AIR Y
    # ============================================================
    meas_air_1mm_y_dir = withair_df_small[
        ("IN SPACE", "FROM INTEGRAL DATASET", "STD in y (mm)")
    ].values

    meas_air_1mm_y = withair_df_small[
        ("IN SPACE", "FROM INTEGRAL DATASET", "STD in y (mm)")
    ].iloc[0]

    theoretical_y_line_air = np.full(
        len(z_air),
        meas_air_1mm_y,
    )

    quad_air_unique_y = withair_df_unique[
        ("IN SPACE", "FROM SUM OF SQUARES", "STD in y (mm)")
    ].values

    conv_air_unique_y = withair_df_unique[
        ("IN SPACE", "FROM CONVOLUTION", "STD in y (mm)")
    ].values

    ss_air_unique_y = withair_df_unique[
        ("IN SPACE", "FROM STATE SPACE", "STD in y (mm)")
    ].values

    diff_quad_air_unique_y = inverse_df_unique[
        ("IN SPACE", "FROM DIFFERENCE OF SQUARES", "STD in y (mm)")
    ].values

    deconv_air_unique_y = inverse_df_unique[
        ("IN SPACE", "FROM DECONVOLUTION", "STD in y (mm)")
    ].values

    inverse_ss_air_unique_y = inverse_df_unique[
        ("IN SPACE", "FROM INVERSE STATE SPACE", "STD in y (mm)")
    ].values

    axes[1].plot(
        z_air,
        meas_air_1mm_y_dir,
        "-o",
        color=c_theory_air,
        lw=line_width,
        ms=marker_size,
    )

    axes[1].plot(
        z_air,
        quad_air_unique_y,
        "-o",
        color=c_quad_air,
        lw=line_width,
        ms=marker_size,
    )

    axes[1].plot(
        z_air,
        conv_air_unique_y,
        "-^",
        color=c_conv_air,
        lw=line_width,
        ms=marker_size,
    )

    axes[1].plot(
        z_air,
        ss_air_unique_y,
        "-v",
        color=c_ss_air,
        lw=line_width,
        ms=marker_size,
    )

    axes[1].plot(
        z_air,
        theoretical_y_line_air,
        "--",
        color=lighten_color(c_theory_air, 0.40),
        lw=line_width,
    )

    axes[1].plot(
        z_air,
        diff_quad_air_unique_y,
        "-o",
        color=lighten_color(c_quad_air, 0.40),
        lw=line_width,
        ms=marker_size,
    )

    axes[1].plot(
        z_air,
        deconv_air_unique_y,
        "-^",
        color=lighten_color(c_conv_air, 0.40),
        lw=line_width,
        ms=marker_size,
    )

    axes[1].plot(
        z_air,
        inverse_ss_air_unique_y,
        "-v",
        color=lighten_color(c_ss_air, 0.40),
        lw=line_width,
        ms=marker_size,
    )

    axes[1].set_title(
        "Direct and inverse methods comparison in air — y direction",
        fontsize=title_size,
        fontweight="bold",
        pad=12,
    )

    axes[1].set_xlabel(
        "Scoring surfaces",
        fontsize=label_size,
        fontweight="bold",
        labelpad=10,
    )

    axes[1].set_ylabel(
        "STD in y [mm]",
        fontsize=label_size,
        fontweight="bold",
        labelpad=10,
    )

    axes[1].tick_params(
        axis="both",
        which="major",
        labelsize=tick_size,
        width=1.5,
        length=6,
    )

    for tick in axes[1].get_xticklabels():
        tick.set_fontweight("bold")

    for tick in axes[1].get_yticklabels():
        tick.set_fontweight("bold")

    axes[1].grid(alpha=0.3, linewidth=1.0)

    # No legend:
    # axes[1].legend(...)

    fig.tight_layout()

    return fig

def twiss_ellipse_points(alpha, beta, epsilon, x0=0.0, xp0=0.0, npts=600):
    alpha = float(alpha)
    beta = float(beta)
    epsilon = float(epsilon)
    x0 = float(x0)
    xp0 = float(xp0)

    if beta <= 0:
        raise ValueError(f"beta must be positive, found beta={beta}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, found epsilon={epsilon}")

    t = np.linspace(0.0, 2.0*np.pi, int(npts))
    x = np.sqrt(epsilon * beta) * np.cos(t)
    xp = -np.sqrt(epsilon / beta) * (alpha * np.cos(t) + np.sin(t))

    return x + x0, xp + xp0
def get_twiss_level_result(surface_twiss_results, target_label="60.65% peak density"):
    for item in surface_twiss_results:
        if not isinstance(item, dict):
            continue
        if str(item.get("label", "")).strip() == str(target_label).strip():
            return item

    available = [
        item.get("label", None)
        for item in surface_twiss_results
        if isinstance(item, dict)
    ]
    raise ValueError(
        f"Label {target_label!r} not found. Available: {available}"
    )


def plot_twiss_ellipse_comparison(
    stats_surface,
    twiss_surface_results_xxp,
    twiss_surface_results_yyp,
    target_label="60.65% peak density",
    center_stat_ellipse_on_fit=True,
    figsize=(15, 7.5),
    show=False
):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_configs = [
        {
            "ax": axes[0],
            "twiss_surface_results": twiss_surface_results_xxp,
            "plane": "x_xp",
            "alpha_key": "alpha_x",
            "beta_key": "beta_x",
            "gamma_key": "gamma_x",
            "epsilon_key": "emittance_x",
            "xlabel": "x [mm]",
            "ylabel": "x' [mrad]",
            "title_plane": "x-x'"
        },
        {
            "ax": axes[1],
            "twiss_surface_results": twiss_surface_results_yyp,
            "plane": "y_yp",
            "alpha_key": "alpha_y",
            "beta_key": "beta_y",
            "gamma_key": "gamma_y",
            "epsilon_key": "emittance_y",
            "xlabel": "y [mm]",
            "ylabel": "y' [mrad]",
            "title_plane": "y-y'"
        }
    ]

    zpos = stats_surface.get("z_position", stats_surface.get("zposition", "N/A"))

    for cfg in plot_configs:
        ax = cfg["ax"]
        twiss_surface_results = cfg["twiss_surface_results"]

        level_res = get_twiss_level_result(
            twiss_surface_results,
            target_label=target_label
        )

        x_fit_ellipse = np.asarray(level_res["x_fit_ellipse"], dtype=float)
        y_fit_ellipse = np.asarray(level_res["y_fit_ellipse"], dtype=float)

        x0_fit = float(level_res.get("x0", 0.0))
        y0_fit = float(level_res.get("y0", level_res.get("xp0", 0.0)))

        alpha_stat = float(stats_surface[cfg["alpha_key"]])
        beta_stat = float(stats_surface[cfg["beta_key"]])
        gamma_stat = float(stats_surface[cfg["gamma_key"]])
        epsilon_stat = float(stats_surface[cfg["epsilon_key"]])

        if center_stat_ellipse_on_fit:
            x0_stat = x0_fit
            y0_stat = y0_fit
        else:
            x0_stat = 0.0
            y0_stat = 0.0

        x_stat_ellipse, y_stat_ellipse = twiss_ellipse_points(
            alpha=alpha_stat,
            beta=beta_stat,
            epsilon=epsilon_stat,
            x0=x0_stat,
            xp0=y0_stat,
            npts=600
        )

        all_labels = []
        for item in twiss_surface_results:
            if not isinstance(item, dict):
                continue
            if "x_contour" not in item:
                continue

            x_cont = np.asarray(item["x_contour"], dtype=float)
            y_cont = np.asarray(item.get("y_contour", item.get("xp_contour")), dtype=float)
            this_label = str(item.get("label", "unknown"))

            is_target = (this_label.strip() == str(target_label).strip())

            ax.scatter(
                x_cont,
                y_cont,
                s=18 if is_target else 12,
                c="black",
                alpha=0.95 if is_target else 0.55,
                linewidths=0,
                label=f"Contour from 3D plot at {this_label}" if this_label not in all_labels else None,
                zorder=3 if is_target else 2
            )
            all_labels.append(this_label)

        ax.plot(
            x_fit_ellipse,
            y_fit_ellipse,
            color="red",
            lw=1.6,
            ls="-",
            alpha=1.0,
            label=f"Ellipse from fit ({target_label})",
            zorder=4
        )

        ax.plot(
            x_stat_ellipse,
            y_stat_ellipse,
            color="tab:blue",
            lw=1.75,
            ls="--",
            alpha=0.5,
            label="Ellipse from covariance matrix",
            zorder=1
        )

        ax.set_title(f"{cfg['title_plane']} | z = {zpos} cm")
        ax.set_xlabel(cfg["xlabel"])
        ax.set_ylabel(cfg["ylabel"])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlim(-20, 20)
        ax.set_ylim(-10, 10)

    fig.tight_layout()

    if show:
        plt.show()

    return fig


def make_global_analysis_grid(
    dataframes,
    columns,
    n_points=4097,
    low_percentile=0.05,
    high_percentile=99.95,
    padding_fraction=0.30,
    minimum_half_width=1.0,
    min_points_per_sigma=10,
    verbose=True,
):
    """
    Crea una griglia uniforme globale per convoluzione/deconvoluzione.

    Diversamente dal binning per istogrammi, questa funzione NON forza
    la griglia a includere tutti i min/max assoluti: outlier estremi
    renderebbero dx troppo grande e distruggerebbero la rappresentazione
    di PDF e kernel stretti.

    Il range è costruito dai percentili robusti più un padding fisico.
    """
    if isinstance(columns, str):
        columns = [columns]

    columns = list(columns)

    if not columns:
        raise ValueError("columns cannot be empty.")

    if int(n_points) < 3:
        raise ValueError("n_points must be at least 3.")

    values_list = []

    for dataframe in dataframes:
        if dataframe is None:
            continue

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "All entries in dataframes must be pandas DataFrames or None."
            )

        for column in columns:
            if column not in dataframe.columns:
                print(
                    f"WARNING: column {column!r} is absent; "
                    f"available columns: {list(dataframe.columns)}"
                )
                continue

            values = pd.to_numeric(
                dataframe[column],
                errors="coerce"
            ).to_numpy(dtype=float)

            values = values[np.isfinite(values)]

            if values.size > 0:
                values_list.append(values)

    if not values_list:
        raise ValueError(
            f"No finite values were found for columns {columns!r}."
        )

    all_values = np.concatenate(values_list)

    data_min = float(np.min(all_values))
    data_max = float(np.max(all_values))

    robust_min = float(
        np.percentile(all_values, low_percentile)
    )

    robust_max = float(
        np.percentile(all_values, high_percentile)
    )

    robust_width = robust_max - robust_min

    if not np.isfinite(robust_width) or robust_width <= 0.0:
        center = float(np.median(all_values))

        half_width = max(
            float(minimum_half_width),
            0.05 * abs(center),
            1.0e-6,
        )

        robust_min = center - half_width
        robust_max = center + half_width
        robust_width = robust_max - robust_min

    padding = max(
        float(padding_fraction) * robust_width,
        0.5 * float(minimum_half_width),
        1.0e-6,
    )

    grid_min = robust_min - padding
    grid_max = robust_max + padding

    if grid_max <= grid_min:
        center = 0.5 * (robust_min + robust_max)

        half_width = max(
            float(minimum_half_width),
            0.5 * robust_width,
            1.0e-6,
        )

        grid_min = center - half_width
        grid_max = center + half_width

    grid = np.linspace(
        grid_min,
        grid_max,
        int(n_points),
        dtype=float,
    )

    dx = float(grid[1] - grid[0])

    n_outside = int(
        np.count_nonzero(
            (all_values < grid_min) |
            (all_values > grid_max)
        )
    )

    fraction_outside = n_outside / all_values.size

    robust_sigma = float(
        np.std(
            all_values[
                (all_values >= robust_min) &
                (all_values <= robust_max)
            ]
        )
    )

    if verbose:
        print("\n--- Global convolution/deconvolution grid ---")
        print(f"Columns:                   {columns}")
        print(f"Absolute data range:       [{data_min:.6f}, {data_max:.6f}]")
        print(
            f"Robust percentile range:   "
            f"[{robust_min:.6f}, {robust_max:.6f}]"
        )
        print(f"Final numerical range:     [{grid_min:.6f}, {grid_max:.6f}]")
        print(f"Full width:                {grid_max - grid_min:.6f}")
        print(f"N points:                  {grid.size}")
        print(f"dx:                        {dx:.8f}")
        print(f"Values outside grid:       {n_outside}/{all_values.size}")
        print(f"Outside fraction:          {100.0 * fraction_outside:.6f}%")
        print(f"Approx. robust sigma:      {robust_sigma:.6f}")
        print(
            f"Approx. sigma/dx:          "
            f"{robust_sigma / max(dx, 1.0e-30):.2f}"
        )

    return grid


def refine_uniform_grid(grid, maximum_step):
    """
    Mantiene gli estremi della griglia e aumenta il campionamento.
    Restituisce un numero dispari di punti.
    """
    grid = np.asarray(grid, dtype=float)
    maximum_step = float(maximum_step)

    if (
        grid.ndim != 1
        or grid.size < 3
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ValueError(
            "grid deve essere un array finito e crescente."
        )

    if not np.isfinite(maximum_step) or maximum_step <= 0.0:
        raise ValueError("maximum_step deve essere positivo.")

    width = float(grid[-1] - grid[0])

    n_intervals = max(
        grid.size - 1,
        int(np.ceil(width / maximum_step)),
    )

    # Numero pari di intervalli -> numero dispari di punti.
    if n_intervals % 2 != 0:
        n_intervals += 1

    refined_grid = np.linspace(
        grid[0],
        grid[-1],
        n_intervals + 1,
    )

    print("\n--- RAFFINAMENTO GRIGLIA ---")
    print(
        f"Range mantenuto = "
        f"[{refined_grid[0]:.6f}, {refined_grid[-1]:.6f}]"
    )
    print(f"Punti precedenti = {grid.size}")
    print(f"Punti nuovi = {refined_grid.size}")
    print(
        f"Passo precedente = "
        f"{np.mean(np.diff(grid)):.10f}"
    )
    print(
        f"Passo nuovo = "
        f"{np.mean(np.diff(refined_grid)):.10f}"
    )

    return refined_grid

def run_whole(file_vacuum, file_air, constant_parameters):
    """
    Complete vacuum/air analysis pipeline.

    The function performs:

    1. Dataset loading and sequential particle filtering.
    2. Original, filtered and elliptically-masked analysis for vacuum and air.
    3. Histograms, phase-space, Twiss and evolution plots.
    4. Scatterer-statistics extraction.
    5. Direct and inverse reconstruction methods in vacuum.
    6. Air reconstruction with:
       - one unique air block;
       - 1 mm air discretization;
       - 10 mm air discretization.
    7. Equivalent-thickness estimates from MCS and Landau observables.
    8. Scatterer comparison tables and method-comparison figures.
    """

    # ============================================================
    # CONSTANTS
    # ============================================================

    configure_histogram_binning(
        constant_parameters
    )
    
    nRows_value = constant_parameters.get("nRows")

    if nRows_value is None:
        nRows = None
    else:
        nRows = int(nRows_value)

        if nRows <= 0:
            raise ValueError(
                "nRows deve essere un intero positivo oppure None."
            )
    
    ripple_filters = bool(
        constant_parameters["ripple_filters"]
    )

    column_names = constant_parameters[
        "column_names"
    ]

    positions_scoring = list(
        constant_parameters["positions_scoring"]
    )

    tolerance = float(
        constant_parameters["tolerance"]
    )

    rest_energy = float(
        constant_parameters["rest_energy"]
    )

    kinetic_energy = float(
        constant_parameters["kinetic_energy"]
    )

    FWHM_0 = float(
        constant_parameters["FWHM_0"]
    )

    sigma_0_angle = float(
        constant_parameters["sigma_0_angle"]
    )

    sigma_0_theta = float(
        constant_parameters["sigma_0_theta"]
    )

    z0 = float(
        constant_parameters["z0"]
    )

    z = float(
        constant_parameters["z"]
    )

    X0 = float(
        constant_parameters["X0"]
    )

    j = float(
        constant_parameters["j"]
    )

    Z_A_water = float(
        constant_parameters["Z_A_water"]
    )

    K = float(
        constant_parameters["K"]
    )

    I = float(
        constant_parameters["I"]
    )

    rest_energy_electron = float(
        constant_parameters[
            "rest_energy_electron"
        ]
    )

    particle_type = constant_parameters[
        "particle_type"
    ]

    cleaning_cut_mode = constant_parameters.get(
    "cleaning_cut_mode",
        "ellipse",
    )

    theta_percentile_min = constant_parameters.get(
        "theta_percentile_min",
        0.0,
    )

    theta_percentile_max = constant_parameters.get(
        "theta_percentile_max",
        99.0,
    )

    ellipse_peak_fraction = constant_parameters.get(
        "ellipse_peak_fraction",
        0.1,
    )
    angle_percentile_min = constant_parameters.get(
        "angle_percentile_min",
        0.0,
    )
    angle_percentile_max = constant_parameters.get(
        "angle_percentile_max",
        99.0,
    )

    save_plots = bool(
        constant_parameters.get(
            "save_plots",
            False,
        )
    )
    build_interactive_outputs = bool(
        constant_parameters.get(
            "build_interactive_outputs",
            True,
        )
    )

    case_output_dir_value = constant_parameters.get(
        "case_output_dir",
    )

    case_output_dir = (
        Path(case_output_dir_value)
        if case_output_dir_value
        else None
    )
    print("\n" + "=" * 72)
    print("PLOT-SAVING DEBUG INSIDE run_whole")
    print("=" * 72)
    print("save_plots:", save_plots)
    print("case_output_dir_value:", case_output_dir_value)
    print("case_output_dir:", case_output_dir)
    print("=" * 72 + "\n")

    # ============================================================
    # INPUT VALIDATION
    # ============================================================

    if not positions_scoring:
        raise ValueError(
            "positions_scoring cannot be empty."
        )

    if tolerance <= 0.0:
        raise ValueError(
            "tolerance must be positive."
        )

    # ============================================================
    # BEAM AND GEOMETRY PARAMETERS
    # ============================================================

    E_0 = rest_energy + kinetic_energy

    sigma_0_space = FWHM_0 / (
        2.0 * np.sqrt(
            2.0 * np.log(2.0)
        )
    )

    pc = np.sqrt(
        kinetic_energy**2
        + 2.0
        * kinetic_energy
        * rest_energy
    )

    beta = pc / E_0

    gamma = 1.0 / np.sqrt(
        1.0 - beta**2
    )

    positions_scoring_mm = (
        10.0
        * np.asarray(
            positions_scoring,
            dtype=float,
        )
    )

    z_nozzle_start = -925.5
    z_nozzle_end = -705.5

    # ============================================================
    # LOAD DATASETS
    # ============================================================

    setup_vacuum = setup_dataset(
        file_vacuum,
        "Vacuum",
        E_0,
        positions_scoring,
        tolerance,
        column_names,
        nRows=nRows,
    )

    setup_air = setup_dataset(
        file_air,
        "Air",
        E_0,
        positions_scoring,
        tolerance,
        column_names,
        nRows=nRows,
    )

    if setup_vacuum is None:
        raise ValueError(
            "Could not load the vacuum dataset."
        )

    if setup_air is None:
        raise ValueError(
            "Could not load the air dataset."
        )

    df_original_vacuum = setup_vacuum.copy()
    df_original_air = setup_air.copy()

    # ============================================================
    # SEQUENTIAL PARTICLE FILTERING
    # ============================================================

    (
        df_filtered_vacuum,
        idx_vacuum_dropped,
    ) = build_sequential_particles(
        setup_vacuum,
        label="Vacuum",
        positions_scoring=positions_scoring,
    )

    (
        df_filtered_air,
        idx_air_dropped,
    ) = build_sequential_particles(
        setup_air,
        label="Air",
        positions_scoring=positions_scoring,
    )

    print(
        "Indexes dropped rows (Vacuum):",
        idx_vacuum_dropped.tolist()[:20],
        "...",
    )

    print(
        "Indexes dropped rows (Air):",
        idx_air_dropped.tolist()[:20],
        "...",
    )

    # ============================================================
    # ANALYZE VACUUM DATASET
    # ============================================================

    (
        results_original_vacuum,
        results_filtered_vacuum,
        results_masked_vacuum,
        df_filtered_vacuum,
        df_masked_vacuum,
        surface_data_original_vacuum,
        surface_data_filtered_vacuum,
        surface_data_masked_vacuum,
        stored_surface_plots_vacuum,
        stored_distribution_plots_vacuum,
        surface_data_masked_variations_vacuum,
    ) = analyze_dataset(
        df_original_vacuum,
        df_filtered_vacuum,
        "Vacuum",
        positions_scoring,
        ripple_filters=ripple_filters,
        particle_type=particle_type,
        cleaning_cut_mode=cleaning_cut_mode,
        theta_percentile_min=theta_percentile_min,
        theta_percentile_max=theta_percentile_max,
        ellipse_peak_fraction=ellipse_peak_fraction,
        angle_percentile_min=angle_percentile_min,
        angle_percentile_max=angle_percentile_max
    )

    # ============================================================
    # ANALYZE AIR DATASET
    # ============================================================

    (
        results_original_air,
        results_filtered_air,
        results_masked_air,
        df_filtered_air,
        df_masked_air,
        surface_data_original_air,
        surface_data_filtered_air,
        surface_data_masked_air,
        stored_surface_plots_air,
        stored_distribution_plots_air,
        surface_data_masked_variations_air,
    ) = analyze_dataset(
        df_original_air,
        df_filtered_air,
        "Air",
        positions_scoring,
        ripple_filters=ripple_filters,
        particle_type=particle_type,
        cleaning_cut_mode=cleaning_cut_mode,
        theta_percentile_min=theta_percentile_min,
        theta_percentile_max=theta_percentile_max,
        ellipse_peak_fraction=ellipse_peak_fraction,
        angle_percentile_min=angle_percentile_min,
        angle_percentile_max=angle_percentile_max
    )

    if save_plots and case_output_dir is not None:
        surface_plot_dir = (
            case_output_dir
            / "surface_plots"
        )

        surface_plot_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
    # ============================================================
    # KDE-BASED ELLIPTICAL-CUT CONTROL
    #
    # This is a separate diagnostic study. It does not replace
    # results_original, results_filtered or results_masked.
    #
    # The function starts from the original datasets, creates
    # one elliptical selection for each KDE density level, and
    # computes beam_parameters on every selected subset.
    # ============================================================


    analysis_dataframes = [
        df_original_vacuum,
        df_original_air,
        df_filtered_vacuum,
        df_filtered_air,
    ]

    xbin = make_global_analysis_grid(
        analysis_dataframes,
        columns=(
            "x - coordinate",
            "y - coordinate",
        ),
        n_points=4097,
        low_percentile=0.1,
        high_percentile=99.9,
        padding_fraction=0.10,
        minimum_half_width=1.0,
    )

    xbin = refine_uniform_grid(
        xbin,
        maximum_step=40.0 / 2047.0,
    )

    dx_previous = float(
        np.mean(np.diff(xbin))
    )

    half_width = float(
        np.max(np.abs(xbin))
    )

    n_half = int(
        np.ceil(half_width / dx_previous)
    )

    dx_centered = half_width / n_half

    xbin = (
        np.arange(-n_half, n_half + 1, dtype=float)
        * dx_centered
    )

    print("\n[WIENER GRID]")
    print(f"Range: [{xbin[0]:.8f}, {xbin[-1]:.8f}] mm")
    print(f"N punti: {xbin.size}")
    print(f"Passo: {dx_centered:.10f} mm")
    print(f"Coordinata centrale: {xbin[n_half]:.10f} mm")

    xpbin = make_global_analysis_grid(
        analysis_dataframes,
        columns=(
            "x_prime",
            "y_prime",
        ),
        n_points=4097,
        low_percentile=0.1,
        high_percentile=99.9,
        padding_fraction=0.10,
        minimum_half_width=0.1,
    )

    # Test diagnostici separati dall'analisi dei dati reali.
    wiener_synthetic_test = None
    synthetic_test_rows = []

    dx_test = float(np.mean(np.diff(xbin)))
    n_test = len(xbin)

    if n_test % 2 == 0:
        raise ValueError(
            "Il confronto sintetico richiede un numero dispari di punti."
        )

    xbin_centered_test = (
        np.arange(n_test) - n_test // 2
    ) * dx_test

    print("\n" + "=" * 78)
    print("CONFRONTO WIENER: GRIGLIA ORIGINALE E CENTRATA")
    print("=" * 78)
    print(
        f"Centro griglia originale = "
        f"{xbin[n_test // 2]:.8f} mm"
    )
    print(
        f"Centro griglia centrata = "
        f"{xbin_centered_test[n_test // 2]:.8f} mm"
    )


    print("\n" + "=" * 78)
    print("TEST ESPLICITO DELLA FUNZIONE SINTETICA")
    print("=" * 78)

    for kernel_sigma in (7.794, 16.301):
        print(f"\nKernel = {kernel_sigma:.3f} mm")

        run_wiener_synthetic_test(
            xgrid=xbin_centered_test,
            sigma_intrinsic=6.30,
            sigma_kernel=kernel_sigma,
            lambd=1.0e-10,
            make_plot=False,
        )
    lambda_values = (
        1e-8,
        1e-10,
        1e-12,
        1e-14,
        1e-16,
        1e-18,
    )
    for grid_name, test_grid in (
        ("Originale", xbin),
        ("Centrata", xbin_centered_test),
    ):
        for kernel_sigma in (7.794, 16.301):
            for lambda_test in lambda_values:

                pdf_true = normalize_pdf(
                    np.exp(-0.5 * (test_grid / 6.30) ** 2),
                    test_grid,
                )

                pdf_kernel = normalize_pdf(
                    np.exp(
                        -0.5 * (test_grid / kernel_sigma) ** 2
                    ),
                    test_grid,
                )

                pdf_measured = reconvolve_same_grid(
                    pdf_true,
                    pdf_kernel,
                    test_grid,
                )

                pdf_recovered, raw_info = (
                    wiener_deconvolution_fixed_lambda(
                        signal=pdf_measured,
                        kernel=pdf_kernel,
                        xgrid=test_grid,
                        lambd=lambda_test,
                        return_raw=True,
                    )
                )

                pdf_raw = raw_info["raw_pdf"]

                raw_mean = np.trapezoid(
                    test_grid * pdf_raw,
                    test_grid,
                )

                raw_second_moment = np.trapezoid(
                    (test_grid - raw_mean) ** 2 * pdf_raw,
                    test_grid,
                )

                pdf_reconvolved = reconvolve_same_grid(
                    pdf_recovered,
                    pdf_kernel,
                    test_grid,
                )

                sigma_recovered = measure_sigma(
                    test_grid,
                    pdf_recovered,
                    method="fit",
                    fit_window=None,
                )
                sigma_recovered_moment = measure_sigma(
                    test_grid,
                    pdf_recovered,
                    method="moment",
                    fit_window=None,
                )

                sigma_reconvolved = measure_sigma(
                    test_grid,
                    pdf_reconvolved,
                    method="fit",
                    fit_window=None,
                )

                forward_error = relative_forward_error(
                    measured_pdf=pdf_measured,
                    reconvolved_pdf=pdf_reconvolved,
                    xgrid=test_grid,
                )

                sigma_relative_error = abs(
                    sigma_recovered / 6.30 - 1.0
                )

                synthetic_test_rows.append({
                    "Griglia": grid_name,
                    "Kernel [mm]": kernel_sigma,
                    "Lambda": lambda_test,
                    "Momento raw [mm2]": raw_second_moment,
                    "Massa negativa raw": raw_info["negative_mass"],
                    "Sigma fit [mm]": sigma_recovered,
                    "Sigma momento [mm]": sigma_recovered_moment,
                    "Sigma riconvoluta [mm]": sigma_reconvolved,
                    "Errore forward": forward_error,
                    "Errore sigma [%]": 100.0 * sigma_relative_error,
                    "Sigma entro 1%": sigma_relative_error < 0.01,
                })

    synthetic_test_table = pd.DataFrame(synthetic_test_rows)

    print("\nRISULTATI DEL CONFRONTO SINTETICO")
    print(
        synthetic_test_table.to_string(
            index=False,
            float_format=lambda value: f"{value:.8g}",
        )
    )
    print("=" * 78)


    surface_slider_vacuum_hist = None
    surface_slider_air_hist = None
    surface_slider_vacuum_distr = None
    surface_slider_air_distr = None


    if build_interactive_outputs:

        surface_slider_vacuum_hist = (
            show_surface_plots_slider(
                stored_surface_plots_vacuum,
                surface_data_masked_vacuum,
                surface_data_masked_vacuum,
                surface_data_masked_variations_vacuum,
                ripple_filters=ripple_filters,
                particle_type=particle_type,
                positions_scoring=positions_scoring,
                show=False,
            )
        )

        surface_slider_air_hist = (
            show_surface_plots_slider(
                stored_surface_plots_air,
                surface_data_masked_air,
                surface_data_masked_air,
                surface_data_masked_variations_air,
                ripple_filters=ripple_filters,
                particle_type=particle_type,
                positions_scoring=positions_scoring,
                show=False,
            )
        )

        # ============================================================
        # DISTRIBUTION, PHASE-SPACE AND TWISS PLOTS
        # ============================================================

        surface_slider_vacuum_distr = (
            show_distribution_slider(
                stored_distribution_plots_vacuum,
                results_masked_vacuum,
                show=False,
            )
        )

        surface_slider_air_distr = (
            show_distribution_slider(
                stored_distribution_plots_air,
                results_masked_air,
                show=False,
            )
        )


    evolution_plot_vacuum = plot_evolution_fit(
        results_original_vacuum,
        results_filtered_vacuum,
        results_masked_vacuum,
        show=False,
    )

    evolution_plot_air = plot_evolution_fit(
        results_original_air,
        results_filtered_air,
        results_masked_air,
        show=False,
    )


    plot_comparisons = plot_comparison(
        results_masked_vacuum,
        results_masked_air,
        results_original_vacuum,
        results_original_air,
        show=False,
    )

    # ============================================================
    # SCATTERER STATISTICS
    # ============================================================

    dataframe_std = generate_comparison_dataframe(
        results_masked_vacuum,
        results_masked_air,
        positions_scoring,
    )

    stats_angle = compute_all_stats(
        dataframe_std,
        ripple_filters=ripple_filters,
    )

    # ============================================================
    # VACUUM DIRECT/INVERSE RECONSTRUCTION METHODS
    # ============================================================

    (
        scatterers_comparison,
        direct_methods_comparison,
        inverse_methods_comparison,
        sigma_meas_xp_vacuum,
        sigma_meas_yp_vacuum,
        sigma_meas_x_vacuum,
        sigma_meas_y_vacuum,
        sigma_meas_theta_vacuum,
        sigma_meas_xp_air,
        sigma_meas_yp_air,
        sigma_meas_x_air,
        sigma_meas_y_air,
        sigma_meas_theta_air,
    ) = run_analysis(
        results_masked_vacuum,
        results_masked_air,
        df_masked_vacuum,
        stats_angle,
        xbin,
        xpbin,
        z0,
        sigma_0_angle,
        sigma_0_space,
        sigma_0_theta,
        z_nozzle_start,
        z_nozzle_end,
        ripple_filters=ripple_filters,
    )

    # ============================================================
    # MASKED-DATA SUMMARY
    # ============================================================

    masked_summary = pd.DataFrame(
        {
            "z [cm]": positions_scoring,

            "STD(xp) Vacuum": [
                r["sigma_xp"]
                for r in results_masked_vacuum
            ],

            "STD(yp) Vacuum": [
                r["sigma_yp"]
                for r in results_masked_vacuum
            ],

            "STD(xp) Air": [
                r["sigma_xp"]
                for r in results_masked_air
            ],

            "STD(yp) Air": [
                r["sigma_yp"]
                for r in results_masked_air
            ],

            "STD(x) Vacuum": [
                r["sigma_x"]
                for r in results_masked_vacuum
            ],

            "STD(y) Vacuum": [
                r["sigma_y"]
                for r in results_masked_vacuum
            ],

            "STD(x) Air": [
                r["sigma_x"]
                for r in results_masked_air
            ],

            "STD(y) Air": [
                r["sigma_y"]
                for r in results_masked_air
            ],

            "STD(delta energy) Vacuum": [
                r["sigma_delta_energy"]
                for r in results_masked_vacuum
            ],

            "STD(delta energy) Air": [
                r["sigma_delta_energy"]
                for r in results_masked_air
            ],

            "STD(theta) Vacuum": [
                r["sigma_theta"]
                for r in results_masked_vacuum
            ],

            "STD(theta) Air": [
                r["sigma_theta"]
                for r in results_masked_air
            ],

            "STD(delta xp) Vacuum": [
                r["sigma_var_xp"]
                for r in results_masked_vacuum
            ],

            "STD(delta yp) Vacuum": [
                r["sigma_var_yp"]
                for r in results_masked_vacuum
            ],

            "STD(delta xp) Air": [
                r["sigma_var_xp"]
                for r in results_masked_air
            ],

            "STD(delta yp) Air": [
                r["sigma_var_yp"]
                for r in results_masked_air
            ],

            "MPV(mode energy) Vacuum": [
                r["mpv"]
                for r in results_masked_vacuum
            ],

            "MPV(mode energy) Air": [
                r["mpv"]
                for r in results_masked_air
            ],

            "STD(variation delta energy) Vacuum": [
                r["sigma_var_delta_energy"]
                for r in results_masked_vacuum
            ],

            "STD(variation delta energy) Air": [
                r["sigma_var_delta_energy"]
                for r in results_masked_air
            ],

            "STD(variation theta) Vacuum": [
                r["sigma_var_theta"]
                for r in results_masked_vacuum
            ],

            "STD(variation theta) Air": [
                r["sigma_var_theta"]
                for r in results_masked_air
            ],
        }
    )

    # ============================================================
    # AIR SCATTERING COEFFICIENTS
    # ============================================================

    sigma_air_permm_xp = float(
        stats_angle["xp"]["air_1000"]
    ) / np.sqrt(1000.0)

    sigma_air_permm_yp = float(
        stats_angle["yp"]["air_1000"]
    ) / np.sqrt(1000.0)

    # ============================================================
    # AIR ANALYSIS: UNIQUE CONTINUOUS BLOCK
    # ============================================================

    (
        direct_methods_comparison_withair,
        direct_methods_comparison_withair_angle,
        inverse_methods_comparison_withair,
        sigma_scatter_attr_withair,
    ) = run_analysis_with_air_unique_block(
        stats_angle,
        sigma_air_permm_xp,
        sigma_air_permm_yp,
        xbin,
        xpbin,
        z0,
        sigma_0_space,
        sigma_0_angle,
        sigma_meas_xp_air,
        sigma_meas_yp_air,
        sigma_meas_x_air,
        sigma_meas_y_air,
        positions_scoring_mm,
        results_masked_air,
        ripple_filters=ripple_filters,
    )

    # ============================================================
    # AIR ANALYSIS: 1 mm AND 10 mm DISCRETIZATION
    # ============================================================

    (
        direct_methods_comparison_withair_total,
        direct_methods_comparison_withair_angle_total,
        inverse_methods_comparison_withair_total,
        comparison_convolution,
    ) = run_analysis_with_air_small_components(
        positions_scoring_mm,
        sigma_air_permm_xp,
        sigma_air_permm_yp,
        stats_angle,
        sigma_scatter_attr_withair,
        sigma_0_space,
        sigma_0_angle,
        sigma_meas_xp_air,
        sigma_meas_yp_air,
        sigma_meas_x_air,
        sigma_meas_y_air,
        xbin,
        xpbin,
        z0,
        direct_methods_comparison_withair,
        direct_methods_comparison_withair_angle,
        ripple_filters=ripple_filters,
    )

    # ============================================================
    # GLOBAL COMPARISON PLOT
    # ============================================================

    plot_comparisons_of_dataframes = (
        plot_comparison_of_dataframes(
            direct_methods_comparison,
            inverse_methods_comparison,
            inverse_methods_comparison_withair,
            inverse_methods_comparison_withair_total,
            direct_methods_comparison_withair,
            direct_methods_comparison_withair_total,
        )
    )

    plots_final_comparison_all_methods = plot_final_comparison_all_methods(
            direct_methods_comparison,
            inverse_methods_comparison,
            inverse_methods_comparison_withair,
            inverse_methods_comparison_withair_total,
            direct_methods_comparison_withair,
            direct_methods_comparison_withair_total,
        )

    # ============================================================
    # EQUIVALENT THICKNESS: AIR
    # ============================================================

    air_delta_xp = dataframe_std[
        "STD(delta xp) Air"
    ].to_numpy(dtype=float)

    air_delta_yp = dataframe_std[
        "STD(delta yp) Air"
    ].to_numpy(dtype=float)

    air_theta = dataframe_std[
        "STD (variation theta) Air"
    ].to_numpy(dtype=float)

    MPV_energy_air = dataframe_std[
        "MPV (mode energy) Air"
    ].to_numpy(dtype=float)

    if len(air_delta_xp) < 4:
        raise ValueError(
            "At least four scoring surfaces are required "
            "to extract the 60 cm and 100 cm air contributions."
        )

    delta_xp_air_100cm = float(
        air_delta_xp[3]
    )

    delta_yp_air_100cm = float(
        air_delta_yp[3]
    )

    delta_theta_air_100cm = float(
        air_theta[3]
    )

    delta_energy_air_100cm = float(
        MPV_energy_air[3]
    )

    delta_xp_air_60cm = float(
        air_delta_xp[2]
    )

    delta_yp_air_60cm = float(
        air_delta_yp[2]
    )

    delta_theta_air_60cm = float(
        air_theta[2]
    )

    delta_energy_air_60cm = float(
        MPV_energy_air[2]
    )

    (
        equivalent_x_MCS_air_100cm_xp,
        equivalent_x_MCS_air_100cm_yp,
        equivalent_x_MCS_air_100cm_theta,
        x_equivalent_landau_air_100cm,
    ) = calc_equivalent_x(
        delta_xp_air_100cm,
        delta_yp_air_100cm,
        delta_theta_air_100cm,
        pc,
        beta,
        z,
        X0,
        delta_energy_air_100cm,
        K,
        Z_A_water,
        rest_energy_electron,
        I,
        j,
        gamma,
    )

    (
        equivalent_x_MCS_air_60cm_xp,
        equivalent_x_MCS_air_60cm_yp,
        equivalent_x_MCS_air_60cm_theta,
        x_equivalent_landau_air_60cm,
    ) = calc_equivalent_x(
        delta_xp_air_60cm,
        delta_yp_air_60cm,
        delta_theta_air_60cm,
        pc,
        beta,
        z,
        X0,
        delta_energy_air_60cm,
        K,
        Z_A_water,
        rest_energy_electron,
        I,
        j,
        gamma,
    )

    # ============================================================
    # EQUIVALENT THICKNESS: VACUUM WINDOW AND NOZZLE
    # ============================================================

    MPV_energy_vacuum = dataframe_std[
        "MPV (mode energy) Vacuum"
    ].to_numpy(dtype=float)

    delta_xp_vacuum = dataframe_std[
        "STD(delta xp) Vacuum"
    ].to_numpy(dtype=float)

    delta_yp_vacuum = dataframe_std[
        "STD(delta yp) Vacuum"
    ].to_numpy(dtype=float)

    delta_theta_vacuum = dataframe_std[
        "STD (variation theta) Vacuum"
    ].to_numpy(dtype=float)

    delta_energy_window = float(
        MPV_energy_vacuum[0]
    )

    window_angle_std_xp = float(
        delta_xp_vacuum[0]
    )

    window_angle_std_yp = float(
        delta_yp_vacuum[0]
    )

    window_angle_std_theta = float(
        delta_theta_vacuum[0]
    )

    if ripple_filters:

        scatterer_nozzle_name = (
            "Nozzle + ripple filters"
        )

        delta_energy_nozzle = float(
            MPV_energy_vacuum[1]
        )

        nozzle_angle_std_xp = float(
            delta_xp_vacuum[1]
        )

        nozzle_angle_std_yp = float(
            delta_yp_vacuum[1]
        )

        nozzle_angle_std_theta = float(
            delta_theta_vacuum[1]
        )

    else:

        scatterer_nozzle_name = "Nozzle"

        delta_energy_nozzle = float(
            MPV_energy_vacuum[1]
        )

        nozzle_angle_std_xp = float(
            delta_xp_vacuum[1]
        )

        nozzle_angle_std_yp = float(
            delta_yp_vacuum[1]
        )

        # Mantiene la convenzione della vecchia run_wholei:
        # senza ripple filter, la variazione theta del nozzle
        # viene associata alla prima superficie.
        nozzle_angle_std_theta = float(
            delta_theta_vacuum[1]
        )

    (
        equivalent_x_MCS_window_xp,
        equivalent_x_MCS_window_yp,
        equivalent_x_MCS_window_theta,
        x_equivalent_landau_window,
    ) = calc_equivalent_x(
        window_angle_std_xp,
        window_angle_std_yp,
        window_angle_std_theta,
        pc,
        beta,
        z,
        X0,
        delta_energy_window,
        K,
        Z_A_water,
        rest_energy_electron,
        I,
        j,
        gamma,
    )

    (
        equivalent_x_MCS_nozzle_xp,
        equivalent_x_MCS_nozzle_yp,
        equivalent_x_MCS_nozzle_theta,
        x_equivalent_landau_nozzle,
    ) = calc_equivalent_x(
        nozzle_angle_std_xp,
        nozzle_angle_std_yp,
        nozzle_angle_std_theta,
        pc,
        beta,
        z,
        X0,
        delta_energy_nozzle,
        K,
        Z_A_water,
        rest_energy_electron,
        I,
        j,
        gamma,
    )

    # ============================================================
    # FINAL SCATTERER TABLE
    # ============================================================

    export_table_scatterers = pd.DataFrame(
        {
            "Scatterers": [
                "VacWin",
                scatterer_nozzle_name,
                "100cmAir",
                "60cmAir",
            ],

            "Contribution in x' [mrad]": [
                window_angle_std_xp,
                nozzle_angle_std_xp,
                delta_xp_air_100cm,
                delta_xp_air_60cm,
            ],

            "Contribution in y' [mrad]": [
                window_angle_std_yp,
                nozzle_angle_std_yp,
                delta_yp_air_100cm,
                delta_yp_air_60cm,
            ],

            "Contribution in theta [mrad]": [
                window_angle_std_theta,
                nozzle_angle_std_theta,
                delta_theta_air_100cm,
                delta_theta_air_60cm,
            ],

            "Contribution in energy [MeV]": [
                delta_energy_window,
                delta_energy_nozzle,
                delta_energy_air_100cm,
                delta_energy_air_60cm,
            ],

            "Equivalent x (MCS) in x' [mm]": [
                equivalent_x_MCS_window_xp,
                equivalent_x_MCS_nozzle_xp,
                equivalent_x_MCS_air_100cm_xp,
                equivalent_x_MCS_air_60cm_xp,
            ],

            "Equivalent x (MCS) in y' [mm]": [
                equivalent_x_MCS_window_yp,
                equivalent_x_MCS_nozzle_yp,
                equivalent_x_MCS_air_100cm_yp,
                equivalent_x_MCS_air_60cm_yp,
            ],

            "Equivalent x (MCS) in theta [mm]": [
                equivalent_x_MCS_window_theta,
                equivalent_x_MCS_nozzle_theta,
                equivalent_x_MCS_air_100cm_theta,
                equivalent_x_MCS_air_60cm_theta,
            ],

            "Equivalent x (Landau) in energy [mm]": [
                x_equivalent_landau_window,
                x_equivalent_landau_nozzle,
                x_equivalent_landau_air_100cm,
                x_equivalent_landau_air_60cm,
            ],
        }
    )

    # ============================================================
    # CONSOLE SUMMARY
    # ============================================================

    print("\n" + "=" * 72)
    print("EQUIVALENT SCATTERER THICKNESS SUMMARY")
    print("=" * 72)
    print(export_table_scatterers.to_string(index=False))
    print("=" * 72 + "\n")

    # ============================================================
    # RETURN RESULTS FOR STREAMLIT
    # ============================================================

    result = {
        "results_original_vacuum": (
            results_original_vacuum
        ),

        "results_filtered_vacuum": (
            results_filtered_vacuum
        ),

        "results_masked_vacuum": (
            results_masked_vacuum
        ),

        "results_original_air": (
            results_original_air
        ),

        "results_filtered_air": (
            results_filtered_air
        ),

        "results_masked_air": (
            results_masked_air
        ),

        "df_filtered_vacuum": (
            df_filtered_vacuum
        ),

        "df_masked_vacuum": (
            df_masked_vacuum
        ),

        "df_filtered_air": (
            df_filtered_air
        ),

        "df_masked_air": (
            df_masked_air
        ),

        "surface_data_original_vacuum": (
            surface_data_original_vacuum
        ),

        "surface_data_filtered_vacuum": (
            surface_data_filtered_vacuum
        ),

        "surface_data_masked_vacuum": (
            surface_data_masked_vacuum
        ),

        "surface_data_masked_variations_vacuum": (
            surface_data_masked_variations_vacuum
        ),

        "surface_data_original_air": (
            surface_data_original_air
        ),

        "surface_data_filtered_air": (
            surface_data_filtered_air
        ),

        "surface_data_masked_air": (
            surface_data_masked_air
        ),

        "surface_data_masked_variations_air": (
            surface_data_masked_variations_air
        ),

        "surface_slider_vacuum_hist": (
            surface_slider_vacuum_hist
        ),

        "surface_slider_air_hist": (
            surface_slider_air_hist
        ),

        "surface_slider_vacuum_distr": (
            surface_slider_vacuum_distr
        ),

        "surface_slider_air_distr": (
            surface_slider_air_distr
        ),


        "evolution_plot_vacuum": (
            evolution_plot_vacuum
        ),

        "evolution_plot_air": (
            evolution_plot_air
        ),


        "plot_comparisons": (
            plot_comparisons
        ),


        "plot_comparisons_of_dataframes": (
            plot_comparisons_of_dataframes
        ),

        "plots_final_comparison_all_methods": (
            plots_final_comparison_all_methods
        ),

        "dataframe_std": dataframe_std,

        "masked_summary": masked_summary,

        "stats_angle": stats_angle,

        "scatterers_comparison": (
            scatterers_comparison
        ),

        "export_table_scatterers": (
            export_table_scatterers
        ),

        "direct_methods_comparison": (
            direct_methods_comparison
        ),

        "inverse_methods_comparison": (
            inverse_methods_comparison
        ),

        "direct_methods_comparison_withair": (
            direct_methods_comparison_withair
        ),

        "inverse_methods_comparison_withair": (
            inverse_methods_comparison_withair
        ),

        "direct_methods_comparison_withair_angle": (
            direct_methods_comparison_withair_angle
        ),

        "direct_methods_comparison_withair_total": (
            direct_methods_comparison_withair_total
        ),

        "inverse_methods_comparison_withair_total": (
            inverse_methods_comparison_withair_total
        ),

        "direct_methods_comparison_withair_angle_total": (
            direct_methods_comparison_withair_angle_total
        ),

        "comparison_convolution": (
            comparison_convolution
        ),

        "sigma_meas_xp_vacuum": (
            sigma_meas_xp_vacuum
        ),

        "sigma_meas_yp_vacuum": (
            sigma_meas_yp_vacuum
        ),

        "sigma_meas_x_vacuum": (
            sigma_meas_x_vacuum
        ),

        "sigma_meas_y_vacuum": (
            sigma_meas_y_vacuum
        ),

        "sigma_meas_theta_vacuum": (
            sigma_meas_theta_vacuum
        ),

        "sigma_meas_xp_air": (
            sigma_meas_xp_air
        ),

        "sigma_meas_yp_air": (
            sigma_meas_yp_air
        ),

        "sigma_meas_x_air": (
            sigma_meas_x_air
        ),

        "sigma_meas_y_air": (
            sigma_meas_y_air
        ),

        "sigma_meas_theta_air": (
            sigma_meas_theta_air
        ),

        "sigma_air_permm_xp": (
            sigma_air_permm_xp
        ),

        "sigma_air_permm_yp": (
            sigma_air_permm_yp
        ),

        "delta_energy_nozzle_with_or_without_ripple": (
            delta_energy_nozzle
        ),

        "equivalent_thicknesses": {
            "vacuum_window": {
                "mcs_xp_mm": (
                    equivalent_x_MCS_window_xp
                ),

                "mcs_yp_mm": (
                    equivalent_x_MCS_window_yp
                ),

                "mcs_theta_mm": (
                    equivalent_x_MCS_window_theta
                ),

                "landau_energy_mm": (
                    x_equivalent_landau_window
                ),
            },

            "nozzle": {
                "mcs_xp_mm": (
                    equivalent_x_MCS_nozzle_xp
                ),

                "mcs_yp_mm": (
                    equivalent_x_MCS_nozzle_yp
                ),

                "mcs_theta_mm": (
                    equivalent_x_MCS_nozzle_theta
                ),

                "landau_energy_mm": (
                    x_equivalent_landau_nozzle
                ),
            },

            "air_100cm": {
                "mcs_xp_mm": (
                    equivalent_x_MCS_air_100cm_xp
                ),

                "mcs_yp_mm": (
                    equivalent_x_MCS_air_100cm_yp
                ),

                "mcs_theta_mm": (
                    equivalent_x_MCS_air_100cm_theta
                ),

                "landau_energy_mm": (
                    x_equivalent_landau_air_100cm
                ),
            },

            "air_60cm": {
                "mcs_xp_mm": (
                    equivalent_x_MCS_air_60cm_xp
                ),

                "mcs_yp_mm": (
                    equivalent_x_MCS_air_60cm_yp
                ),

                "mcs_theta_mm": (
                    equivalent_x_MCS_air_60cm_theta
                ),

                "landau_energy_mm": (
                    x_equivalent_landau_air_60cm
                ),
            },
        },
    }

    if save_plots and case_output_dir is not None:
        case_output_dir = Path(
            case_output_dir
        )

        surface_plot_dir = (
            case_output_dir
            / "surface_plots"
        )

        surface_plot_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("\n" + "=" * 72)
        print("SAVING CASE OUTPUTS")
        print("=" * 72)
        print(
            "Case output directory:",
            case_output_dir,
        )
        print(
            "Surface plot directory:",
            surface_plot_dir,
        )
        print(
            "Vacuum surfaces available:",
            len(stored_surface_plots_vacuum),
        )
        print(
            "Air surfaces available:",
            len(stored_surface_plots_air),
        )

        print(
            "\nSaving Vacuum histograms and phase-space plots..."
        )

        save_surface_outputs(
            stored_surface_plots=stored_surface_plots_vacuum,
            stored_distribution_plots=stored_distribution_plots_vacuum,
            results_masked=results_masked_vacuum,
            positions_scoring=positions_scoring,
            ripple_filters=ripple_filters,
            particle_type=particle_type,
            medium="vacuum",
            case_output_dir=surface_plot_dir,
        )

        print(
            "Saving Air histograms and phase-space plots..."
        )

        save_surface_outputs(
            stored_surface_plots=stored_surface_plots_air,
            stored_distribution_plots=stored_distribution_plots_air,
            results_masked=results_masked_air,
            positions_scoring=positions_scoring,
            ripple_filters=ripple_filters,
            particle_type=particle_type,
            medium="air",
            case_output_dir=surface_plot_dir,
        )

        print(
            "Saving generic tables and comparison plots..."
        )

        save_generic_outputs(
            result=result,
            case_output_dir=case_output_dir,
            constant_parameters=constant_parameters,
        )

        print(
            "Case plots and tables saved successfully."
        )
        plt.close("all")

    if constant_parameters.get(
        "compact_batch_result",
        False,
    ):
        return {
            "export_table_scatterers": (
                result["export_table_scatterers"]
            )
        }        
    return result


# CODICI PER ANALISI TUTTE ENERGIE

import gzip
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# Particle constants
# ------------------------------------------------------------------
PROTON_REST_MASS_MEV = 938.2720813
CARBON12_REST_MASS_MEV = 12.0 * 931.49410242

PROTON_CHARGE = 1
CARBON12_CHARGE = 6
CARBON12_MASS_NUMBER = 12


@dataclass
class BeamCase:
    particle: str
    range_mm: float
    case_dir: Path
    input_file: Path
    kinetic_energy_mev: float
    rest_energy_mev: float
    charge_z: int
    mass_number: int
    input_energy_mev_u: float | None
    files: dict


def parse_range_from_folder(folder_name: str) -> float:
    """
    Examples:
        p030mm -> 30.0
        c270mm -> 270.0
        P320mm -> 320.0
    """

    match = re.fullmatch(
        r"[pcPC](\d+(?:\.\d+)?)mm",
        folder_name.strip(),
    )

    if match is None:
        raise ValueError(
            f"Cannot infer range from folder name: {folder_name!r}. "
            "Expected p***mm or c***mm."
        )

    return float(match.group(1))


def particle_from_folder(folder_name: str) -> str:
    """
    Infer particle family from p***mm or c***mm folder name.
    """

    name = folder_name.strip().lower()

    if name.startswith("p"):
        return "proton"

    if name.startswith("c"):
        return "carbon"

    raise ValueError(
        f"Cannot infer particle from folder name: {folder_name!r}."
    )


def find_input_file(case_dir: Path) -> Path:
    """
    Find exactly one FLUKA input file in one energy folder.
    """

    input_files = sorted(case_dir.glob("*.inp"))

    if len(input_files) == 0:
        raise FileNotFoundError(
            f"No .inp file found in {case_dir}."
        )

    if len(input_files) > 1:
        raise RuntimeError(
            f"More than one .inp file found in {case_dir}: "
            f"{[path.name for path in input_files]}"
        )

    return input_files[0]


def strip_fluka_comment(line: str) -> str:
    """
    Remove common FLUKA comment portions before parsing.
    """

    return line.split("*", maxsplit=1)[0].strip()


def _parse_fluka_beam_cards(input_path):
    """
    Extract beam kinetic energy and ion properties from a FLUKA input file.

    Supported input pattern for the carbon files:
        BEAM      -0.11523   -1.50000   -1.50000
        HEAVYION  HI-PROPE   6.         12.

    Returns
    -------
    dict with:
        beam_value_gev : float
        heavyion_z     : int or None
        heavyion_a     : int or None
    """

    input_path = Path(input_path)

    beam_value_gev = None
    heavyion_z = None
    heavyion_a = None

    with input_path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as handle:

        for raw_line in handle:

            stripped_line = raw_line.strip()

            if not stripped_line:
                continue

            # A FLUKA comment line can start with *.
            if stripped_line.startswith("*"):
                continue

            # Remove inline comments, if any.
            line = stripped_line.split(
                "*",
                maxsplit=1,
            )[0].strip()

            if not line:
                continue

            tokens = line.split()

            if not tokens:
                continue

            card = tokens[0].upper()

            # Example:
            # BEAM -0.11523 -1.50000 -1.50000
            if card == "BEAM" and beam_value_gev is None:

                for token in tokens[1:]:
                    try:
                        beam_value_gev = float(token)
                        break
                    except ValueError:
                        continue

            # Example:
            # HEAVYION HI-PROPE 6. 12.
            elif card == "HEAVYION":

                numeric_values = []

                for token in tokens[1:]:
                    try:
                        numeric_values.append(
                            float(token)
                        )
                    except ValueError:
                        continue

                if len(numeric_values) >= 2:
                    heavyion_z = int(
                        round(numeric_values[-2])
                    )

                    heavyion_a = int(
                        round(numeric_values[-1])
                    )

    if beam_value_gev is None:
        raise ValueError(
            f"No numeric BEAM card found in {input_path}."
        )

    return {
        "beam_value_gev": beam_value_gev,
        "heavyion_z": heavyion_z,
        "heavyion_a": heavyion_a,
    }


def _beam_parameters_from_input(
    input_path,
    particle,
):
    """
    Read the BEAM card from a FLUKA input and create the parameters
    required by the batch analysis.

    Dataset convention used here:
    - proton folders contain proton simulations;
    - carbon folders contain 12C simulations (Z=6, A=12).
    """

    input_path = Path(input_path)

    particle = str(particle).strip().lower()

    beam_value_gev = None

    with input_path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as handle:

        for raw_line in handle:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("*"):
                continue

            tokens = line.split()

            if not tokens:
                continue

            if tokens[0].upper() != "BEAM":
                continue

            for token in tokens[1:]:
                try:
                    beam_value_gev = float(token)
                    break
                except ValueError:
                    continue

            if beam_value_gev is not None:
                break

    if beam_value_gev is None:
        raise ValueError(
            f"No numeric BEAM card found in {input_path}."
        )

    beam_energy_mev = abs(
        float(beam_value_gev)
    ) * 1000.0

    if particle == "proton":
        return {
            "rest_energy": 938.2720813,
            "kinetic_energy": beam_energy_mev,
            "z": 1.0,
            "mass_number": 1,
            "energy_per_u_mev": None,
        }

    if particle == "carbon":
        # All datasets under the CARBON root are 12C.
        z = 6
        a = 12

        return {
            "rest_energy": a * 931.49410242,
            "kinetic_energy": a * beam_energy_mev,
            "z": float(z),
            "mass_number": int(a),
            "energy_per_u_mev": beam_energy_mev,
        }

    raise ValueError(
        f"Unsupported particle type: {particle!r}."
    )

def decompress_gz_file(gz_path: Path, overwrite: bool = False) -> Path:
    """
    Decompress file.gz to file, keeping the original .gz.

    Example:
        output_001.23.gz -> output_001.23
    """

    if gz_path.suffix.lower() != ".gz":
        raise ValueError(
            f"Expected a .gz file, received: {gz_path}"
        )

    output_path = gz_path.with_suffix("")

    if output_path.exists() and not overwrite:
        return output_path

    with gzip.open(gz_path, "rb") as source:
        with output_path.open("wb") as target:
            shutil.copyfileobj(source, target)

    return output_path



def _classify_input_file(input_path, particle):
    """
    Infer (medium, has_ripple_filter) from an input filename.

    Conventions
    -----------
    Protons:
        every case is considered WITHOUT ripple filters, irrespective
        of the filename.

    Carbon:
        filename containing 'noRippFilters' -> WITHOUT ripple filters;
        filename not containing it          -> WITH ripple filters.
    """

    name = input_path.name.lower()

    if "air" in name:
        medium = "air"

    elif "vac" in name:
        medium = "vacuum"

    else:
        raise ValueError(
            f"Cannot determine medium from input filename: "
            f"{input_path.name}"
        )

    if particle == "proton":
        has_ripple_filter = False

    elif particle == "carbon":
        no_ripple_keywords = (
            "norippfilters",
            "noripplefilters",
            "no_ripple",
            "no-ripple",
        )

        has_no_ripple_keyword = any(
            keyword in name
            for keyword in no_ripple_keywords
        )

        has_ripple_filter = not has_no_ripple_keyword

    else:
        raise ValueError(
            f"Unsupported particle type: {particle!r}"
        )

    return medium, has_ripple_filter

def _find_input_files(case_dir, particle):
    """
    Return the input files indexed as:

        vacuum_no_rf
        air_no_rf
        vacuum_rf
        air_rf

    Proton files are always classified as no-ripple-filter files.
    Carbon files follow their filename convention:
    'noRippFilters' means no ripple filters.
    """

    input_files = sorted(case_dir.glob("*.inp"))

    if not input_files:
        raise FileNotFoundError(
            f"No .inp files found in {case_dir}."
        )

    input_map = {}

    for input_path in input_files:
        medium, has_ripple_filter = _classify_input_file(
            input_path,
            particle,
        )

        key = (
            f"{medium}_rf"
            if has_ripple_filter
            else f"{medium}_no_rf"
        )

        if key in input_map:
            raise RuntimeError(
                f"More than one .inp file classified as '{key}' "
                f"in {case_dir}: "
                f"{input_map[key].name} and {input_path.name}"
            )

        input_map[key] = input_path

    return input_map
def _classify_output_file(output_path, particle):
    """
    Infer (medium, has_ripple_filter) from an output file name.
    """

    name = output_path.name.lower()

    if "air" in name:
        medium = "air"

    elif "vac" in name:
        medium = "vacuum"

    else:
        raise ValueError(
            f"Cannot determine medium from output filename: "
            f"{output_path.name}"
        )

    if particle == "proton":
        has_ripple_filter = False

    elif particle == "carbon":
        has_ripple_filter = (
            "norippfilters" not in name
            and "noripplefilters" not in name
            and "no_ripple" not in name
            and "no-ripple" not in name
        )

    else:
        raise ValueError(
            f"Unsupported particle type: {particle!r}"
        )

    return medium, has_ripple_filter

def find_output_files(case_dir: Path, particle: str) -> dict:
    """
    Find .23 and .23.gz outputs, decompress .gz if needed, then map them.

    Returned keys:
        vacuum_no_rf
        vacuum_rf
        air_no_rf
        air_rf

    Proton folders are expected to contain at least:
        vacuum_no_rf
        air_no_rf

    Carbon folders are expected to contain:
        vacuum_no_rf
        vacuum_rf
        air_no_rf
        air_rf
    """

    output_files = sorted(
        list(case_dir.glob("*.23"))
        + list(case_dir.glob("*.23.gz"))
    )

    if not output_files:
        raise FileNotFoundError(
            f"No .23 or .23.gz output files found in {case_dir}."
        )

    mapping = {}

    for path in output_files:
        if path.suffix.lower() == ".gz":
            path = decompress_gz_file(path)

        medium, ripple_filter = _classify_output_file(path, particle)

        key = f"{medium}_{'rf' if ripple_filter else 'no_rf'}"

        if key in mapping:
            raise RuntimeError(
                f"More than one file classified as {key} in {case_dir}: "
                f"{mapping[key].name}, {path.name}"
            )

        mapping[key] = path

    return mapping


def discover_beam_cases(root_dir: str | Path) -> list[BeamCase]:
    """
    Discover all p***mm and c***mm folders inside:
        root_dir/PROTON/
        root_dir/CARBON/
    """

    root_dir = Path(root_dir)

    particle_roots = {
        "proton": root_dir / "PROTON",
        "carbon": root_dir / "CARBON",
    }

    cases = []

    for expected_particle, particle_root in particle_roots.items():
        if not particle_root.exists():
            print(
                f"WARNING: particle directory not found: {particle_root}"
            )
            continue

        for case_dir in sorted(particle_root.iterdir()):
            if not case_dir.is_dir():
                continue

            try:
                inferred_particle = particle_from_folder(case_dir.name)
                range_mm = parse_range_from_folder(case_dir.name)
            except ValueError:
                continue

            if inferred_particle != expected_particle:
                print(
                    f"WARNING: folder {case_dir.name} is inside "
                    f"{particle_root.name}, but its prefix indicates "
                    f"{inferred_particle}."
                )

            input_file = find_input_file(case_dir)
            beam_parameters = _beam_parameters_from_input(
                inferred_particle,
                input_file,
            )

            output_files = find_output_files(case_dir)

            case = BeamCase(
                particle=inferred_particle,
                range_mm=range_mm,
                case_dir=case_dir,
                input_file=input_file,
                kinetic_energy_mev=beam_parameters[
                    "kinetic_energy_mev"
                ],
                rest_energy_mev=beam_parameters[
                    "rest_energy_mev"
                ],
                charge_z=beam_parameters["charge_z"],
                mass_number=beam_parameters["mass_number"],
                input_energy_mev_u=beam_parameters[
                    "input_energy_mev_u"
                ],
                files=output_files,
            )

            cases.append(case)

    cases.sort(
        key=lambda case: (
            case.particle,
            case.range_mm,
        )
    )

    return cases

def normalize_scatterer_name(name: str) -> str:
    """
    Standardise row labels across all cases.
    """

    normalized = str(name).strip().lower()

    if "vac" in normalized or "win" in normalized:
        return "VW"

    if "nozzle" in normalized:
        return "Nozzle + ripple" if "ripple" in normalized else "Nozzle"

    if "100" in normalized and "air" in normalized:
        return "100 cm air"

    if "60" in normalized and "air" in normalized:
        return "60 cm air"

    return str(name).strip()


def analyse_one_case(
    case: BeamCase,
    analysis_function,
    base_parameters: dict,
) -> pd.DataFrame:
    """
    Analyse every available configuration of one energy folder.

    analysis_function must return export_table_scatterers.
    """

    configurations = []

    if case.particle == "proton":
        configurations = [
            ("No ripple filter", "vacuum_no_rf", "air_no_rf"),
        ]

    elif case.particle == "carbon":
        configurations = [
            ("No ripple filter", "vacuum_no_rf", "air_no_rf"),
            ("With ripple filter", "vacuum_rf", "air_rf"),
        ]

    output_rows = []

    for configuration_name, vacuum_key, air_key in configurations:
        if vacuum_key not in case.files or air_key not in case.files:
            print(
                f"WARNING: skipping {case.case_dir.name} | "
                f"{configuration_name}: missing {vacuum_key} or {air_key}."
            )
            continue

        parameters = dict(base_parameters)

        parameters.update({
            "particle_type": case.particle,
            "rest_energy": case.rest_energy_mev,
            "kinetic_energy": case.kinetic_energy_mev,
            "z": case.charge_z,
            "ripplefilters": (
                configuration_name == "With ripple filter"
            ),
        })

        try:
            scatterer_table = analysis_function(
                case.files[vacuum_key],
                case.files[air_key],
                parameters,
            )
        except Exception as exc:
            print(
                f"ERROR in {case.case_dir.name} | "
                f"{configuration_name}: {exc}"
            )
            continue

        scatterer_table = scatterer_table.copy()

        scatterer_column = "Scatterers"

        if scatterer_column not in scatterer_table.columns:
            raise KeyError(
                f"Expected column {scatterer_column!r} was not found. "
                f"Available columns: {list(scatterer_table.columns)}"
            )

        scatterer_table["Scatterer"] = scatterer_table[
            scatterer_column
        ].map(normalize_scatterer_name)

        scatterer_table["Particle"] = (
            "Protons"
            if case.particle == "proton"
            else "Carbon ions"
        )

        scatterer_table["Range [mm]"] = case.range_mm
        scatterer_table["Configuration"] = configuration_name
        scatterer_table["Kinetic energy [MeV]"] = (
            case.kinetic_energy_mev
        )
        scatterer_table["Energy per nucleon [MeV/u]"] = (
            case.input_energy_mev_u
        )
        scatterer_table["Input file"] = case.input_file.name
        scatterer_table["Case folder"] = case.case_dir.name

        output_rows.append(scatterer_table)

    if not output_rows:
        return pd.DataFrame()

    return pd.concat(
        output_rows,
        ignore_index=True,
    )


def analyse_all_cases(
    root_dir,
    analysis_function,
    base_parameters,
) -> pd.DataFrame:
    """
    Run the scatterer analysis over all proton and carbon folders.
    """

    cases = discover_beam_cases(root_dir)

    all_tables = []

    for index, case in enumerate(cases, start=1):
        print(
            f"\n[{index}/{len(cases)}] "
            f"Analysing {case.case_dir.name} "
            f"({case.particle}, range={case.range_mm:.1f} mm)"
        )

        result_table = analyse_one_case(
            case=case,
            analysis_function=analysis_function,
            base_parameters=base_parameters,
        )

        if not result_table.empty:
            all_tables.append(result_table)

    if not all_tables:
        raise RuntimeError(
            "No dataset was successfully analysed."
        )

    final_table = pd.concat(
        all_tables,
        ignore_index=True,
    )

    final_table = final_table.sort_values(
        by=[
            "Particle",
            "Scatterer",
            "Configuration",
            "Range [mm]",
        ],
    ).reset_index(drop=True)

    return final_table


def plot_scattering_contributions_vs_range(
    df,
    scatterer,
):
    """
    Plot angular and energy scattering contributions versus range
    for one selected scatterer.
    """

    subset = df[df["Scatterer"] == scatterer].copy()

    if subset.empty:
        raise ValueError(
            f"No rows available for scatterer: {scatterer}"
        )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 11),
        sharex=True,
    )

    ax_angular, ax_energy = axes

    styles = {
        ("Carbon ions", "No ripple filter"): {
            "color": "tab:blue",
            "marker": "o",
            "linestyle": "-",
            "label": "Carbon ions — no RF",
        },
        ("Carbon ions", "With ripple filter"): {
            "color": "tab:orange",
            "marker": "s",
            "linestyle": "--",
            "label": "Carbon ions — with RF",
        },
        ("Protons", "No ripple filter"): {
            "color": "tab:green",
            "marker": "^",
            "linestyle": "-.",
            "label": "Protons",
        },
    }

    angular_columns = {
        "contr_xp": r"$x'$",
        "contr_yp": r"$y'$",
        "contr_theta": r"$\theta$",
    }

    angular_line_styles = {
        "contr_xp": "-",
        "contr_yp": "--",
        "contr_theta": ":",
    }

    for (particle, configuration), group in subset.groupby(
        ["Particle", "Configuration"]
    ):
        group = group.sort_values("Range [mm]")

        key = (particle, configuration)

        if key not in styles:
            continue

        style = styles[key]

        for column, observable_label in angular_columns.items():
            ax_angular.plot(
                group["Range [mm]"],
                group[column],
                color=style["color"],
                marker=style["marker"],
                linestyle=angular_line_styles[column],
                linewidth=2.5,
                markersize=7,
                label=(
                    f"{style['label']} — {observable_label}"
                ),
            )

        ax_energy.plot(
            group["Range [mm]"],
            group["contr_energy"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.7,
            markersize=7,
            label=style["label"],
        )

    ax_angular.set_title(
        f"{scatterer} — angular scattering contribution",
        fontsize=16,
        fontweight="bold",
    )

    ax_angular.set_ylabel(
        "Contribution [mrad]",
        fontsize=14,
        fontweight="bold",
    )

    ax_angular.grid(alpha=0.3)
    ax_angular.legend(fontsize=8, ncol=2)

    ax_energy.set_title(
        f"{scatterer} — energy-loss contribution",
        fontsize=16,
        fontweight="bold",
    )

    ax_energy.set_xlabel(
        "Range [mm]",
        fontsize=14,
        fontweight="bold",
    )

    ax_energy.set_ylabel(
        "Contribution [MeV]",
        fontsize=14,
        fontweight="bold",
    )

    ax_energy.grid(alpha=0.3)
    ax_energy.legend(fontsize=10)

    for ax in axes:
        ax.tick_params(labelsize=12)

        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")

        for tick in ax.get_yticklabels():
            tick.set_fontweight("bold")

    fig.suptitle(
        f"Scattering contributions — {scatterer}",
        fontsize=19,
        fontweight="bold",
        y=0.98,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    return fig

def plot_equivalent_thickness_vs_range(
    df,
    scatterer,
):
    """
    Plot MCS and Landau water-equivalent thickness versus range
    for one selected scatterer.

    MCS uses WET_theta only.
    """

    subset = df[df["Scatterer"] == scatterer].copy()

    if subset.empty:
        raise ValueError(
            f"No rows available for scatterer: {scatterer}"
        )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 11),
        sharex=True,
    )

    ax_mcs, ax_landau = axes

    styles = {
        ("Carbon ions", "No ripple filter"): {
            "color": "tab:blue",
            "marker": "o",
            "linestyle": "-",
            "label": "Carbon ions — no RF",
        },
        ("Carbon ions", "With ripple filter"): {
            "color": "tab:orange",
            "marker": "s",
            "linestyle": "--",
            "label": "Carbon ions — with RF",
        },
        ("Protons", "No ripple filter"): {
            "color": "tab:green",
            "marker": "^",
            "linestyle": "-.",
            "label": "Protons",
        },
    }

    for (particle, configuration), group in subset.groupby(
        ["Particle", "Configuration"]
    ):
        group = group.sort_values("Range [mm]")

        key = (particle, configuration)

        if key not in styles:
            continue

        style = styles[key]

        ax_mcs.plot(
            group["Range [mm]"],
            group["WET_theta"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.8,
            markersize=7,
            label=style["label"],
        )

        ax_landau.plot(
            group["Range [mm]"],
            group["WET_energy"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.8,
            markersize=7,
            label=style["label"],
        )

    ax_mcs.set_title(
        "MCS-equivalent thickness from angular scattering",
        fontsize=16,
        fontweight="bold",
    )

    ax_mcs.set_ylabel(
        "WET [mm water]",
        fontsize=14,
        fontweight="bold",
    )

    ax_mcs.grid(alpha=0.3)
    ax_mcs.legend(fontsize=10)

    ax_landau.set_title(
        "Landau-equivalent thickness from energy loss",
        fontsize=16,
        fontweight="bold",
    )

    ax_landau.set_xlabel(
        "Range [mm]",
        fontsize=14,
        fontweight="bold",
    )

    ax_landau.set_ylabel(
        "WET [mm water]",
        fontsize=14,
        fontweight="bold",
    )

    ax_landau.grid(alpha=0.3)
    ax_landau.legend(fontsize=10)

    for ax in axes:
        ax.tick_params(labelsize=12)

        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")

        for tick in ax.get_yticklabels():
            tick.set_fontweight("bold")

    fig.suptitle(
        f"Equivalent scatterer thickness — {scatterer}",
        fontsize=19,
        fontweight="bold",
        y=0.98,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    return fig


import gzip
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================================
# Constants used to translate the FLUKA input into run_whole parameters.
# ======================================================================
PROTON_REST_ENERGY_MEV = 938.2720813
ATOMIC_MASS_UNIT_MEV = 931.49410242


def _normalise_particle_type(value):
    """Return either 'proton' or 'carbon' from common input labels."""
    name = str(value).strip().lower()

    if name in {"p", "proton", "protons", "proton ion", "proton ions"}:
        return "proton"

    if name in {"c", "carbon", "carbon ion", "carbon ions", "c12", "12c"}:
        return "carbon"

    raise ValueError(
        f"Unsupported particle type {value!r}. "
        "Use proton/protons or carbon/carbon ions."
    )


def _parse_case_folder(folder_name):
    """Parse p030mm/c270mm (case insensitive) into particle type and range."""
    match = re.fullmatch(
        r"([pcPC])(\d+(?:\.\d+)?)mm",
        folder_name.strip(),
    )

    if match is None:
        raise ValueError(
            f"Folder {folder_name!r} does not follow the expected "
            "p***mm or c***mm convention."
        )

    particle = "proton" if match.group(1).lower() == "p" else "carbon"
    return particle, float(match.group(2))


def _find_single_input_file(case_dir):
    """Locate exactly one .inp file in a single energy directory."""
    input_files = sorted(case_dir.glob("*.inp"))

    if not input_files:
        raise FileNotFoundError(f"No .inp file found in {case_dir}.")

    if len(input_files) > 1:
        raise RuntimeError(
            f"More than one .inp file found in {case_dir}: "
            f"{[path.name for path in input_files]}"
        )

    return input_files[0]


def _strip_fluka_comment(line):
    """Remove full-line/inline FLUKA comments before interpreting a card."""
    return line.split("*", maxsplit=1)[0].strip()



def _file_tokens(file_path):
    """Tokenise file stems so RF is not confused with unrelated letters."""
    stem = Path(file_path).name.lower()
    return set(token for token in re.split(r"[^a-z0-9]+", stem) if token)


def _classify_output_file(output_path, particle):
    """
    Infer (medium, has_ripple_filter) from an output file name.
    """

    name = output_path.name.lower()

    if "air" in name:
        medium = "air"

    elif "vac" in name:
        medium = "vacuum"

    else:
        raise ValueError(
            f"Cannot determine medium from output filename: "
            f"{output_path.name}"
        )

    if particle == "proton":
        has_ripple_filter = False

    elif particle == "carbon":
        has_ripple_filter = (
            "norippfilters" not in name
            and "noripplefilters" not in name
            and "no_ripple" not in name
            and "no-ripple" not in name
        )

    else:
        raise ValueError(
            f"Unsupported particle type: {particle!r}"
        )

    return medium, has_ripple_filter

def _discover_outputs(
    case_dir,
    particle,
    *,
    file_classifier=None,
    overwrite_gz=False,
):
    """
    Individua i file .23/.23.gz senza decomprimerli.

    Restituisce una mappa con le chiavi:
        vacuum_no_rf
        vacuum_rf
        air_no_rf
        air_rf

    Se esistono entrambe le versioni dello stesso file,
    preferisce quella compressa .23.gz.

    overwrite_gz è mantenuto per compatibilità con le
    chiamate esistenti, ma non viene utilizzato.
    """
    case_dir = Path(case_dir)

    if not case_dir.is_dir():
        raise NotADirectoryError(
            f"Case directory does not exist: {case_dir}"
        )

    candidates = (
        sorted(case_dir.glob("*.23"))
        + sorted(case_dir.glob("*.23.gz"))
    )

    candidates = [
        path
        for path in candidates
        if path.is_file()
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No .23/.23.gz files found in {case_dir}."
        )

    selected_files = {}

    for candidate in candidates:
        # Serve soltanto per identificare due versioni
        # dello stesso file. Non crea nulla su disco.
        logical_path = (
            candidate.with_suffix("")
            if candidate.suffix.lower() == ".gz"
            else candidate
        )

        # I .gz vengono elaborati dopo i .23:
        # quindi hanno la precedenza.
        selected_files[logical_path] = candidate

    output_map = {}

    for logical_path, actual_path in sorted(
        selected_files.items(),
        key=lambda item: item[0].name,
    ):
        if file_classifier is None:
            # Il classificatore predefinito riceve il nome
            # senza .gz, come nel precedente percorso.
            medium, has_rf = _classify_output_file(
                logical_path,
                particle,
            )
        else:
            # Un classificatore custom riceve il file reale.
            medium, has_rf = file_classifier(actual_path)

        key = f"{medium}_{'rf' if has_rf else 'no_rf'}"

        if key in output_map and output_map[key] != actual_path:
            raise RuntimeError(
                f"Multiple files classified as {key} in {case_dir}: "
                f"{output_map[key].name!r} and "
                f"{actual_path.name!r}."
            )

        # Restituisce il percorso del file realmente esistente,
        # incluso il suffisso .gz quando presente.
        output_map[key] = actual_path

        print(
            f"[INFO] Selected {key}: {actual_path.name}"
        )

    return output_map

def _normalise_scatterer_name(value):
    """Map the labels returned by run_whole to stable labels for batch plots."""
    name = str(value).strip().lower().replace(" ", "")

    if "vac" in name or "win" in name:
        return "VW"

    if "nozzle" in name:
        return "Nozzle + ripple" if "ripple" in name else "Nozzle"

    if "100" in name and "air" in name:
        return "100 cm air"

    if "60" in name and "air" in name:
        return "60 cm air"

    return str(value).strip()


def _copy_base_parameters(base_parameters):
    """Copy the user dictionary without changing the original object."""
    if base_parameters is None:
        raise ValueError("base_parameters cannot be None.")
    return dict(base_parameters)


def _required_output_keys(particle):
    """
    Return the output-map keys required for each analysis configuration.

    Proton cases:
        only one configuration, always without ripple filters.

    Carbon cases:
        files without 'noRippFilters' are with ripple filters;
        files with 'noRippFilters' are without ripple filters.
    """

    if particle == "proton":
        return [
            (
                "No ripple filter",
                "vacuum_no_rf",
                "air_no_rf",
            ),
        ]

    if particle == "carbon":
        return [
            (
                "No ripple filter",
                "vacuum_no_rf",
                "air_no_rf",
            ),
            (
                "With ripple filter",
                "vacuum_rf",
                "air_rf",
            ),
        ]

    raise ValueError(f"Unsupported particle type: {particle!r}")


def _prepare_parameters_for_case(base_parameters, particle, beam_parameters, ripple_filters):
    """Inject case-specific beam settings into a copy of base parameters."""
    params = _copy_base_parameters(base_parameters)
    params.update({
        "particle_type": particle,
        "rest_energy": float(beam_parameters["rest_energy"]),
        "kinetic_energy": float(beam_parameters["kinetic_energy"]),
        "z": float(beam_parameters["z"]),
        "ripple_filters": bool(ripple_filters),
    })
    return params


def _validate_base_parameters(base_parameters):
    """Check the keys required by the supplied run_whole implementation."""
    required = {
        "column_names",
        "positions_scoring",
        "tolerance",
        "FWHM_0",
        "sigma_0_angle",
        "sigma_0_theta",
        "z0",
        "X0",
        "j",
        "Z_A_water",
        "K",
        "I",
        "rest_energy_electron",
    }

    missing = sorted(required.difference(base_parameters))
    if missing:
        raise KeyError(
            "base_parameters is missing required keys: " + ", ".join(missing)
        )

def _filename_or_empty(file_map, key):
    """
    Return the filename for key if present; otherwise return an empty string.
    """

    path = file_map.get(key)

    if path is None:
        return ""

    return Path(path).name

def _count_planned_configurations(root_dir):
    """
    Count the configurations that the batch is expected to process.

    Proton:
        one configuration per valid p***mm folder.

    Carbon:
        two configurations per valid c***mm folder:
        no ripple filter and with ripple filter.
    """

    root_dir = Path(root_dir)

    roots = {
        "proton": root_dir / "PROTON",
        "carbon": root_dir / "CARBON",
    }

    total = 0

    for expected_particle, particle_root in roots.items():

        if not particle_root.exists():
            continue

        for case_dir in particle_root.iterdir():

            if not case_dir.is_dir():
                continue

            try:
                particle, _ = _parse_case_folder(
                    case_dir.name
                )
            except ValueError:
                continue

            if particle != expected_particle:
                continue

            total += len(
                _required_output_keys(particle)
            )

    return total

def _notify_progress(progress_callback, **payload):
    """
    Call an optional user-interface callback without allowing
    progress-display errors to stop the scientific analysis.
    """

    if progress_callback is None:
        return

    try:
        progress_callback(payload)
    except Exception as exc:
        print(
            "WARNING: progress callback failed "
            f"and was ignored: {exc}"
        )



# ======================================================================
# Batch plotting functions
# ======================================================================
def _batch_plot_style():
    """Return a consistent style for particle/configuration combinations."""
    return {
        ("Carbon ions", "No ripple filter"): {
            "color": "tab:blue",
            "marker": "o",
            "linestyle": "-",
            "label": "Carbon ions — no RF",
        },
        ("Carbon ions", "With ripple filter"): {
            "color": "tab:orange",
            "marker": "s",
            "linestyle": "--",
            "label": "Carbon ions — with RF",
        },
        ("Protons", "No ripple filter"): {
            "color": "tab:green",
            "marker": "^",
            "linestyle": "-.",
            "label": "Protons",
        },
    }


def _format_batch_axis(ax, title, ylabel, xlabel=None, legend_loc="best"):
    """Use the same presentation formatting for all global batch plots."""
    ax.set_title(title, fontsize=17, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=15, fontweight="bold", labelpad=8)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=15, fontweight="bold", labelpad=8)

    ax.tick_params(axis="both", which="major", labelsize=13, width=1.4, length=6)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.grid(True, alpha=0.30, linewidth=1.0)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(fontsize=10, frameon=True, loc=legend_loc, ncol=2)
        for text in legend.get_texts():
            text.set_fontweight("bold")


def plot_scattering_contributions_vs_range(batch_table, scatterer):
    """
    Plot scattering contributions vs range for a single scatterer.

    Upper panel: x', y' and theta contributions.
    Lower panel: energy contribution.
    """
    data = batch_table[batch_table["Scatterer"] == scatterer].copy()

    if data.empty:
        raise ValueError(f"No rows found for scatterer {scatterer!r}.")

    styles = _batch_plot_style()
    observable_line_style = {"contr_xp": "-", "contr_yp": "--", "contr_theta": ":"}
    observable_label = {"contr_xp": r"$x'$", "contr_yp": r"$y'$", "contr_theta": r"$\theta$"}

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    ax_angular, ax_energy = axes

    for (particle, configuration), group in data.groupby(["Particle", "Configuration"]):
        key = (particle, configuration)
        if key not in styles:
            continue

        style = styles[key]
        group = group.sort_values("Range [mm]")

        for column in ("contr_xp", "contr_yp", "contr_theta"):
            ax_angular.plot(
                group["Range [mm]"],
                group[column],
                color=style["color"],
                marker=style["marker"],
                linestyle=observable_line_style[column],
                linewidth=2.6,
                markersize=7,
                label=f"{style['label']} — {observable_label[column]}",
            )

        ax_energy.plot(
            group["Range [mm]"],
            group["contr_energy"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.8,
            markersize=7,
            label=style["label"],
        )

    _format_batch_axis(
        ax_angular,
        title="Angular scattering contributions",
        ylabel="Contribution [mrad]",
        legend_loc="best",
    )
    _format_batch_axis(
        ax_energy,
        title="Energy-loss contribution",
        ylabel="Contribution [MeV]",
        xlabel="Range [mm]",
        legend_loc="best",
    )

    fig.suptitle(
        f"{scatterer} — scattering contributions versus range",
        fontsize=20,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_equivalent_thickness_vs_range(batch_table, scatterer):
    """
    Plot equivalent water thickness vs range for one scatterer.

    Upper panel: MCS equivalent thickness inferred from theta only.
    Lower panel: Landau equivalent thickness inferred from energy loss.
    """
    data = batch_table[batch_table["Scatterer"] == scatterer].copy()

    if data.empty:
        raise ValueError(f"No rows found for scatterer {scatterer!r}.")

    styles = _batch_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    ax_mcs, ax_landau = axes

    for (particle, configuration), group in data.groupby(["Particle", "Configuration"]):
        key = (particle, configuration)
        if key not in styles:
            continue

        style = styles[key]
        group = group.sort_values("Range [mm]")

        ax_mcs.plot(
            group["Range [mm]"],
            group["WET_theta"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.8,
            markersize=7,
            label=style["label"],
        )

        ax_landau.plot(
            group["Range [mm]"],
            group["WET_energy"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.8,
            markersize=7,
            label=style["label"],
        )

    _format_batch_axis(
        ax_mcs,
        title="MCS-equivalent thickness from $\\theta$",
        ylabel="WET [mm water]",
        legend_loc="best",
    )
    _format_batch_axis(
        ax_landau,
        title="Landau-equivalent thickness from energy loss",
        ylabel="WET [mm water]",
        xlabel="Range [mm]",
        legend_loc="best",
    )

    fig.suptitle(
        f"{scatterer} — equivalent thickness versus range",
        fontsize=20,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# ======================================================================
# Example use (run in a separate notebook cell / driver script)
# ======================================================================
# base_parameters = {
#     "column_names": [...],
#     "positions_scoring": [...],
#     "tolerance": 1e-6,
#     "FWHM_0": 15.0,
#     "sigma_0_angle": 0.0,
#     "sigma_0_theta": 0.0,
#     "z0": -600.0,
#     "X0": 36.08,
#     "j": 0.200,
#     "Z_A_water": 0.55509,
#     "K": 0.307075,
#     "I": 75e-6,
#     "rest_energy_electron": 0.51099895,
#     "cleaning_cut_mode": "ellipse",
#     "ellipse_peak_fraction": 0.10,
#     "theta_percentile_min": 0.0,
#     "theta_percentile_max": 99.0,
# }
#
# batch_result = run_whole_all_energies(
#     root_dir="/path/to/root/with/PROTON/and/CARBON",
#     base_parameters=base_parameters,
# )
#
# all_results = batch_result["all_scatterer_table"]
# display(all_results)
# display(batch_result["cases_table"])
# display(batch_result["errors"])
#
# for figure in batch_result["contribution_plots"].values():
#     plt.show()
# for figure in batch_result["wet_plots"].values():
#     plt.show()

# CODICI PER ANALISI TUTTE ENERGIE

import gzip
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# Particle constants
# ------------------------------------------------------------------
PROTON_REST_MASS_MEV = 938.2720813
CARBON12_REST_MASS_MEV = 12.0 * 931.49410242

PROTON_CHARGE = 1
CARBON12_CHARGE = 6
CARBON12_MASS_NUMBER = 12


@dataclass
class BeamCase:
    particle: str
    range_mm: float
    case_dir: Path
    input_file: Path
    kinetic_energy_mev: float
    rest_energy_mev: float
    charge_z: int
    mass_number: int
    input_energy_mev_u: float | None
    files: dict


def _beam_parameters_from_input(
    input_path,
    particle,
):
    """
    Read the BEAM card from a FLUKA input and create the parameters
    required by the batch analysis.

    Dataset convention used here:
    - proton folders contain proton simulations;
    - carbon folders contain 12C simulations (Z=6, A=12).
    """

    input_path = Path(input_path)

    particle = str(particle).strip().lower()

    beam_value_gev = None

    with input_path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as handle:

        for raw_line in handle:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("*"):
                continue

            tokens = line.split()

            if not tokens:
                continue

            if tokens[0].upper() != "BEAM":
                continue

            for token in tokens[1:]:
                try:
                    beam_value_gev = float(token)
                    break
                except ValueError:
                    continue

            if beam_value_gev is not None:
                break

    if beam_value_gev is None:
        raise ValueError(
            f"No numeric BEAM card found in {input_path}."
        )

    beam_energy_mev = abs(
        float(beam_value_gev)
    ) * 1000.0

    if particle == "proton":
        return {
            "rest_energy": 938.2720813,
            "kinetic_energy": beam_energy_mev,
            "z": 1.0,
            "mass_number": 1,
            "energy_per_u_mev": None,
        }

    if particle == "carbon":
        # All datasets under the CARBON root are 12C.
        z = 6
        a = 12

        return {
            "rest_energy": a * 931.49410242,
            "kinetic_energy": a * beam_energy_mev,
            "z": float(z),
            "mass_number": int(a),
            "energy_per_u_mev": beam_energy_mev,
        }

    raise ValueError(
        f"Unsupported particle type: {particle!r}."
    )

def decompress_gz_file(gz_path: Path, overwrite: bool = False) -> Path:
    """
    Decompress file.gz to file, keeping the original .gz.

    Example:
        output_001.23.gz -> output_001.23
    """

    if gz_path.suffix.lower() != ".gz":
        raise ValueError(
            f"Expected a .gz file, received: {gz_path}"
        )

    output_path = gz_path.with_suffix("")

    if output_path.exists() and not overwrite:
        return output_path

    with gzip.open(gz_path, "rb") as source:
        with output_path.open("wb") as target:
            shutil.copyfileobj(source, target)

    return output_path



def _classify_input_file(input_path, particle):
    """
    Infer (medium, has_ripple_filter) from an input filename.

    Conventions
    -----------
    Protons:
        every case is considered WITHOUT ripple filters, irrespective
        of the filename.

    Carbon:
        filename containing 'noRippFilters' -> WITHOUT ripple filters;
        filename not containing it          -> WITH ripple filters.
    """

    name = input_path.name.lower()

    if "air" in name:
        medium = "air"

    elif "vac" in name:
        medium = "vacuum"

    else:
        raise ValueError(
            f"Cannot determine medium from input filename: "
            f"{input_path.name}"
        )

    if particle == "proton":
        has_ripple_filter = False

    elif particle == "carbon":
        no_ripple_keywords = (
            "norippfilters",
            "noripplefilters",
            "no_ripple",
            "no-ripple",
        )

        has_no_ripple_keyword = any(
            keyword in name
            for keyword in no_ripple_keywords
        )

        has_ripple_filter = not has_no_ripple_keyword

    else:
        raise ValueError(
            f"Unsupported particle type: {particle!r}"
        )

    return medium, has_ripple_filter

def _find_input_files(case_dir, particle):
    """
    Return the input files indexed as:

        vacuum_no_rf
        air_no_rf
        vacuum_rf
        air_rf

    Proton files are always classified as no-ripple-filter files.
    Carbon files follow their filename convention:
    'noRippFilters' means no ripple filters.
    """

    input_files = sorted(case_dir.glob("*.inp"))

    if not input_files:
        raise FileNotFoundError(
            f"No .inp files found in {case_dir}."
        )

    input_map = {}

    for input_path in input_files:
        medium, has_ripple_filter = _classify_input_file(
            input_path,
            particle,
        )

        key = (
            f"{medium}_rf"
            if has_ripple_filter
            else f"{medium}_no_rf"
        )

        if key in input_map:
            raise RuntimeError(
                f"More than one .inp file classified as '{key}' "
                f"in {case_dir}: "
                f"{input_map[key].name} and {input_path.name}"
            )

        input_map[key] = input_path

    return input_map


import gzip
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================================
# Constants used to translate the FLUKA input into run_whole parameters.
# ======================================================================
PROTON_REST_ENERGY_MEV = 938.2720813
ATOMIC_MASS_UNIT_MEV = 931.49410242


def _parse_case_folder(folder_name):
    """Parse p030mm/c270mm (case insensitive) into particle type and range."""
    match = re.fullmatch(
        r"([pcPC])(\d+(?:\.\d+)?)mm",
        folder_name.strip(),
    )

    if match is None:
        raise ValueError(
            f"Folder {folder_name!r} does not follow the expected "
            "p***mm or c***mm convention."
        )

    particle = "proton" if match.group(1).lower() == "p" else "carbon"
    return particle, float(match.group(2))



def _classify_output_file(output_path, particle):
    """
    Infer (medium, has_ripple_filter) from an output file name.
    """

    name = output_path.name.lower()

    if "air" in name:
        medium = "air"

    elif "vac" in name:
        medium = "vacuum"

    else:
        raise ValueError(
            f"Cannot determine medium from output filename: "
            f"{output_path.name}"
        )

    if particle == "proton":
        has_ripple_filter = False

    elif particle == "carbon":
        has_ripple_filter = (
            "norippfilters" not in name
            and "noripplefilters" not in name
            and "no_ripple" not in name
            and "no-ripple" not in name
        )

    else:
        raise ValueError(
            f"Unsupported particle type: {particle!r}"
        )

    return medium, has_ripple_filter


def _normalise_scatterer_name(value):
    """Map the labels returned by run_whole to stable labels for batch plots."""
    name = str(value).strip().lower().replace(" ", "")

    if "vac" in name or "win" in name:
        return "VW"

    if "nozzle" in name:
        return "Nozzle + ripple" if "ripple" in name else "Nozzle"

    if "100" in name and "air" in name:
        return "100 cm air"

    if "60" in name and "air" in name:
        return "60 cm air"

    return str(value).strip()


def _copy_base_parameters(base_parameters):
    """Copy the user dictionary without changing the original object."""
    if base_parameters is None:
        raise ValueError("base_parameters cannot be None.")
    return dict(base_parameters)


def _required_output_keys(particle):
    """
    Return the output-map keys required for each analysis configuration.

    Proton cases:
        only one configuration, always without ripple filters.

    Carbon cases:
        files without 'noRippFilters' are with ripple filters;
        files with 'noRippFilters' are without ripple filters.
    """

    if particle == "proton":
        return [
            (
                "No ripple filter",
                "vacuum_no_rf",
                "air_no_rf",
            ),
        ]

    if particle == "carbon":
        return [
            (
                "No ripple filter",
                "vacuum_no_rf",
                "air_no_rf",
            ),
            (
                "With ripple filter",
                "vacuum_rf",
                "air_rf",
            ),
        ]

    raise ValueError(f"Unsupported particle type: {particle!r}")


def _prepare_parameters_for_case(base_parameters, particle, beam_parameters, ripple_filters):
    """Inject case-specific beam settings into a copy of base parameters."""
    params = _copy_base_parameters(base_parameters)
    params.update({
        "particle_type": particle,
        "rest_energy": float(beam_parameters["rest_energy"]),
        "kinetic_energy": float(beam_parameters["kinetic_energy"]),
        "z": float(beam_parameters["z"]),
        "ripple_filters": bool(ripple_filters),
    })
    return params


def _validate_base_parameters(base_parameters):
    """Check the keys required by the supplied run_whole implementation."""
    required = {
        "column_names",
        "positions_scoring",
        "tolerance",
        "FWHM_0",
        "sigma_0_angle",
        "sigma_0_theta",
        "z0",
        "X0",
        "j",
        "Z_A_water",
        "K",
        "I",
        "rest_energy_electron",
    }

    missing = sorted(required.difference(base_parameters))
    if missing:
        raise KeyError(
            "base_parameters is missing required keys: " + ", ".join(missing)
        )

def _filename_or_empty(file_map, key):
    """
    Return the filename for key if present; otherwise return an empty string.
    """

    path = file_map.get(key)

    if path is None:
        return ""

    return Path(path).name

def _count_planned_configurations(root_dir):
    """
    Count the configurations that the batch is expected to process.

    Proton:
        one configuration per valid p***mm folder.

    Carbon:
        two configurations per valid c***mm folder:
        no ripple filter and with ripple filter.
    """

    root_dir = Path(root_dir)

    roots = {
        "proton": root_dir / "PROTON",
        "carbon": root_dir / "CARBON",
    }

    total = 0

    for expected_particle, particle_root in roots.items():

        if not particle_root.exists():
            continue

        for case_dir in particle_root.iterdir():

            if not case_dir.is_dir():
                continue

            try:
                particle, _ = _parse_case_folder(
                    case_dir.name
                )
            except ValueError:
                continue

            if particle != expected_particle:
                continue

            total += len(
                _required_output_keys(particle)
            )

    return total

def _notify_progress(progress_callback, **payload):
    """
    Call an optional user-interface callback without allowing
    progress-display errors to stop the scientific analysis.
    """

    if progress_callback is None:
        return

    try:
        progress_callback(payload)
    except Exception as exc:
        print(
            "WARNING: progress callback failed "
            f"and was ignored: {exc}"
        )


def run_whole_all_energies(
    root_dir,
    base_parameters,
    *,
    continue_on_error=True,
    overwrite_gz=False,
    file_classifier=None,
    run_single_case=None,
    progress_callback=None
):
    """
    Batch version of run_whole for all p***mm and c***mm folders.

    Directory layout expected:
        root_dir/
          PROTON/
            p030mm/
              one .inp
              vacuum .23/.23.gz
              air .23/.23.gz
          CARBON/
            c030mm/
              one .inp
              vacuum/no-RF .23/.23.gz
              air/no-RF .23/.23.gz
              vacuum/RF .23/.23.gz
              air/RF .23/.23.gz

    For each energy case, this function:
      1. identifies the particle type and range from the folder name;
      2. reads BEAM and, for carbon, HEAVYION from the .inp file;
      3. decompresses .23.gz files when necessary;
      4. calls the pre-existing run_whole for every available configuration;
      5. collects export_table_scatterers into one summary dataframe;
      6. returns the all-energy contribution/WET plots.

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Parent directory containing PROTON and CARBON.
    base_parameters : dict
        Parameters common to every dataset. Do not include case-specific
        rest_energy, kinetic_energy, z, particle_type or ripple_filters;
        these are inferred/set separately for each case.
    continue_on_error : bool
        If True, record failed cases and continue with the remaining folders.
    overwrite_gz : bool
        If True, regenerate decompressed .23 files even when they exist.
    file_classifier : callable or None
        Optional function path -> (medium, has_ripple_filter).
    run_single_case : callable or None
        Optional replacement for run_whole, useful for testing. It must have
        the same signature and return a dict containing export_table_scatterers.

    Returns
    -------
    dict with keys:
        all_scatterer_table : long-form dataframe for all successful cases
        cases_table          : case-level metadata/status table
        errors               : dataframe of failed cases/configurations
        contribution_plots   : dict scatterer -> matplotlib Figure
        wet_plots            : dict scatterer -> matplotlib Figure
    """
    _validate_base_parameters(base_parameters)
    configure_histogram_binning(
        base_parameters
    )

    if run_single_case is None:
        run_single_case = run_whole

    root_dir = Path(root_dir).expanduser().resolve()

    roots = {
        "proton": root_dir / "PROTON",
        "carbon": root_dir / "CARBON",
    }

    only = base_parameters.get("only")

    if only is not None:
        if not isinstance(only, str):
            raise TypeError(
                "Il parametro 'only' deve essere "
                "una stringa oppure None."
            )

        only = only.strip()

        if (
            not only
            or only in {".", ".."}
            or "/" in only
            or "\\" in only
        ):
            raise ValueError(
                "'only' deve contenere il nome di un caso, "
                "per esempio p030 o p030mm."
            )

    def case_selection_key(name):
        normalized = name.strip().lower()

        if normalized.endswith("mm"):
            normalized = normalized[:-2]

        return normalized

    only_key = (
        case_selection_key(only)
        if only is not None
        else None
    )

    # Individua i casi prima di leggere o decomprimere file.
    selected_case_dirs = {}

    for expected_particle, particle_root in roots.items():
        folders = []

        if particle_root.is_dir():
            for case_dir in sorted(particle_root.iterdir()):
                if not case_dir.is_dir():
                    continue

                if (
                    only_key is not None
                    and case_selection_key(case_dir.name)
                    != only_key
                ):
                    continue

                try:
                    _parse_case_folder(case_dir.name)
                except ValueError:
                    continue

                folders.append(case_dir)

        selected_case_dirs[expected_particle] = folders

    if only is not None:
        matched_folders = [
            folder
            for folders in selected_case_dirs.values()
            for folder in folders
        ]

        if not matched_folders:
            raise FileNotFoundError(
                f"Nessun caso valido corrisponde a {only!r} "
                f"nelle cartelle PROTON/ o CARBON/ "
                f"all'interno di {root_dir}."
            )

        print(
            "\n[INFO] Analisi limitata ai casi:",
            ", ".join(str(folder) for folder in matched_folders),
        )
    else:
        print("\n[INFO] Analisi di tutti i casi disponibili.")

    case_metadata = []
    all_scatterer_tables = []
    errors = []

    # Conta solo le configurazioni dei casi selezionati.
    # Non apre e non decomprime i file.
    total_configurations = 0

    for expected_particle, folders in selected_case_dirs.items():
        for case_dir in folders:
            particle, _ = _parse_case_folder(case_dir.name)

            if particle == expected_particle:
                total_configurations += sum(
                    1
                    for _ in _required_output_keys(particle)
                )

    completed_configurations = 0

    _notify_progress(
        progress_callback,
        event="start",
        current=0,
        total=total_configurations,
        case_folder="",
        particle="",
        range_mm=None,
        configuration="",
        status="Starting batch analysis",
    )

    for expected_particle, particle_root in roots.items():
        if not particle_root.is_dir():
            # Con --only, l'assenza della directory dell'altra
            # specie non deve produrre un errore.
            if only is None:
                errors.append({
                    "case_folder": "",
                    "particle": expected_particle,
                    "range_mm": np.nan,
                    "configuration": "",
                    "stage": "discovery",
                    "error": (
                        f"Directory not found: {particle_root}"
                    ),
                })

            continue

        for case_dir in selected_case_dirs[expected_particle]:
            try:
                particle, range_mm = _parse_case_folder(
                    case_dir.name
                )
            except ValueError:
                # Ignore unrelated directories without interrupting the batch.
                continue

            if particle != expected_particle:
                errors.append({
                    "case_folder": case_dir.name,
                    "particle": particle,
                    "range_mm": range_mm,
                    "configuration": "",
                    "stage": "discovery",
                    "error": (
                        f"Folder is under {particle_root.name} but its prefix "
                        f"identifies {particle}."
                    ),
                })
                if not continue_on_error:
                    raise RuntimeError(errors[-1]["error"])
                continue

            try:
                input_map = _find_input_files(
                    case_dir,
                    particle,
                )

                output_map = _discover_outputs(
                    case_dir,
                    particle,
                    file_classifier=file_classifier,
                    overwrite_gz=overwrite_gz,
                )
            except Exception as exc:
                errors.append({
                    "case_folder": case_dir.name,
                    "particle": particle,
                    "range_mm": range_mm,
                    "configuration": "",
                    "stage": "input discovery",
                    "error": str(exc),
                })
                if not continue_on_error:
                    raise
                continue

            case_metadata.append({
                "case_folder": case_dir.name,
                "case_directory": str(case_dir),

                "particle": particle,
                "range_mm": range_mm,

                # Input cards
                "input_vacuum_no_rf": _filename_or_empty(
                    input_map,
                    "vacuum_no_rf",
                ),

                "input_air_no_rf": _filename_or_empty(
                    input_map,
                    "air_no_rf",
                ),

                "input_vacuum_rf": _filename_or_empty(
                    input_map,
                    "vacuum_rf",
                ),

                "input_air_rf": _filename_or_empty(
                    input_map,
                    "air_rf",
                ),

                # FLUKA output files
                "output_vacuum_no_rf": _filename_or_empty(
                    output_map,
                    "vacuum_no_rf",
                ),

                "output_air_no_rf": _filename_or_empty(
                    output_map,
                    "air_no_rf",
                ),

                "output_vacuum_rf": _filename_or_empty(
                    output_map,
                    "vacuum_rf",
                ),

                "output_air_rf": _filename_or_empty(
                    output_map,
                    "air_rf",
                ),
            })

            print("\n" + "=" * 76)
            print(
                f"CASE {case_dir.name} | "
                f"particle={particle} | "
                f"range={range_mm:g} mm"
            )
            print("=" * 76)

            for configuration, vacuum_key, air_key in _required_output_keys(particle):

                input_key = (
                    "vacuum_rf"
                    if configuration == "With ripple filter"
                    else "vacuum_no_rf"
                )

                if input_key not in input_map:
                    # tuo blocco di gestione errore
                    continue

                input_file = input_map[input_key]

                beam_parameters = _beam_parameters_from_input(
                    input_file,
                    particle,
                )

                case_label = (
                    f"{particle} | "
                    f"Range={range_mm:.3f} mm | "
                    f"T_total={beam_parameters['kinetic_energy']:.6g} MeV | "
                    f"{configuration}"
                )

                missing = [
                    key
                    for key in (vacuum_key, air_key)
                    if key not in output_map
                ]


                if missing:
                    message = (
                        f"Missing files for {configuration}: "
                        + ", ".join(missing)
                    )

                    errors.append({
                        "case_folder": case_dir.name,
                        "particle": particle,
                        "range_mm": range_mm,
                        "configuration": configuration,
                        "stage": "output discovery",
                        "error": message,
                    })

                    print("WARNING:", message)

                    if not continue_on_error:
                        raise FileNotFoundError(message)

                    continue

                params = _prepare_parameters_for_case(
                    base_parameters,
                    particle=particle,
                    beam_parameters=beam_parameters,
                    ripple_filters=(configuration == "With ripple filter"),
                )
                params["nRows"] = base_parameters.get("nRows")

                batch_output_root = Path(
                    base_parameters.get(
                        "batch_output_root",
                        "analysis_results",
                    )
                ).expanduser().resolve()

                configuration_folder = (
                    "with_ripple_filter"
                    if configuration == "With ripple filter"
                    else "no_ripple_filter"
                )

                case_output_dir = (
                    batch_output_root
                    / "batch_plots"
                    / str(particle)
                    / case_dir.name
                    / configuration_folder
                )

                case_output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                params["save_plots"] = True

                params["case_output_dir"] = str(
                    case_output_dir
                )

                print(
                    "\nPrepared output directory for this configuration:"
                )

                print(
                    params["case_output_dir"]
                )

                configuration_dir = (
                    "with_ripple_filter"
                    if configuration == "With ripple filter"
                    else "no_ripple_filter"
                )

                batch_output_root = Path(
                    base_parameters.get(
                        "batch_output_root",
                        "analysis_results",
                    )
                ).expanduser().resolve()

                params["output_dir"] = str(
                    batch_output_root
                    / particle
                    / case_dir.name
                    / configuration_dir
                )

                _notify_progress(
                    progress_callback,
                    event="running",
                    current=completed_configurations,
                    total=total_configurations,
                    case_folder=case_dir.name,
                    particle=particle,
                    range_mm=float(range_mm),
                    configuration=configuration,
                    input_file=input_file.name,
                    status="Running",
                )


                try:
                    result = run_single_case(
                        output_map[vacuum_key],
                        output_map[air_key],
                        params,
                    )

                    if "export_table_scatterers" not in result:
                        raise KeyError(
                            "run_whole output has no 'export_table_scatterers' key."
                        )

                    table = result["export_table_scatterers"].copy()

                    if table.empty:
                        raise ValueError("export_table_scatterers is empty.")

                    required_columns = {
                        "Scatterers",
                        "Contribution in x' [mrad]",
                        "Contribution in y' [mrad]",
                        "Contribution in theta [mrad]",
                        "Contribution in energy [MeV]",
                        "Equivalent x (MCS) in x' [mm]",
                        "Equivalent x (MCS) in y' [mm]",
                        "Equivalent x (MCS) in theta [mm]",
                        "Equivalent x (Landau) in energy [mm]",
                    }

                    missing_columns = sorted(
                        required_columns.difference(table.columns)
                    )

                    if missing_columns:
                        raise KeyError(
                            "Scatterer table misses columns: "
                            + ", ".join(missing_columns)
                        )

                    table["Scatterer"] = table["Scatterers"].map(
                        _normalise_scatterer_name
                    )
                    table["Particle"] = (
                        "Protons" if particle == "proton" else "Carbon ions"
                    )
                    table["Configuration"] = configuration
                    table["Range [mm]"] = float(range_mm)
                    table["Kinetic energy total [MeV]"] = float(
                        beam_parameters["kinetic_energy"]
                    )
                    table["Input energy [MeV/u]"] = beam_parameters["energy_per_u_mev"]
                    table["Input file"] = input_file.name
                    table["Case folder"] = case_dir.name
                    table["Vacuum file"] = output_map[vacuum_key].name
                    table["Air file"] = output_map[air_key].name

                    all_scatterer_tables.append(table)
                    print(f"Completed: {configuration}")


                    completed_configurations += 1

                    _notify_progress(
                        progress_callback,
                        event="completed",
                        current=completed_configurations,
                        total=total_configurations,
                        case_folder=case_dir.name,
                        particle=particle,
                        range_mm=float(range_mm),
                        configuration=configuration,
                        input_file=input_file.name,
                        status="Completed",
                    )

                except Exception as exc:
                    errors.append({
                        "case_folder": case_dir.name,
                        "particle": particle,
                        "range_mm": range_mm,
                        "configuration": configuration,
                        "stage": "run_whole",
                        "error": repr(exc),
                    })
                    print(f"ERROR in {case_dir.name} | {configuration}: {exc}")


                    completed_configurations += 1

                    _notify_progress(
                        progress_callback,
                        event="failed",
                        current=completed_configurations,
                        total=total_configurations,
                        case_folder=case_dir.name,
                        particle=particle,
                        range_mm=float(range_mm),
                        configuration=configuration,
                        input_file=input_file.name,
                        status=f"Failed: {exc}",
                    )

                    if not continue_on_error:
                        raise

    cases_table = pd.DataFrame(case_metadata)
    errors_table = pd.DataFrame(errors)

    if all_scatterer_tables:
        all_scatterer_table = pd.concat(all_scatterer_tables, ignore_index=True)
        all_scatterer_table = all_scatterer_table.rename(columns={
            "Contribution in x' [mrad]": "contr_xp",
            "Contribution in y' [mrad]": "contr_yp",
            "Contribution in theta [mrad]": "contr_theta",
            "Contribution in energy [MeV]": "contr_energy",
            "Equivalent x (MCS) in x' [mm]": "WET_xp",
            "Equivalent x (MCS) in y' [mm]": "WET_yp",
            "Equivalent x (MCS) in theta [mm]": "WET_theta",
            "Equivalent x (Landau) in energy [mm]": "WET_energy",
        })

        all_scatterer_table = all_scatterer_table.sort_values(
            ["Scatterer", "Particle", "Configuration", "Range [mm]"]
        ).reset_index(drop=True)
    else:
        all_scatterer_table = pd.DataFrame()
    

    _notify_progress(
        progress_callback,
        event="finished",
        current=completed_configurations,
        total=total_configurations,
        case_folder="",
        particle="",
        range_mm=None,
        configuration="",
        status="Batch analysis completed",
    )




    return {
        "all_scatterer_table": all_scatterer_table,
        "cases_table": cases_table,
        "errors": errors_table,
    }




def run_whole_all_energiee(
    root_dir,
    base_parameters,
    *,
    continue_on_error=True,
    overwrite_gz=False,
    file_classifier=None,
    run_single_case=None,
    progress_callback=None
):
    """
    Batch version of run_whole for all p***mm and c***mm folders.

    Directory layout expected:
        root_dir/
          PROTON/
            p030mm/
              one .inp
              vacuum .23/.23.gz
              air .23/.23.gz
          CARBON/
            c030mm/
              one .inp
              vacuum/no-RF .23/.23.gz
              air/no-RF .23/.23.gz
              vacuum/RF .23/.23.gz
              air/RF .23/.23.gz

    For each energy case, this function:
      1. identifies the particle type and range from the folder name;
      2. reads BEAM and, for carbon, HEAVYION from the .inp file;
      3. decompresses .23.gz files when necessary;
      4. calls the pre-existing run_whole for every available configuration;
      5. collects export_table_scatterers into one summary dataframe;
      6. returns the all-energy contribution/WET plots.

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Parent directory containing PROTON and CARBON.
    base_parameters : dict
        Parameters common to every dataset. Do not include case-specific
        rest_energy, kinetic_energy, z, particle_type or ripple_filters;
        these are inferred/set separately for each case.
    continue_on_error : bool
        If True, record failed cases and continue with the remaining folders.
    overwrite_gz : bool
        If True, regenerate decompressed .23 files even when they exist.
    file_classifier : callable or None
        Optional function path -> (medium, has_ripple_filter).
    run_single_case : callable or None
        Optional replacement for run_whole, useful for testing. It must have
        the same signature and return a dict containing export_table_scatterers.

    Returns
    -------
    dict with keys:
        all_scatterer_table : long-form dataframe for all successful cases
        cases_table          : case-level metadata/status table
        errors               : dataframe of failed cases/configurations
        contribution_plots   : dict scatterer -> matplotlib Figure
        wet_plots            : dict scatterer -> matplotlib Figure
    """
    _validate_base_parameters(base_parameters)

    if run_single_case is None:
        run_single_case = run_whole

    root_dir = Path(root_dir).expanduser().resolve()
    roots = {
        "proton": root_dir / "PROTON",
        "carbon": root_dir / "CARBON",
    }

    case_metadata = []
    all_scatterer_tables = []
    errors = []

    total_configurations = _count_planned_configurations(
        root_dir
    )

    completed_configurations = 0

    _notify_progress(
        progress_callback,
        event="start",
        current=0,
        total=total_configurations,
        case_folder="",
        particle="",
        range_mm=None,
        configuration="",
        status="Starting batch analysis",
    )

    for expected_particle, particle_root in roots.items():
        if not particle_root.exists():
            errors.append({
                "case_folder": "",
                "particle": expected_particle,
                "range_mm": np.nan,
                "configuration": "",
                "stage": "discovery",
                "error": f"Directory not found: {particle_root}",
            })
            continue

        for case_dir in sorted(particle_root.iterdir()):
            if not case_dir.is_dir():
                continue

            try:
                particle, range_mm = _parse_case_folder(case_dir.name)
            except ValueError:
                # Ignore unrelated directories without interrupting the batch.
                continue

            if particle != expected_particle:
                errors.append({
                    "case_folder": case_dir.name,
                    "particle": particle,
                    "range_mm": range_mm,
                    "configuration": "",
                    "stage": "discovery",
                    "error": (
                        f"Folder is under {particle_root.name} but its prefix "
                        f"identifies {particle}."
                    ),
                })
                if not continue_on_error:
                    raise RuntimeError(errors[-1]["error"])
                continue

            try:
                input_map = _find_input_files(
                    case_dir,
                    particle,
                )

                output_map = _discover_outputs(
                    case_dir,
                    particle,
                    file_classifier=file_classifier,
                    overwrite_gz=overwrite_gz,
                )
            except Exception as exc:
                errors.append({
                    "case_folder": case_dir.name,
                    "particle": particle,
                    "range_mm": range_mm,
                    "configuration": "",
                    "stage": "input discovery",
                    "error": str(exc),
                })
                if not continue_on_error:
                    raise
                continue

            case_metadata.append({
                "case_folder": case_dir.name,
                "case_directory": str(case_dir),

                "particle": particle,
                "range_mm": range_mm,

                # Input cards
                "input_vacuum_no_rf": _filename_or_empty(
                    input_map,
                    "vacuum_no_rf",
                ),

                "input_air_no_rf": _filename_or_empty(
                    input_map,
                    "air_no_rf",
                ),

                "input_vacuum_rf": _filename_or_empty(
                    input_map,
                    "vacuum_rf",
                ),

                "input_air_rf": _filename_or_empty(
                    input_map,
                    "air_rf",
                ),

                # FLUKA output files
                "output_vacuum_no_rf": _filename_or_empty(
                    output_map,
                    "vacuum_no_rf",
                ),

                "output_air_no_rf": _filename_or_empty(
                    output_map,
                    "air_no_rf",
                ),

                "output_vacuum_rf": _filename_or_empty(
                    output_map,
                    "vacuum_rf",
                ),

                "output_air_rf": _filename_or_empty(
                    output_map,
                    "air_rf",
                ),
            })

            print("\n" + "=" * 76)
            print(
                f"CASE {case_dir.name} | "
                f"particle={particle} | "
                f"range={range_mm:g} mm"
            )
            print("=" * 76)

            for configuration, vacuum_key, air_key in _required_output_keys(particle):

                input_key = (
                    "vacuum_rf"
                    if configuration == "With ripple filter"
                    else "vacuum_no_rf"
                )

                if input_key not in input_map:
                    # tuo blocco di gestione errore
                    continue

                input_file = input_map[input_key]

                beam_parameters = _beam_parameters_from_input(
                    input_file,
                    particle,
                )

                case_label = (
                    f"{particle} | "
                    f"Range={range_mm:.3f} mm | "
                    f"T_total={beam_parameters['kinetic_energy']:.6g} MeV | "
                    f"{configuration}"
                )

                missing = [
                    key
                    for key in (vacuum_key, air_key)
                    if key not in output_map
                ]


                if missing:
                    message = (
                        f"Missing files for {configuration}: "
                        + ", ".join(missing)
                    )

                    errors.append({
                        "case_folder": case_dir.name,
                        "particle": particle,
                        "range_mm": range_mm,
                        "configuration": configuration,
                        "stage": "output discovery",
                        "error": message,
                    })

                    print("WARNING:", message)

                    if not continue_on_error:
                        raise FileNotFoundError(message)

                    continue

                params = _prepare_parameters_for_case(
                    base_parameters,
                    particle=particle,
                    beam_parameters=beam_parameters,
                    ripple_filters=(configuration == "With ripple filter"),
                )
                params["histogram_n_bins"] = base_parameters[
                        "histogram_n_bins"
                    ]

                _notify_progress(
                    progress_callback,
                    event="running",
                    current=completed_configurations,
                    total=total_configurations,
                    case_folder=case_dir.name,
                    particle=particle,
                    range_mm=float(range_mm),
                    configuration=configuration,
                    input_file=input_file.name,
                    status="Running",
                )


                try:
                    result = run_single_case(
                        output_map[vacuum_key],
                        output_map[air_key],
                        params,
                    )

                    if "export_table_scatterers" not in result:
                        raise KeyError(
                            "run_whole output has no 'export_table_scatterers' key."
                        )

                    table = result["export_table_scatterers"].copy()

                    if table.empty:
                        raise ValueError("export_table_scatterers is empty.")

                    required_columns = {
                        "Scatterers",
                        "Contribution in x' [mrad]",
                        "Contribution in y' [mrad]",
                        "Contribution in theta [mrad]",
                        "Contribution in energy [MeV]",
                        "Equivalent x (MCS) in x' [mm]",
                        "Equivalent x (MCS) in y' [mm]",
                        "Equivalent x (MCS) in theta [mm]",
                        "Equivalent x (Landau) in energy [mm]",
                    }

                    missing_columns = sorted(
                        required_columns.difference(table.columns)
                    )

                    if missing_columns:
                        raise KeyError(
                            "Scatterer table misses columns: "
                            + ", ".join(missing_columns)
                        )

                    table["Scatterer"] = table["Scatterers"].map(
                        _normalise_scatterer_name
                    )
                    table["Particle"] = (
                        "Protons" if particle == "proton" else "Carbon ions"
                    )
                    table["Configuration"] = configuration
                    table["Range [mm]"] = float(range_mm)
                    table["Kinetic energy total [MeV]"] = float(
                        beam_parameters["kinetic_energy"]
                    )
                    table["Input energy [MeV/u]"] = beam_parameters["energy_per_u_mev"]
                    table["Input file"] = input_file.name
                    table["Case folder"] = case_dir.name
                    table["Vacuum file"] = output_map[vacuum_key].name
                    table["Air file"] = output_map[air_key].name

                    all_scatterer_tables.append(table)
                    print(f"Completed: {configuration}")


                    completed_configurations += 1

                    _notify_progress(
                        progress_callback,
                        event="completed",
                        current=completed_configurations,
                        total=total_configurations,
                        case_folder=case_dir.name,
                        particle=particle,
                        range_mm=float(range_mm),
                        configuration=configuration,
                        input_file=input_file.name,
                        status="Completed",
                    )

                except Exception as exc:
                    errors.append({
                        "case_folder": case_dir.name,
                        "particle": particle,
                        "range_mm": range_mm,
                        "configuration": configuration,
                        "stage": "run_whole",
                        "error": repr(exc),
                    })
                    print(f"ERROR in {case_dir.name} | {configuration}: {exc}")


                    completed_configurations += 1

                    _notify_progress(
                        progress_callback,
                        event="failed",
                        current=completed_configurations,
                        total=total_configurations,
                        case_folder=case_dir.name,
                        particle=particle,
                        range_mm=float(range_mm),
                        configuration=configuration,
                        input_file=input_file.name,
                        status=f"Failed: {exc}",
                    )

                    if not continue_on_error:
                        raise

    cases_table = pd.DataFrame(case_metadata)
    errors_table = pd.DataFrame(errors)

    if all_scatterer_tables:
        all_scatterer_table = pd.concat(all_scatterer_tables, ignore_index=True)
        all_scatterer_table = all_scatterer_table.rename(columns={
            "Contribution in x' [mrad]": "contr_xp",
            "Contribution in y' [mrad]": "contr_yp",
            "Contribution in theta [mrad]": "contr_theta",
            "Contribution in energy [MeV]": "contr_energy",
            "Equivalent x (MCS) in x' [mm]": "WET_xp",
            "Equivalent x (MCS) in y' [mm]": "WET_yp",
            "Equivalent x (MCS) in theta [mm]": "WET_theta",
            "Equivalent x (Landau) in energy [mm]": "WET_energy",
        })

        all_scatterer_table = all_scatterer_table.sort_values(
            ["Scatterer", "Particle", "Configuration", "Range [mm]"]
        ).reset_index(drop=True)
    else:
        all_scatterer_table = pd.DataFrame()
    

    _notify_progress(
        progress_callback,
        event="finished",
        current=completed_configurations,
        total=total_configurations,
        case_folder="",
        particle="",
        range_mm=None,
        configuration="",
        status="Batch analysis completed",
    )




    return {
        "all_scatterer_table": all_scatterer_table,
        "cases_table": cases_table,
        "errors": errors_table,
    }