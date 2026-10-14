from __future__ import annotations

import argparse
import importlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


LOG = logging.getLogger("cluster_job")


def safe_filename(value: object) -> str:
    """Convert a label into a portable filename component."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return name.strip("._") or "unnamed"


def save_figure(fig: Any, path: Path, dpi: int = 180) -> bool:
    """Save and close one matplotlib figure; return True if it was saved."""
    if fig is None or not hasattr(fig, "savefig"):
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def write_batch_summaries(summary: dict[str, Any], output_root: Path) -> None:
    """Write the global tables returned by run_whole_all_energies."""
    output_root.mkdir(parents=True, exist_ok=True)

    table_keys = {
        "all_scatterer_table": "all_scatterer_table.csv",
        "cases_table": "cases_table.csv",
        "errors": "errors.csv",
    }

    for key, filename in table_keys.items():
        table = summary.get(key)

        if not isinstance(table, pd.DataFrame):
            raise TypeError(
                f"Batch summary key {key!r} must contain a pandas DataFrame; "
                f"received {type(table).__name__}."
            )

        table.to_csv(output_root / filename, index=False)


def write_batch_plots(summary: dict[str, Any], output_root: Path) -> int:
    """
    Save global figures produced by run_whole_all_energies.

    Expected summary entries:
    - contribution_plots: {scatterer_name: matplotlib.figure.Figure}
    - wet_plots: {scatterer_name: matplotlib.figure.Figure}
    """
    plot_root = output_root / "batch_plots"
    plot_root.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    plot_groups = {
        "contributions": summary.get("contribution_plots", {}),
        "wet": summary.get("wet_plots", {}),
    }

    for group_name, figures in plot_groups.items():
        if figures is None:
            continue

        if not isinstance(figures, dict):
            LOG.warning(
                "Expected a dictionary for summary[%r], received %s; skipping.",
                f"{group_name}_plots",
                type(figures).__name__,
            )
            continue

        for label, fig in figures.items():
            output_path = plot_root / (
                f"{group_name}_{safe_filename(label)}.png"
            )

            if save_figure(fig, output_path):
                saved_count += 1
                LOG.info("Saved batch plot: %s", output_path)

    return saved_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run all proton/carbon FLUKA cases using the non-interactive "
            "analysis library and save CSV summaries plus global batch plots."
        )
    )

    parser.add_argument(
        "root_dir",
        type=Path,
        help="Directory containing PROTON/ and/or CARBON/ case directories.",
    )

    parser.add_argument(
        "parameters_json",
        type=Path,
        help="JSON file containing common base_parameters.",
    )

    parser.add_argument(
        "--module",
        default="analysis_library",
        help=(
            "Python module defining run_whole_all_energies "
            "(default: analysis_library). Do not include '.py'."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis_results"),
        help=(
            "Root directory for CSV summaries, global plots and any "
            "case-level outputs written by the library. This command-line "
            "value overrides batch_output_root in the JSON."
        ),
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when one case/configuration fails.",
    )

    parser.add_argument(
        "--overwrite-gz",
        action="store_true",
        help="Regenerate decompressed .23 files even if they already exist.",
    )

    parser.add_argument(
        "--fail-if-any-error",
        action="store_true",
        help=(
            "Complete all possible cases and write all outputs, then exit "
            "with status 1 if at least one case failed."
        ),
    )
    parser.add_argument(
        "--nRows",
        type = int,
        default = None,
        help = (
            "Total number of rows to be read in the MyLip.load_data"
            "If omitted, the whole dataset is read"
        )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    root_dir = args.root_dir.expanduser().resolve()
    parameters_json = args.parameters_json.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not root_dir.is_dir():
        raise NotADirectoryError(
            f"Input root directory does not exist: {root_dir}"
        )

    if not parameters_json.is_file():
        raise FileNotFoundError(
            f"Parameters JSON does not exist: {parameters_json}"
        )

    with parameters_json.open("r", encoding="utf-8") as handle:
        base_parameters = json.load(handle)
    base_parameters["nRows"]=args.nRows
    if base_parameters["nRows"] is None:
        print("[INFO] The whole dataset is read.")
    else:
        print(
            "[INFO] Only: "
            f"{constant_parameters['nRows']} rows are read."
        )

    if not isinstance(base_parameters, dict):
        raise TypeError("The parameters JSON must contain one JSON object.")

    output_root.mkdir(parents=True, exist_ok=True)

    # The library can use these entries to place case-level tables and plots
    # below the same output-root directory.
    base_parameters["batch_output_root"] = str(output_root)
    base_parameters["save_plots"] = True

    analysis = importlib.import_module(args.module)

    if not hasattr(analysis, "run_whole_all_energies"):
        raise AttributeError(
            f"Module {args.module!r} does not define run_whole_all_energies."
        )

    LOG.info("Input root: %s", root_dir)
    LOG.info("Parameters JSON: %s", parameters_json)
    LOG.info("Output root: %s", output_root)
    LOG.info("Analysis module: %s", args.module)

    summary = analysis.run_whole_all_energies(
        root_dir=root_dir,
        base_parameters=base_parameters,
        continue_on_error=not args.stop_on_error,
        overwrite_gz=args.overwrite_gz,
    )

    if not isinstance(summary, dict):
        raise TypeError(
            "run_whole_all_energies must return a dictionary."
        )

    write_batch_summaries(summary, output_root)
    n_plots = write_batch_plots(summary, output_root)

    cases_table = summary.get("cases_table", pd.DataFrame())
    errors_table = summary.get("errors", pd.DataFrame())
    scatterer_table = summary.get("all_scatterer_table", pd.DataFrame())

    n_cases = len(cases_table)
    n_errors = len(errors_table)
    n_rows = len(scatterer_table)

    LOG.info(
        "Batch completed: %d discovered case(s), %d scatterer-table row(s), "
        "%d error(s), %d global plot(s) saved.",
        n_cases,
        n_rows,
        n_errors,
        n_plots,
    )
    LOG.info("Global CSV files written to %s", output_root)
    LOG.info("Global batch plots written to %s", output_root / "batch_plots")

    if n_errors and args.fail_if_any_error:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
