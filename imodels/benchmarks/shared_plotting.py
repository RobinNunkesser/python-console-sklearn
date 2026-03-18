"""Shared plotting helpers for benchmark result CSVs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TITLE_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10
LEGEND_FONTSIZE = 10
LEGEND_TITLE_FONTSIZE = 11


@dataclass(frozen=True)
class FontConfig:
    title: int = TITLE_FONTSIZE
    axis_label: int = AXIS_LABEL_FONTSIZE
    tick: int = TICK_FONTSIZE
    legend: int = LEGEND_FONTSIZE
    legend_title: int = LEGEND_TITLE_FONTSIZE


@dataclass(frozen=True)
class FigureConfig:
    # Letter-friendly combined layout.
    combined_figsize: tuple[float, float] = (8.5, 11.0)
    combined_width: float = 8.5
    combined_height_per_dataset: float = 1
    combined_min_height: float = 3.0
    separate_figsize: tuple[float, float] = (8.0, 5.4)
    by_dataset_width_per_metric: float = 4.8
    by_dataset_legend_width: float = 2.8
    by_dataset_row_height: float = 2.4
    by_dataset_min_height: float = 5.8
    combined_legend_width_ratio: float = 0.34
    combined_y_margin: float = 0.01
    separate_height_ratios: tuple[float, float] = (0.2, 1.0)


@dataclass(frozen=True)
class MarkerConfig:
    dot_offsets_min: float = -0.3
    dot_offsets_max: float = 0.3
    scatter_size: float = 36.0
    errorbar_marker_size: float = 5.0
    errorbar_capsize: float = 4.0
    errorbar_linewidth: float = 1.2
    log_clip_fraction: float = 0.9999


@dataclass(frozen=True)
class LegendConfig:
    side_ncol: int = 1
    max_inline_ncol: int = 4
    column_spacing: float = 1.2
    handle_text_pad: float = 0.6


@dataclass(frozen=True)
class StyleConfig:
    dataset_band_facecolor: str = "0.96"
    dataset_band_fill_fraction: float = 1.0
    grid_alpha: float = 0.3


@dataclass(frozen=True)
class PlotConfig:
    fonts: FontConfig = field(default_factory=FontConfig)
    figure: FigureConfig = field(default_factory=FigureConfig)
    markers: MarkerConfig = field(default_factory=MarkerConfig)
    legend: LegendConfig = field(default_factory=LegendConfig)
    style: StyleConfig = field(default_factory=StyleConfig)


DEFAULT_PLOT_CONFIG = PlotConfig()

UCI_METRICS: list[dict[str, str]] = [
    {
        "metric": "f1",
        "label": "F1",
        "title": "F1 by Dataset and Algorithm",
        "separate_title": "F1 by Dataset and Algorithm",
        "xscale": "linear",
    },
    {
        "metric": "model_size",
        "label": "Model Size",
        "title": "Model Size by Dataset and Algorithm",
        "separate_title": "Model Size",
        "xscale": "log",
    },
]

ALGORITHM_SHORT_NAMES: dict[str, str] = {
    # imodels
    "C45TreeClassifier": "C4.5",
    "DecisionTreeClassifier": "DT",
    "FIGSClassifier": "FIGS",
    "GreedyRuleListClassifier": "GRL",
    "GreedyTreeClassifier": "CART",
    "HSTreeClassifier": "HS",
    "OneRClassifier": "OneR",
    "SlipperClassifier": "Slipper",
    "TaoTreeClassifier": "Tao",
    # ExSTraCS variants
    "ExSTraCS": "ExS",
    "ExSTraCS_FU1": "ExS-F1",
    "ExSTraCS_FU2": "ExS-F2",
    "ExSTraCS_QRF": "ExSTraCS",
}


def _algorithm_display_name(name: str) -> str:
    return ALGORITHM_SHORT_NAMES.get(name, name)


def _algorithm_display_map(algorithm_names: list[str]) -> dict[str, str]:
    base_names = {name: _algorithm_display_name(name) for name in algorithm_names}
    counts: dict[str, int] = {}
    for base in base_names.values():
        counts[base] = counts.get(base, 0) + 1

    display_map: dict[str, str] = {}
    for name in algorithm_names:
        base = base_names[name]
        # Keep short names concise, but disambiguate collisions safely.
        if counts[base] > 1:
            display_map[name] = f"{base} ({name})"
        else:
            display_map[name] = base
    return display_map


def add_figure_legend(
    legend_ax: plt.Axes,
    source_ax: plt.Axes,
    *,
    side: bool = False,
    config: PlotConfig = DEFAULT_PLOT_CONFIG,
) -> None:
    handles, labels = source_ax.get_legend_handles_labels()
    unique_entries: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label and not label.startswith("_") and label not in unique_entries:
            unique_entries[label] = handle

    legend_ax.axis("off")
    if not unique_entries:
        return

    ncol = config.legend.side_ncol if side else min(len(unique_entries), config.legend.max_inline_ncol)
    legend_ax.legend(
        unique_entries.values(),
        unique_entries.keys(),
        loc="center left" if side else "center",
        ncol=ncol,
        frameon=False,
        title="Algorithm",
        fontsize=config.fonts.legend,
        title_fontsize=config.fonts.legend_title,
        columnspacing=config.legend.column_spacing,
        handletextpad=config.legend.handle_text_pad,
    )


def add_dataset_background_bands(ax: plt.Axes, *, config: PlotConfig = DEFAULT_PLOT_CONFIG) -> None:
    ticks = sorted({float(tick) for tick in ax.get_yticks()})
    if not ticks:
        return

    if len(ticks) == 1:
        bounds = [ticks[0] - 0.5, ticks[0] + 0.5]
    else:
        midpoints = [(left + right) / 2 for left, right in zip(ticks, ticks[1:])]
        first_half_step = (ticks[1] - ticks[0]) / 2
        last_half_step = (ticks[-1] - ticks[-2]) / 2
        bounds = [ticks[0] - first_half_step, *midpoints, ticks[-1] + last_half_step]

    for idx, (lower, upper) in enumerate(zip(bounds, bounds[1:])):
        if idx % 2 == 0:
            center = (lower + upper) / 2
            half_height = (upper - lower) * 0.5 * config.style.dataset_band_fill_fraction
            band_lower = center - half_height
            band_upper = center + half_height
            ax.axhspan(band_lower, band_upper, facecolor=config.style.dataset_band_facecolor, edgecolor="none", zorder=-1)


def save_figure_outputs(fig: plt.Figure, output_base_path: Path) -> None:
    png_path = output_base_path.with_suffix(".png")
    pdf_path = output_base_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    print(f"Figure saved: {png_path}")
    print(f"Figure saved: {pdf_path}")


def plot_metric_panel(
    df: pd.DataFrame,
    *,
    dataset_label_col: str,
    metric: str,
    ax: plt.Axes,
    title: str,
    xlabel: str,
    error_bars: str,
    plot_style: str,
    xscale: str = "linear",
    config: PlotConfig = DEFAULT_PLOT_CONFIG,
) -> None:
    mean_col = f"{metric}_mean"
    if mean_col not in df.columns:
        raise ValueError(f"Missing required column for plotting: {mean_col}")

    raw_pivot = df.pivot(index=dataset_label_col, columns="algorithm", values=mean_col)
    display_map = _algorithm_display_map(list(raw_pivot.columns))
    pivot = raw_pivot.rename(columns=display_map)

    err_data = None
    if error_bars != "none":
        err_col = f"{metric}_{error_bars}"
        if err_col in df.columns:
            err_data = df.pivot(index=dataset_label_col, columns="algorithm", values=err_col).reindex_like(raw_pivot)
            err_data = err_data.rename(columns=display_map).reindex_like(pivot)
            if xscale == "log":
                err_data = err_data.clip(upper=pivot * config.markers.log_clip_fraction)

    if plot_style == "bars":
        kwargs: dict[str, Any] = {"kind": "barh", "ax": ax, "legend": False}
        if err_data is not None:
            kwargs["xerr"] = err_data
            kwargs["capsize"] = config.markers.errorbar_capsize
        pivot.plot(**kwargs)
    elif plot_style == "dots":
        datasets = list(pivot.index)
        algorithms = list(pivot.columns)
        y_base = np.arange(len(datasets), dtype=float)
        offsets = (
            np.array([0.0])
            if len(algorithms) <= 1
            else np.linspace(config.markers.dot_offsets_min, config.markers.dot_offsets_max, num=len(algorithms))
        )

        colors = plt.rcParams.get("axes.prop_cycle", None)
        palette = colors.by_key().get("color", []) if colors is not None else []

        for idx, algorithm in enumerate(algorithms):
            color = palette[idx % len(palette)] if palette else None
            x = pivot[algorithm].to_numpy(dtype=float)
            y = y_base + offsets[idx]
            mask = ~np.isnan(x)
            if not mask.any():
                continue

            if err_data is not None and algorithm in err_data.columns:
                err_frame = cast(pd.DataFrame, err_data)
                err = np.nan_to_num(err_frame[algorithm].to_numpy(dtype=float), nan=0.0)
                ax.errorbar(
                    x[mask],
                    y[mask],
                    xerr=err[mask],
                    fmt="o",
                    capsize=config.markers.errorbar_capsize,
                    markersize=config.markers.errorbar_marker_size,
                    elinewidth=config.markers.errorbar_linewidth,
                    linewidth=config.markers.errorbar_linewidth,
                    color=color,
                    label=algorithm,
                )
            else:
                ax.scatter(x[mask], y[mask], s=config.markers.scatter_size, color=color, label=algorithm)

        ax.set_yticks(y_base, labels=datasets)
        ax.invert_yaxis()
    else:
        raise ValueError(f"Unknown plot_style: {plot_style}")

    ax.set_xscale(xscale)
    ax.set_title(title, fontsize=config.fonts.title)
    ax.set_xlabel(xlabel, fontsize=config.fonts.axis_label)
    ax.set_ylabel("Dataset", fontsize=config.fonts.axis_label)
    ax.tick_params(axis="both", labelsize=config.fonts.tick)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=config.style.grid_alpha)
    add_dataset_background_bands(ax, config=config)


def plot_benchmark_results(
    df: pd.DataFrame,
    *,
    dataset_label_col: str,
    metrics: list[dict[str, str]],
    output_dir: Path,
    output_basename_prefix: str,
    plot_mode: str,
    error_bars: str,
    plot_style: str,
    no_show: bool,
    config: PlotConfig = DEFAULT_PLOT_CONFIG,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if plot_mode == "by_dataset":
        dataset_values = [v for v in df[dataset_label_col].tolist() if pd.notna(v)]
        datasets = list(dict.fromkeys(dataset_values))
        if not datasets:
            return

        n_metrics = len(metrics)
        if n_metrics == 0:
            raise ValueError("metrics must not be empty for by_dataset mode")

        fig_width = config.figure.by_dataset_width_per_metric * n_metrics + config.figure.by_dataset_legend_width
        fig_height = max(config.figure.by_dataset_min_height, config.figure.by_dataset_row_height * len(datasets))
        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
        grid = fig.add_gridspec(len(datasets), n_metrics + 1, width_ratios=[1.0] * n_metrics + [0.32])
        legend_ax = fig.add_subplot(grid[:, n_metrics])

        first_data_ax: plt.Axes | None = None
        for row_idx, dataset in enumerate(datasets):
            subset = df[df[dataset_label_col] == dataset]
            for col_idx, spec in enumerate(metrics):
                ax = fig.add_subplot(grid[row_idx, col_idx])
                if first_data_ax is None:
                    first_data_ax = ax
                label = spec["label"] if error_bars == "none" else f"{spec['label']} (mean +/- {error_bars})"
                panel_title = f"{dataset} - {spec['label']}"
                plot_metric_panel(
                    subset,
                    dataset_label_col=dataset_label_col,
                    metric=spec["metric"],
                    ax=ax,
                    title=panel_title,
                    xlabel=label,
                    error_bars=error_bars,
                    plot_style=plot_style,
                    xscale=spec.get("xscale", "linear"),
                    config=config,
                )

        if first_data_ax is not None:
            add_figure_legend(legend_ax, first_data_ax, side=True, config=config)
        save_figure_outputs(fig, output_dir / f"{output_basename_prefix}_by_dataset")
        if no_show:
            plt.close(fig)
        else:
            plt.show()
        return

    if plot_mode == "combined":
        # Calculate height based on number of datasets
        dataset_values = [v for v in df[dataset_label_col].tolist() if pd.notna(v)]
        n_datasets = len(list(dict.fromkeys(dataset_values)))
        combined_height = max(
            config.figure.combined_min_height,
            n_datasets * config.figure.combined_height_per_dataset
        )
        combined_figsize = (config.figure.combined_width, combined_height)
        
        fig = plt.figure(figsize=combined_figsize, constrained_layout=True)
        grid = fig.add_gridspec(
            1,
            len(metrics) + 1,
            width_ratios=[1.0] * len(metrics) + [config.figure.combined_legend_width_ratio],
        )
        axes = [fig.add_subplot(grid[0, idx]) for idx in range(len(metrics))]
        legend_ax = fig.add_subplot(grid[0, len(metrics)])

        for ax, spec in zip(axes, metrics):
            label = spec["label"] if error_bars == "none" else f"{spec['label']} (mean +/- {error_bars})"
            plot_metric_panel(
                df,
                dataset_label_col=dataset_label_col,
                metric=spec["metric"],
                ax=ax,
                title=spec["title"],
                xlabel=label,
                error_bars=error_bars,
                plot_style=plot_style,
                xscale=spec.get("xscale", "linear"),
                config=config,
            )
            # Combined layout: keep panels compact and avoid redundant labels.
            ax.set_title("")
            ax.set_ylabel("")
            ax.margins(y=config.figure.combined_y_margin)
            if spec.get("metric") == "model_size":
                # In combined view the right model-size panel does not need dataset ticks/label.
                ax.set_ylabel("")
                ax.set_yticks([])
                ax.tick_params(axis="y", left=False, labelleft=False)

        add_figure_legend(legend_ax, axes[0], side=True, config=config)
        save_figure_outputs(fig, output_dir / f"{output_basename_prefix}_combined")
        if no_show:
            plt.close(fig)
        else:
            plt.show()
        return

    if plot_mode == "separate":
        figures: list[plt.Figure] = []
        for spec in metrics:
            fig = plt.figure(figsize=config.figure.separate_figsize, constrained_layout=True)
            grid = fig.add_gridspec(2, 1, height_ratios=list(config.figure.separate_height_ratios))
            legend_ax = fig.add_subplot(grid[0, 0])
            ax = fig.add_subplot(grid[1, 0])
            label = spec["label"] if error_bars == "none" else f"{spec['label']} (mean +/- {error_bars})"
            plot_metric_panel(
                df,
                dataset_label_col=dataset_label_col,
                metric=spec["metric"],
                ax=ax,
                title=spec["separate_title"],
                xlabel=label,
                error_bars=error_bars,
                plot_style=plot_style,
                xscale=spec.get("xscale", "linear"),
                config=config,
            )
            add_figure_legend(legend_ax, ax, config=config)
            save_figure_outputs(fig, output_dir / f"{output_basename_prefix}_{spec['metric']}")
            figures.append(fig)

        if no_show:
            for fig in figures:
                plt.close(fig)
        else:
            plt.show()
        return

    raise ValueError(f"Unknown plot_mode: {plot_mode}")

