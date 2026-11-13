"""Load the raw crop dataset and create reusable exploratory-analysis figures."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.preprocessing import DEFAULT_DATASET, FEATURES, REQUIRED_COLUMNS

FIGURE_FILENAMES = (
    "missing_values.png",
    "feature_distribution.png",
    "feature_boxplot.png",
    "correlation_heatmap.png",
)

FEATURE_LABELS = {
    "N": "Nitrogen (N)",
    "P": "Phosphorus (P)",
    "K": "Potassium (K)",
    "temperature": "Temperature (°C)",
    "humidity": "Humidity (%)",
    "ph": "pH",
    "rainfall": "Rainfall (mm)",
}


def load_eda_dataset(path=DEFAULT_DATASET):
    """Load the original CSV without dropping duplicates or filling missing cells."""
    data = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="raise")
    return data


def ensure_eda_figures(data, source_mtime_ns, output_dir):
    """Reuse current EDA figures, regenerating only when missing or stale."""
    output_dir = Path(output_dir)
    figures = {name: output_dir / name for name in FIGURE_FILENAMES}
    if all(
        path.is_file() and path.stat().st_mtime_ns >= source_mtime_ns
        for path in figures.values()
    ):
        return figures

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    _save_missing_values_figure(data, figures["missing_values.png"])
    _save_feature_distribution_figure(data, figures["feature_distribution.png"])
    _save_boxplot_figure(data, figures["feature_boxplot.png"])
    _save_correlation_figure(data, figures["correlation_heatmap.png"])
    return figures


def _save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_missing_values_figure(data, path):
    counts = data.isna().sum()
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(counts.index, counts.values, color="#197449")
    for bar, count in zip(bars, counts.values):
        ax.annotate(
            str(int(count)),
            (bar.get_x() + bar.get_width() / 2, count),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    ax.set(
        title="Missing Values by Dataset Column",
        xlabel="Dataset column",
        ylabel="Missing-value count",
    )
    ax.tick_params(axis="x", labelrotation=35)
    ax.set_ylim(0, max(1, int(counts.max()) * 1.15))
    if counts.sum() == 0:
        ax.text(
            0.5,
            0.94,
            "No missing values found",
            transform=ax.transAxes,
            ha="center",
            va="top",
            color="#197449",
            fontweight="bold",
        )
    _save_figure(fig, path)


def _save_feature_distribution_figure(data, path):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, feature in zip(axes.flat, FEATURES):
        ax.hist(data[feature].dropna(), bins=20, color="#4c956c", edgecolor="white")
        ax.set(
            title=f"Distribution of {FEATURE_LABELS[feature]}",
            xlabel=FEATURE_LABELS[feature],
            ylabel="Count",
        )
    for ax in axes.flat[len(FEATURES):]:
        ax.set_visible(False)
    _save_figure(fig, path)


def _save_boxplot_figure(data, path):
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=data[FEATURES], color="#79b791", ax=ax)
    ax.set(
        title="Distribution and Potential Outliers of Numerical Features",
        xlabel="Feature",
        ylabel="Feature value",
    )
    ax.set_xticks(
        range(len(FEATURES)),
        [FEATURE_LABELS[feature] for feature in FEATURES],
        rotation=20,
        ha="right",
    )
    _save_figure(fig, path)


def _save_correlation_figure(data, path):
    correlations = data[FEATURES].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlations,
        annot=True,
        fmt=".2f",
        cmap="YlGn",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation"},
        ax=ax,
    )
    ax.set(title="Pearson Correlation Between Numerical Input Features")
    _save_figure(fig, path)
