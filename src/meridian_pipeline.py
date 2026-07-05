"""
Bayesian Marketing Mix Model with Google Meridian.

This pipeline keeps the original Ridge workflow intact and adds a Meridian
Bayesian MMM on the same 200-week national dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from meridian import constants
from meridian.analysis import visualizer
from meridian.data import data_frame_input_data_builder
from meridian.model import model
from meridian.model import prior_distribution
from meridian.model import spec


CHANNELS = ["TikTok", "Facebook", "Google Ads"]
CONTROL_COLS = ["trend", "week_sin", "week_cos"]
ROI_PRIOR_MEAN = 0.45
ROI_PRIOR_SD = 0.45
MEDIA_COLS = {
    "TikTok": "TikTok_media",
    "Facebook": "Facebook_media",
    "Google Ads": "Google_Ads_media",
}
SPEND_COLS = {
    "TikTok": "TikTok_spend",
    "Facebook": "Facebook_spend",
    "Google Ads": "Google_Ads_spend",
}

DATA_PATH = ROOT / "data" / "marketing_mix.csv"
MERIDIAN_INPUT_PATH = ROOT / "data" / "meridian_input.csv"
OUT_DIR = ROOT / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit the Meridian Bayesian MMM.")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--adapt", type=int, default=100)
    parser.add_argument("--burnin", type=int, default=50)
    parser.add_argument("--keep", type=int, default=150)
    parser.add_argument("--prior-draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--max-tree-depth", type=int, default=6)
    parser.add_argument("--confidence-level", type=float, default=0.9)
    return parser.parse_args()


def configure_runtime() -> None:
    (ROOT / ".matplotlib").mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    warnings.filterwarnings(
        "ignore",
        message="Revenue from the `kpi` data is used when `kpi_type`=`revenue`.*",
    )
    warnings.filterwarnings(
        "ignore",
        message="The `population` argument is ignored in a nationally aggregated model.*",
    )


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    required = {"Date", "Sales", *CHANNELS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def prepare_meridian_input(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=raw_df.index)
    df["geo"] = "national_geo"
    df["time"] = raw_df["Date"].dt.strftime("%Y-%m-%d")
    df["kpi"] = raw_df["Sales"].astype(float)
    df["population"] = 1.0

    week_index = np.arange(len(raw_df), dtype=float)
    df["trend"] = week_index / max(len(raw_df) - 1, 1)
    df["week_sin"] = np.sin(2 * np.pi * week_index / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * week_index / 52.0)

    for channel in CHANNELS:
        spend = raw_df[channel].astype(float)
        df[MEDIA_COLS[channel]] = spend
        df[SPEND_COLS[channel]] = spend

    df.to_csv(MERIDIAN_INPUT_PATH, index=False)
    return df


def build_input_data(meridian_df: pd.DataFrame):
    # Meridian's DataFrame builder handles national data cleanly when it creates
    # the single geo internally. The exported CSV still includes geo for clarity.
    builder_df = meridian_df.drop(columns=["geo"])
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=constants.REVENUE,
        default_time_column="time",
        default_kpi_column="kpi",
        default_population_column="population",
    )
    return (
        builder.with_kpi(builder_df)
        .with_population(builder_df)
        .with_controls(builder_df, CONTROL_COLS, time_col="time")
        .with_media(
            builder_df,
            media_cols=[MEDIA_COLS[channel] for channel in CHANNELS],
            media_spend_cols=[SPEND_COLS[channel] for channel in CHANNELS],
            media_channels=CHANNELS,
            time_col="time",
        )
        .build()
    )


def build_model(input_data):
    roi_prior = prior_distribution.lognormal_dist_from_mean_std(
        mean=np.repeat(np.float32(ROI_PRIOR_MEAN), len(CHANNELS)),
        std=np.repeat(np.float32(ROI_PRIOR_SD), len(CHANNELS)),
    )
    priors = prior_distribution.PriorDistribution(roi_m=roi_prior)
    model_spec = spec.ModelSpec(
        prior=priors,
        knots=12,
        max_lag=8,
        media_prior_type=constants.TREATMENT_PRIOR_TYPE_ROI,
        adstock_decay_spec=constants.GEOMETRIC_DECAY,
        saturation_spec=constants.HILL,
    )
    return model.Meridian(input_data=input_data, model_spec=model_spec)


def flatten_rhat(rhat_dataset) -> np.ndarray:
    values = []
    for data_array in rhat_dataset.data_vars.values():
        array = np.asarray(data_array.values, dtype=float).ravel()
        values.extend(array[np.isfinite(array)])
    return np.asarray(values, dtype=float)


def compute_diagnostics(mmm: model.Meridian) -> dict:
    diagnostics: dict[str, object] = {
        "groups": list(mmm.inference_data.groups()),
    }
    try:
        rhat_values = flatten_rhat(az.rhat(mmm.inference_data.posterior))
        diagnostics["rhat"] = {
            "max": float(np.max(rhat_values)) if rhat_values.size else None,
            "mean": float(np.mean(rhat_values)) if rhat_values.size else None,
            "n_parameters_over_1_10": int(np.sum(rhat_values > 1.10)),
            "n_parameters_over_1_20": int(np.sum(rhat_values > 1.20)),
        }
    except Exception as exc:  # pragma: no cover - diagnostic best effort.
        diagnostics["rhat_error"] = str(exc)

    sample_stats = getattr(mmm.inference_data, "sample_stats", None)
    if sample_stats is not None and "diverging" in sample_stats:
        diagnostics["divergences"] = int(np.asarray(sample_stats["diverging"]).sum())
    else:
        diagnostics["divergences"] = None

    return diagnostics


def posterior_metric(summary_ds, channel: str, variable: str, metric: str) -> float:
    return float(
        summary_ds[variable]
        .sel(channel=channel, distribution=constants.POSTERIOR, metric=metric)
        .item()
    )


def export_channel_summary(
    summary_ds,
    raw_df: pd.DataFrame,
    confidence_level: float,
) -> pd.DataFrame:
    rows = []
    for channel in CHANNELS + [constants.ALL_CHANNELS]:
        row = {
            "channel": channel,
            "total_spend": float(summary_ds["spend"].sel(channel=channel).item()),
        }
        for variable in ["incremental_outcome", "roi", "pct_of_contribution"]:
            for metric in [constants.MEAN, constants.CI_LO, constants.CI_HI]:
                row[f"{variable}_{metric}"] = posterior_metric(
                    summary_ds, channel, variable, metric
                )
        rows.append(row)

    out = pd.DataFrame(rows)
    total_sales = float(raw_df["Sales"].sum())
    total_media_pct = float(
        out.loc[out["channel"] == constants.ALL_CHANNELS, "pct_of_contribution_mean"].iloc[0]
    )
    total_media_incremental = float(
        out.loc[out["channel"] == constants.ALL_CHANNELS, "incremental_outcome_mean"].iloc[0]
    )
    baseline_row = {
        "channel": "Baseline (non-media)",
        "total_spend": np.nan,
        "incremental_outcome_mean": max(total_sales - total_media_incremental, 0.0),
        "incremental_outcome_ci_lo": np.nan,
        "incremental_outcome_ci_hi": np.nan,
        "roi_mean": np.nan,
        "roi_ci_lo": np.nan,
        "roi_ci_hi": np.nan,
        "pct_of_contribution_mean": max(100.0 - total_media_pct, 0.0),
        "pct_of_contribution_ci_lo": np.nan,
        "pct_of_contribution_ci_hi": np.nan,
    }
    out = pd.concat([out, pd.DataFrame([baseline_row])], ignore_index=True)
    out["credible_interval"] = f"{int(confidence_level * 100)}%"
    out.to_csv(OUT_DIR / "meridian_channel_contribution_summary.csv", index=False)
    return out


def export_response_curves(mmm: model.Meridian, confidence_level: float, n_weeks: int) -> pd.DataFrame:
    media_effects = visualizer.MediaEffects(mmm)
    response_ds = media_effects.response_curves_data(confidence_level=confidence_level)

    outcome = (
        response_ds["incremental_outcome"]
        .to_dataframe()
        .reset_index()
        .pivot(
            index=["spend_multiplier", "channel"],
            columns="metric",
            values="incremental_outcome",
        )
        .reset_index()
    )
    spend = response_ds["spend"].to_dataframe().reset_index()
    curves = outcome.merge(spend, on=["spend_multiplier", "channel"], how="left")
    curves["weekly_spend"] = curves["spend"] / n_weeks
    curves = curves.rename(
        columns={
            constants.MEAN: "incremental_outcome_mean",
            constants.CI_LO: "incremental_outcome_ci_lo",
            constants.CI_HI: "incremental_outcome_ci_hi",
        }
    )
    curves.to_csv(OUT_DIR / "meridian_response_curves.csv", index=False)
    with open(OUT_DIR / "meridian_response_curves.json", "w", encoding="utf-8") as f:
        json.dump(curves.to_dict(orient="records"), f, indent=2)
    return curves


def load_ridge_summary() -> pd.DataFrame:
    ridge_path = OUT_DIR / "channel_contribution_summary.csv"
    ridge = pd.read_csv(ridge_path, index_col=0)
    roi_col = [col for col in ridge.columns if "ROI" in col][0]
    contribution_col = [col for col in ridge.columns if "estimated_contribution" in col][0]
    pct_col = [col for col in ridge.columns if "contribution_%" in col][0]
    ridge = ridge.loc[CHANNELS, ["total_spend", contribution_col, pct_col, roi_col]].copy()
    ridge.columns = [
        "ridge_total_spend",
        "ridge_contribution",
        "ridge_pct_of_sales",
        "ridge_roi",
    ]
    ridge["channel"] = ridge.index
    return ridge.reset_index(drop=True)


def compare_with_ridge(meridian_summary: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ridge = load_ridge_summary()
    meridian = meridian_summary[meridian_summary["channel"].isin(CHANNELS)].copy()
    comparison = ridge.merge(meridian, on="channel", how="left")
    comparison = comparison[
        [
            "channel",
            "ridge_contribution",
            "incremental_outcome_mean",
            "incremental_outcome_ci_lo",
            "incremental_outcome_ci_hi",
            "ridge_roi",
            "roi_mean",
            "roi_ci_lo",
            "roi_ci_hi",
            "ridge_pct_of_sales",
            "pct_of_contribution_mean",
        ]
    ]
    comparison.to_csv(OUT_DIR / "ridge_vs_meridian_comparison.csv", index=False)

    ridge_rank = comparison.sort_values("ridge_roi", ascending=False)["channel"].tolist()
    meridian_rank = comparison.sort_values("roi_mean", ascending=False)["channel"].tolist()
    ranking = {
        "ridge_roi_rank_high_to_low": ridge_rank,
        "meridian_roi_rank_high_to_low": meridian_rank,
        "highest_roi_agrees": ridge_rank[0] == meridian_rank[0],
        "lowest_roi_agrees": ridge_rank[-1] == meridian_rank[-1],
        "ridge_highest_roi_channel": ridge_rank[0],
        "ridge_lowest_roi_channel": ridge_rank[-1],
        "meridian_highest_roi_channel": meridian_rank[0],
        "meridian_lowest_roi_channel": meridian_rank[-1],
    }
    return comparison, ranking


def plot_intervals(summary: pd.DataFrame, variable: str, ylabel: str, title: str, filename: str) -> None:
    plot_df = summary[summary["channel"].isin(CHANNELS)].copy()
    x = np.arange(len(plot_df))
    means = plot_df[f"{variable}_mean"].to_numpy(dtype=float)
    lows = plot_df[f"{variable}_ci_lo"].to_numpy(dtype=float)
    highs = plot_df[f"{variable}_ci_hi"].to_numpy(dtype=float)
    yerr = np.vstack([means - lows, highs - means])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, means, color="#4C78A8", alpha=0.85)
    ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="#222222", capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["channel"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=140)
    plt.close(fig)


def plot_response_curves(curves: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, channel in zip(axes, CHANNELS):
        data = curves[curves["channel"] == channel].sort_values("weekly_spend")
        x = data["weekly_spend"].to_numpy(dtype=float)
        mean = data["incremental_outcome_mean"].to_numpy(dtype=float)
        lo = data["incremental_outcome_ci_lo"].to_numpy(dtype=float)
        hi = data["incremental_outcome_ci_hi"].to_numpy(dtype=float)
        ax.plot(x, mean, color="#1F77B4")
        ax.fill_between(x, lo, hi, color="#1F77B4", alpha=0.18)
        current = data.loc[np.isclose(data["spend_multiplier"], 1.0), "weekly_spend"]
        if not current.empty:
            ax.axvline(float(current.iloc[0]), color="#D62728", linestyle="--", linewidth=1)
        ax.set_title(channel)
        ax.set_xlabel("Average weekly spend ($)")
        ax.set_ylabel("Total incremental sales")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "meridian_response_curves.png", dpi=140)
    plt.close(fig)


def plot_comparison(comparison: pd.DataFrame) -> None:
    x = np.arange(len(comparison))
    width = 0.35
    means = comparison["roi_mean"].to_numpy(dtype=float)
    lows = comparison["roi_ci_lo"].to_numpy(dtype=float)
    highs = comparison["roi_ci_hi"].to_numpy(dtype=float)
    yerr = np.vstack([means - lows, highs - means])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, comparison["ridge_roi"], width, label="Ridge", color="#F58518")
    ax.bar(x + width / 2, means, width, label="Meridian", color="#4C78A8")
    ax.errorbar(x + width / 2, means, yerr=yerr, fmt="none", ecolor="#222222", capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["channel"])
    ax.set_ylabel("ROI ($ sales / $ spend)")
    ax.set_title("Ridge vs Meridian ROI")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ridge_vs_meridian_roi.png", dpi=140)
    plt.close(fig)


def export_model_fit(mmm: model.Meridian, confidence_level: float) -> None:
    try:
        fit = visualizer.ModelFit(mmm, confidence_level=confidence_level).model_fit_data
        fit.to_dataframe().reset_index().to_csv(OUT_DIR / "meridian_model_fit.csv", index=False)
    except Exception as exc:  # pragma: no cover - diagnostics only.
        with open(OUT_DIR / "meridian_model_fit_error.txt", "w", encoding="utf-8") as f:
            f.write(str(exc))


def save_results(
    args: argparse.Namespace,
    raw_df: pd.DataFrame,
    elapsed_seconds: float,
    diagnostics: dict,
    meridian_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    ranking: dict,
) -> None:
    channel_rows = meridian_summary[meridian_summary["channel"].isin(CHANNELS)]
    results = {
        "model": "Google Meridian Bayesian MMM",
        "data": {
            "source": str(DATA_PATH.relative_to(ROOT)),
            "meridian_input": str(MERIDIAN_INPUT_PATH.relative_to(ROOT)),
            "n_weeks": int(len(raw_df)),
            "date_start": raw_df["Date"].min().strftime("%Y-%m-%d"),
            "date_end": raw_df["Date"].max().strftime("%Y-%m-%d"),
            "channels": CHANNELS,
            "media_execution_proxy": "Spend is duplicated as media execution because impressions/clicks are unavailable.",
            "controls": CONTROL_COLS,
        },
        "model_spec": {
            "knots": 12,
            "max_lag": 8,
            "media_prior_type": constants.TREATMENT_PRIOR_TYPE_ROI,
            "roi_prior_mean_per_channel": ROI_PRIOR_MEAN,
            "roi_prior_sd_per_channel": ROI_PRIOR_SD,
            "adstock_decay_spec": constants.GEOMETRIC_DECAY,
            "saturation_spec": constants.HILL,
        },
        "sampling": {
            "n_chains": args.chains,
            "n_adapt": args.adapt,
            "n_burnin": args.burnin,
            "n_keep": args.keep,
            "prior_draws": args.prior_draws,
            "seed": args.seed,
            "max_tree_depth": args.max_tree_depth,
            "confidence_level": args.confidence_level,
            "elapsed_seconds": round(elapsed_seconds, 2),
        },
        "diagnostics": diagnostics,
        "posterior_channel_summary": channel_rows.to_dict(orient="records"),
        "comparison": {
            "ranking": ranking,
            "rows": comparison.to_dict(orient="records"),
        },
    }
    with open(OUT_DIR / "meridian_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main() -> None:
    args = parse_args()
    configure_runtime()
    raw_df = load_raw_data()
    meridian_df = prepare_meridian_input(raw_df)
    input_data = build_input_data(meridian_df)
    mmm = build_model(input_data)

    start = time.perf_counter()
    mmm.sample_prior(n_draws=args.prior_draws, seed=args.seed)
    mmm.sample_posterior(
        n_chains=args.chains,
        n_adapt=args.adapt,
        n_burnin=args.burnin,
        n_keep=args.keep,
        seed=args.seed + 1,
        max_tree_depth=args.max_tree_depth,
    )
    elapsed_seconds = time.perf_counter() - start

    summary_ds = visualizer.MediaSummary(
        mmm, confidence_level=args.confidence_level
    ).get_paid_summary_metrics()
    meridian_summary = export_channel_summary(summary_ds, raw_df, args.confidence_level)
    curves = export_response_curves(mmm, args.confidence_level, len(raw_df))
    comparison, ranking = compare_with_ridge(meridian_summary)
    diagnostics = compute_diagnostics(mmm)

    plot_intervals(
        meridian_summary,
        "incremental_outcome",
        "Incremental sales",
        "Meridian posterior channel contribution",
        "meridian_contribution_intervals.png",
    )
    plot_intervals(
        meridian_summary,
        "roi",
        "ROI ($ sales / $ spend)",
        "Meridian posterior ROI",
        "meridian_roi_intervals.png",
    )
    plot_response_curves(curves)
    plot_comparison(comparison)
    export_model_fit(mmm, args.confidence_level)
    save_results(
        args=args,
        raw_df=raw_df,
        elapsed_seconds=elapsed_seconds,
        diagnostics=diagnostics,
        meridian_summary=meridian_summary,
        comparison=comparison,
        ranking=ranking,
    )

    print(json.dumps(json.loads((OUT_DIR / "meridian_results.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
