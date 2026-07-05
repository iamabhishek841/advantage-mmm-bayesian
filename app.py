from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
CHANNELS = ["TikTok", "Facebook", "Google Ads"]


st.set_page_config(page_title="AdVantage MMM", layout="wide")


def money(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"${value:,.0f}"


def ratio(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.2f}x"


@st.cache_data
def load_json(path: str) -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_ridge_summary() -> pd.DataFrame:
    df = pd.read_csv(OUT_DIR / "channel_contribution_summary.csv")
    df = df.rename(columns={df.columns[0]: "channel"})
    return df


@st.cache_data
def load_ridge_curves() -> pd.DataFrame:
    data = load_json("output/response_curves.json")
    rows = []
    for channel, values in data.items():
        for spend, contribution in zip(values["spend"], values["contribution"]):
            rows.append(
                {
                    "channel": channel,
                    "weekly_spend": spend,
                    "contribution": contribution,
                }
            )
    return pd.DataFrame(rows)


@st.cache_data
def load_meridian_summary() -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / "meridian_channel_contribution_summary.csv")


@st.cache_data
def load_meridian_curves() -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / "meridian_response_curves.csv")


@st.cache_data
def load_comparison() -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / "ridge_vs_meridian_comparison.csv")


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str):
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X(f"{x}:N", title=None, sort=None),
            y=alt.Y(f"{y}:Q", title=title),
            color=alt.Color(f"{x}:N", legend=None),
            tooltip=[x, alt.Tooltip(f"{y}:Q", format=",.0f")],
        )
        .properties(height=330)
    )


def interval_chart(df: pd.DataFrame, metric: str, title: str, fmt: str):
    plot_df = df[df["channel"].isin(CHANNELS)].copy()
    base = alt.Chart(plot_df).encode(
        x=alt.X("channel:N", title=None, sort=None),
        color=alt.Color("channel:N", legend=None),
    )
    bars = base.mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        y=alt.Y(f"{metric}_mean:Q", title=title),
        tooltip=[
            "channel",
            alt.Tooltip(f"{metric}_mean:Q", title="mean", format=fmt),
            alt.Tooltip(f"{metric}_ci_lo:Q", title="ci low", format=fmt),
            alt.Tooltip(f"{metric}_ci_hi:Q", title="ci high", format=fmt),
        ],
    )
    error = base.mark_errorbar(ticks=True).encode(
        y=alt.Y(f"{metric}_ci_lo:Q"),
        y2=alt.Y2(f"{metric}_ci_hi:Q"),
    )
    return (bars + error).properties(height=330)


def ridge_tab():
    results = load_json("output/results.json")
    summary = load_ridge_summary()
    channel_summary = summary[summary["channel"].isin(CHANNELS)].copy()
    roi_col = [col for col in channel_summary.columns if "ROI" in col][0]
    contribution_col = [col for col in channel_summary.columns if "estimated_contribution" in col][0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Test R2", f"{results['test_r2']:.3f}")
    c2.metric("Test MAPE", f"{results['test_mape_pct']:.1f}%")
    c3.metric("Best ROI", results["budget_recommendation"]["highest_roi_channel"])

    left, right = st.columns([1, 1])
    with left:
        st.altair_chart(
            bar_chart(channel_summary, "channel", contribution_col, "Estimated sales"),
            width="stretch",
        )
    with right:
        st.altair_chart(
            bar_chart(channel_summary, "channel", roi_col, "ROI"),
            width="stretch",
        )

    curves = load_ridge_curves()
    line = (
        alt.Chart(curves)
        .mark_line()
        .encode(
            x=alt.X("weekly_spend:Q", title="Weekly spend"),
            y=alt.Y("contribution:Q", title="Estimated incremental sales"),
            color="channel:N",
            tooltip=[
                "channel",
                alt.Tooltip("weekly_spend:Q", format=",.0f"),
                alt.Tooltip("contribution:Q", format=",.0f"),
            ],
        )
        .properties(height=360)
    )
    st.altair_chart(line, width="stretch")
    st.dataframe(channel_summary, width="stretch", hide_index=True)


def meridian_tab():
    results = load_json("output/meridian_results.json")
    summary = load_meridian_summary()
    channel_summary = summary[summary["channel"].isin(CHANNELS)].copy()
    diagnostics = results["diagnostics"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max R-hat", f"{diagnostics['rhat']['max']:.3f}")
    c2.metric("Divergences", diagnostics["divergences"])
    c3.metric("Posterior draws", results["sampling"]["n_chains"] * results["sampling"]["n_keep"])
    c4.metric("Best ROI", results["comparison"]["ranking"]["meridian_highest_roi_channel"])

    left, right = st.columns([1, 1])
    with left:
        st.altair_chart(
            interval_chart(channel_summary, "incremental_outcome", "Incremental sales", ",.0f"),
            width="stretch",
        )
    with right:
        st.altair_chart(
            interval_chart(channel_summary, "roi", "ROI", ".2f"),
            width="stretch",
        )

    curves = load_meridian_curves()
    base = alt.Chart(curves).encode(
        x=alt.X("weekly_spend:Q", title="Average weekly spend"),
        color="channel:N",
    )
    band = base.mark_area(opacity=0.18).encode(
        y=alt.Y("incremental_outcome_ci_lo:Q", title="Total incremental sales"),
        y2="incremental_outcome_ci_hi:Q",
    )
    line = base.mark_line().encode(
        y="incremental_outcome_mean:Q",
        tooltip=[
            "channel",
            alt.Tooltip("weekly_spend:Q", format=",.0f"),
            alt.Tooltip("incremental_outcome_mean:Q", format=",.0f"),
            alt.Tooltip("incremental_outcome_ci_lo:Q", format=",.0f"),
            alt.Tooltip("incremental_outcome_ci_hi:Q", format=",.0f"),
        ],
    )
    st.altair_chart((band + line).properties(height=360), width="stretch")
    st.dataframe(channel_summary, width="stretch", hide_index=True)


def comparison_tab():
    comparison = load_comparison()
    melted = comparison.melt(
        id_vars=["channel"],
        value_vars=["ridge_roi", "roi_mean"],
        var_name="model",
        value_name="roi",
    )
    melted["model"] = melted["model"].replace(
        {"ridge_roi": "Ridge regression", "roi_mean": "Bayesian Meridian"}
    )
    bars = (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X("channel:N", title=None, sort=None),
            y=alt.Y("roi:Q", title="ROI"),
            xOffset="model:N",
            color="model:N",
            tooltip=["channel", "model", alt.Tooltip("roi:Q", format=".2f")],
        )
        .properties(height=360)
    )
    st.altair_chart(bars, width="stretch")
    st.dataframe(comparison, width="stretch", hide_index=True)


st.title("AdVantage MMM")
ridge, meridian, comparison = st.tabs(
    ["Ridge regression", "Bayesian Meridian", "Comparison"]
)

with ridge:
    ridge_tab()

with meridian:
    meridian_tab()

with comparison:
    comparison_tab()
