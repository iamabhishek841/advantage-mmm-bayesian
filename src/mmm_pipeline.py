"""
Marketing Mix Model (MMM) — end-to-end pipeline
=================================================
Data: weekly spend for TikTok, Facebook, Google Ads + weekly Sales (200 weeks).

Pipeline:
1. EDA — correlation, spend/sales trends
2. Feature engineering — adstock (carryover) + saturation (diminishing returns) transforms
3. Time-based train/test split (no shuffling — this is time series)
4. Ridge regression (regularized, handles multicollinearity between channels)
5. Evaluation — R^2, MAPE on held-out weeks
6. Business outputs — channel contribution %, ROI per channel, response/saturation
   curves, and a budget-reallocation recommendation
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHANNELS = ["TikTok", "Facebook", "Google Ads"]
DATA_PATH = "data/marketing_mix.csv"
OUT_DIR = "output"


# ---------------------------------------------------------------------------
# 1. Transforms
# ---------------------------------------------------------------------------
def adstock_transform(x, decay):
    """Geometric adstock: this week's effective spend = spend + decay * last week's
    effective spend. Models the idea that an ad seen this week still influences
    purchases in future weeks (carryover effect)."""
    result = np.zeros_like(x, dtype=float)
    result[0] = x[0]
    for t in range(1, len(x)):
        result[t] = x[t] + decay * result[t - 1]
    return result


def saturation_transform(x, alpha):
    """Hill-style saturation: models diminishing returns — doubling spend does not
    double the effect. alpha controls how quickly returns diminish."""
    x_scaled = x / (x.max() + 1e-9)
    return x_scaled ** alpha


def best_decay_and_alpha(x, y, decays, alphas):
    """Simple grid search per channel: pick the (decay, alpha) pair whose
    transformed feature correlates most strongly with sales. This is the
    standard lightweight alternative to a full Bayesian MMM (PyMC/Meridian)
    when time/compute is limited — correlation-based hyperparameter search
    on adstock+saturation curves."""
    best = (0.0, 1.0, -1)
    for d in decays:
        ads = adstock_transform(x, d)
        for a in alphas:
            sat = saturation_transform(ads, a)
            if sat.std() == 0:
                continue
            corr = abs(np.corrcoef(sat, y)[0, 1])
            if corr > best[2]:
                best = (d, a, corr)
    return best[0], best[1]


# ---------------------------------------------------------------------------
# 2. Load + EDA
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def run_eda(df):
    corr = df[CHANNELS + ["Sales"]].corr()
    corr.to_csv(f"{OUT_DIR}/correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    for ch in CHANNELS:
        ax.plot(df["Date"], df[ch], label=ch, alpha=0.7)
    ax.set_title("Weekly spend by channel")
    ax.set_ylabel("Spend ($)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/spend_over_time.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["Date"], df["Sales"], color="black")
    ax.set_title("Weekly sales")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/sales_over_time.png", dpi=120)
    plt.close(fig)

    return corr


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df):
    decays = [0.1, 0.3, 0.5, 0.7, 0.9]
    alphas = [0.3, 0.5, 0.7, 1.0]

    params = {}
    features = pd.DataFrame(index=df.index)
    for ch in CHANNELS:
        d, a = best_decay_and_alpha(df[ch].values, df["Sales"].values, decays, alphas)
        params[ch] = {"decay": d, "alpha": a}
        ads = adstock_transform(df[ch].values, d)
        sat = saturation_transform(ads, a)
        features[ch] = sat

    # seasonality controls
    features["week_of_year"] = df["Date"].dt.isocalendar().week.astype(float)
    features["month"] = df["Date"].dt.month.astype(float)
    features["trend"] = np.arange(len(df), dtype=float)

    return features, params


# ---------------------------------------------------------------------------
# 4. Modeling — time-based split, Ridge regression
# ---------------------------------------------------------------------------
def train_and_evaluate(features, target):
    n = len(features)
    split = int(n * 0.8)  # last 20% of weeks held out — time-based, not random

    X_train, X_test = features.iloc[:split], features.iloc[split:]
    y_train, y_test = target.iloc[:split], target.iloc[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)

    pred_test = model.predict(X_test_s)
    r2 = model.score(X_test_s, y_test)
    mape = np.mean(np.abs((y_test.values - pred_test) / y_test.values)) * 100

    return model, scaler, r2, mape, (X_test, y_test, pred_test)


# ---------------------------------------------------------------------------
# 5. Business outputs — contribution, ROI, response curves, budget recommendation
# ---------------------------------------------------------------------------
def channel_contributions(model, scaler, features, df):
    X_s = scaler.transform(features)
    coefs = dict(zip(features.columns, model.coef_))

    contributions = {}
    for ch in CHANNELS:
        idx = list(features.columns).index(ch)
        contributions[ch] = X_s[:, idx] * coefs[ch]

    contrib_df = pd.DataFrame(contributions, index=df.index)
    contrib_df[contrib_df < 0] = 0  # a channel's estimated contribution can't be negative for this business framing
    total_media_contribution = contrib_df.sum(axis=1).sum()
    total_sales = df["Sales"].sum()
    baseline_contribution = total_sales - total_media_contribution

    contribution_pct = (contrib_df.sum() / total_sales * 100).round(2)
    spend_total = df[CHANNELS].sum()
    roi = (contrib_df.sum() / spend_total).round(3)  # incremental sales $ per $ spent

    summary = pd.DataFrame({
        "total_spend": spend_total,
        "estimated_contribution_$": contrib_df.sum().round(2),
        "contribution_%_of_sales": contribution_pct,
        "estimated_ROI_($sales/$spend)": roi,
    })
    summary.loc["Baseline (non-media)"] = [
        np.nan, round(baseline_contribution, 2),
        round(baseline_contribution / total_sales * 100, 2), np.nan
    ]
    return summary, contrib_df


def response_curves(model, scaler, features, params, df):
    """For each channel, show predicted incremental sales as spend varies from
    0 to 2x its current average — this is the diminishing-returns curve used
    for budget decisions."""
    coefs = dict(zip(features.columns, model.coef_))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    curve_data = {}

    for i, ch in enumerate(CHANNELS):
        avg_spend = df[ch].mean()
        spend_range = np.linspace(0, avg_spend * 2.5, 50)
        d, a = params[ch]["decay"], params[ch]["alpha"]

        # approximate steady-state adstock for a constant spend level
        steady_state_adstock = spend_range / (1 - d) if d < 1 else spend_range
        sat = saturation_transform(steady_state_adstock, a) if steady_state_adstock.max() > 0 else steady_state_adstock

        idx = list(features.columns).index(ch)
        mean_, scale_ = scaler.mean_[idx], scaler.scale_[idx]
        sat_scaled = (sat - mean_) / scale_
        predicted_contribution = sat_scaled * coefs[ch]
        predicted_contribution = np.clip(predicted_contribution, 0, None)

        curve_data[ch] = {"spend": spend_range.tolist(), "contribution": predicted_contribution.tolist()}

        axes[i].plot(spend_range, predicted_contribution)
        axes[i].axvline(avg_spend, color="red", linestyle="--", label="current avg spend")
        axes[i].set_title(ch)
        axes[i].set_xlabel("Weekly spend ($)")
        axes[i].set_ylabel("Est. incremental sales ($)")
        axes[i].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/response_curves.png", dpi=120)
    plt.close(fig)
    return curve_data


def budget_recommendation(summary):
    roi = summary["estimated_ROI_($sales/$spend)"].drop("Baseline (non-media)", errors="ignore")
    best_channel = roi.idxmax()
    worst_channel = roi.idxmin()
    return {
        "highest_roi_channel": best_channel,
        "highest_roi_value": round(roi[best_channel], 3),
        "lowest_roi_channel": worst_channel,
        "lowest_roi_value": round(roi[worst_channel], 3),
        "recommendation": (
            f"Reallocate a portion of weekly budget from {worst_channel} "
            f"(ROI {round(roi[worst_channel], 2)}x) toward {best_channel} "
            f"(ROI {round(roi[best_channel], 2)}x), subject to the diminishing-returns "
            f"curve for {best_channel} — see response_curves.png for the point where "
            f"additional spend stops paying off."
        )
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    df = load_data()
    corr = run_eda(df)
    features, params = engineer_features(df)
    model, scaler, r2, mape, test_data = train_and_evaluate(features, df["Sales"])
    summary, contrib_df = channel_contributions(model, scaler, features, df)
    curve_data = response_curves(model, scaler, features, params, df)
    rec = budget_recommendation(summary)

    # prediction plot
    X_test, y_test, pred_test = test_data
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["Date"].iloc[X_test.index], y_test.values, label="Actual", color="black")
    ax.plot(df["Date"].iloc[X_test.index], pred_test, label="Predicted", color="orange", linestyle="--")
    ax.set_title(f"Held-out weeks: Actual vs Predicted (R2={r2:.3f}, MAPE={mape:.1f}%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/actual_vs_predicted.png", dpi=120)
    plt.close(fig)

    summary.to_csv(f"{OUT_DIR}/channel_contribution_summary.csv")
    corr.to_csv(f"{OUT_DIR}/correlation_matrix.csv")

    results = {
        "model": "Ridge regression on adstock+saturation transformed channel spend",
        "test_r2": round(r2, 4),
        "test_mape_pct": round(mape, 2),
        "n_weeks_total": len(df),
        "n_weeks_test": len(X_test),
        "adstock_saturation_params": params,
        "budget_recommendation": rec,
    }
    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{OUT_DIR}/response_curves.json", "w") as f:
        json.dump(curve_data, f, indent=2)

    print(json.dumps(results, indent=2))
    print("\nChannel contribution summary:")
    print(summary)


if __name__ == "__main__":
    main()
