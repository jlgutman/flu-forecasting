"""
plot_results.py
Visualise flu forecast model results for research paper publication.

Reads:
  forecast_eval.csv          — evaluation period: actual vs predicted
  forecast_next4weeks.csv    — future 4-week forecast (actual = null)
  xgb_feature_importance.csv — XGBoost feature importances (optional)

Output PNGs (research-paper quality, 150–200 dpi):
  forecast_plot_{STATE}.png         — per-state 3-panel detail
  forecast_summary_grid.png         — multi-state actual vs ensemble
  forecast_future_bars.png          — 4-week ahead grouped bars
  forecast_scatter_comparison.png   — actual vs predicted scatter (all models)
  forecast_stl_decomposition.png    — STL seasonal decomposition
  forecast_acf_pacf.png             — ACF / PACF diagnostics
  forecast_ccf.png                  — cross-correlation: flu vs env drivers
  forecast_xgb_importance.png       — XGBoost feature importance
  forecast_model_comparison.png     — metric bar chart across models
  forecast_error_heatmap.png        — MAPE by CDC week-of-year × model
  forecast_horizon_degradation.png  — accuracy vs forecast lead time (+1..+4 wk)
"""

import sys
from pathlib import Path

# Allow running without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from src.config import DATA_DIR, OUTPUT_DIR

HERE       = OUTPUT_DIR                                   # PNGs saved alongside CSVs
EVAL_CSV   = OUTPUT_DIR / "forecast_eval.csv"
FUTURE_CSV = OUTPUT_DIR / "forecast_next4weeks.csv"
IMP_CSV    = OUTPUT_DIR / "xgb_feature_importance.csv"
ENV_CSV    = DATA_DIR   / "flu_environmental_factors_data.csv"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MODEL_STYLE = {
    "actual":        dict(color="#1f1f1f", lw=2.2,  ls="-",  zorder=5, label="Actual"),
    "arimax_pred":   dict(color="#e07b39", lw=1.6,  ls="--", zorder=4, label="ARIMAX"),
    "hw_pred":       dict(color="#3a86ff", lw=1.6,  ls="-.", zorder=4, label="Holt-Winters"),
    "xgb_pred":      dict(color="#ff006e", lw=1.6,  ls=":",  zorder=4, label="XGBoost"),
    "lstm_pred":     dict(color="#8338ec", lw=1.6,  ls=":",  zorder=4, label="LSTM"),
    "ensemble_pred": dict(color="#2dc653", lw=2.0,  ls="-",  zorder=4, label="Ensemble"),
}
PRED_COLS   = ["arimax_pred", "hw_pred", "xgb_pred", "lstm_pred", "ensemble_pred"]
PRED_LABELS = ["ARIMAX", "Holt-Winters", "XGBoost", "LSTM", "Ensemble"]

FUTURE_COLOR = "#ffc300"
FUTURE_ALPHA = 0.12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data():
    eval_df   = pd.read_csv(EVAL_CSV,   parse_dates=["week_start"])
    future_df = pd.read_csv(FUTURE_CSV, parse_dates=["week_start"])
    return eval_df, future_df


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("week_start").reset_index(drop=True)


def _metrics_text(grp: pd.DataFrame) -> str:
    lines, actual = [], grp["actual"].values
    mask_base = ~np.isnan(actual)
    for col, lbl in zip(PRED_COLS, PRED_LABELS):
        if col not in grp.columns:
            continue
        pred = grp[col].values
        mask = mask_base & ~np.isnan(pred)
        if mask.sum() < 2:
            continue
        a, p  = actual[mask], pred[mask]
        rmse  = np.sqrt(np.mean((a - p) ** 2))
        nz    = a >= 10
        mape  = np.mean(np.abs((a[nz] - p[nz]) / a[nz])) * 100 if nz.any() else np.nan
        ss_res = np.sum((a - p) ** 2)
        ss_tot = np.sum((a - np.mean(a)) ** 2)
        r2    = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        lines.append(f"{lbl:<13} RMSE={rmse:>7,.0f}  MAPE={mape:>5.1f}%  R²={r2:>5.2f}")
    return "\n".join(lines)


def _add_future_band(ax, future_dates):
    if not len(future_dates):
        return
    x0 = future_dates.min() - pd.Timedelta(days=3)
    x1 = future_dates.max() + pd.Timedelta(days=3)
    ax.axvspan(x0, x1, color=FUTURE_COLOR, alpha=FUTURE_ALPHA, zorder=1)
    ax.axvline(x=x0, color=FUTURE_COLOR, lw=1.2, ls="--", alpha=0.7, zorder=2)


def _fmt_y(ax):
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))


def _compute_r2(a, p):
    mask = ~np.isnan(a) & ~np.isnan(p)
    if mask.sum() < 2:
        return np.nan
    a, p = a[mask], p[mask]
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


# ===========================================================================
# 1. Per-state detail plot (3 panels)
# ===========================================================================

def plot_state(state: str, eval_df: pd.DataFrame, future_df: pd.DataFrame) -> Path:
    ev = _sort(eval_df[eval_df["state"] == state].copy())
    fu = _sort(future_df[future_df["state"] == state].copy())
    if ev.empty:
        return None

    fig = plt.figure(figsize=(17, 14))
    fig.suptitle(f"Flu Forecast — {state}", fontsize=15, fontweight="bold", y=0.98)
    gs = fig.add_gridspec(3, 1, hspace=0.45, height_ratios=[3, 1.5, 2])
    ax_full  = fig.add_subplot(gs[0])
    ax_resid = fig.add_subplot(gs[1], sharex=ax_full)
    ax_zoom  = fig.add_subplot(gs[2])

    # Panel 1 — full evaluation period
    ax_full.plot(ev["week_start"], ev["actual"], **MODEL_STYLE["actual"])
    for col in PRED_COLS:
        if col in ev.columns:
            ax_full.plot(ev["week_start"], ev[col], **MODEL_STYLE[col])

    if not fu.empty:
        _add_future_band(ax_full, fu["week_start"])
        fu_stack = fu[[c for c in PRED_COLS if c in fu.columns]].values
        if fu_stack.shape[1]:
            lo = np.nanmin(fu_stack, axis=1)
            hi = np.nanmax(fu_stack, axis=1)
            ax_full.fill_between(fu["week_start"], lo, hi,
                                 color=FUTURE_COLOR, alpha=0.25, zorder=2,
                                 label="Model range (future)")
        for col in PRED_COLS:
            if col in fu.columns:
                s = MODEL_STYLE[col].copy()
                s.update(lw=2.2, label=None)
                ax_full.plot(fu["week_start"], fu[col], **s)

    ax_full.set_ylabel("ILI Cases", fontsize=10)
    ax_full.set_title("Evaluation Period  |  yellow band = 4-week future forecast", fontsize=10)
    _fmt_y(ax_full)
    ax_full.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.8)
    ax_full.text(
        0.99, 0.97, _metrics_text(ev),
        transform=ax_full.transAxes, fontsize=7, va="top", ha="right",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )

    # Panel 2 — residuals
    for col in PRED_COLS:
        if col in ev.columns:
            resid = ev["actual"] - ev[col]
            ax_resid.plot(ev["week_start"], resid,
                          color=MODEL_STYLE[col]["color"], lw=1.1, alpha=0.8,
                          label=MODEL_STYLE[col]["label"])
    ax_resid.axhline(0, color="black", lw=0.8, ls="--")
    ax_resid.set_ylabel("Residual\n(Actual − Pred)", fontsize=9)
    ax_resid.set_title("Residuals by Model", fontsize=9)
    _fmt_y(ax_resid)
    plt.setp(ax_resid.get_xticklabels(), visible=False)

    # Panel 3 — zoomed last 16 + future 4
    ev_zoom = ev.tail(min(16, len(ev)))
    ax_zoom.plot(ev_zoom["week_start"], ev_zoom["actual"], **MODEL_STYLE["actual"])
    for col in PRED_COLS:
        if col in ev_zoom.columns:
            ax_zoom.plot(ev_zoom["week_start"], ev_zoom[col], **MODEL_STYLE[col])
    if not fu.empty:
        _add_future_band(ax_zoom, fu["week_start"])
        for col in PRED_COLS:
            if col in fu.columns:
                s = MODEL_STYLE[col].copy()
                s.update(lw=2.5, marker="o", markersize=5, label=None)
                ax_zoom.plot(fu["week_start"], fu[col], **s)

    ax_zoom.set_xlabel("Week starting", fontsize=10)
    ax_zoom.set_ylabel("ILI Cases", fontsize=10)
    ax_zoom.set_title("Last 16 Eval Weeks + Next 4 Weeks (zoomed)", fontsize=10)
    _fmt_y(ax_zoom)
    ax_zoom.tick_params(axis="x", rotation=30)

    fig.autofmt_xdate(rotation=30)
    plt.setp(ax_full.get_xticklabels(), visible=False)

    # Secondary top x-axis: MMWR week numbers (set up AFTER autofmt_xdate so it isn't reset)
    ax_top = ax_full.twiny()
    x0 = mdates.date2num(ev["week_start"].iloc[0])
    x1 = mdates.date2num(ev["week_start"].iloc[-1])
    ax_top.set_xlim(x0, x1)
    tick_step = max(13, len(ev) // 10)
    tick_rows = ev.iloc[::tick_step]
    ax_top.set_xticks(mdates.date2num(tick_rows["week_start"]))
    ax_top.set_xticklabels(
        [f"W{int(r['week'])}\n{int(r['year'])}" for _, r in tick_rows.iterrows()],
        fontsize=7, ha="center", rotation=0,
    )
    ax_top.set_xlabel("MMWR Week / Year", fontsize=8)

    out = HERE / f"forecast_plot_{state}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")
    return out


# ===========================================================================
# 2. Multi-state summary grid
# ===========================================================================

def plot_summary_grid(eval_df: pd.DataFrame, future_df: pd.DataFrame) -> Path:
    states = sorted(eval_df["state"].unique())
    n      = len(states)
    if n < 2:
        return None

    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 3.2),
                             squeeze=False)
    fig.suptitle("Flu Forecast — All States: Actual vs Ensemble",
                 fontsize=14, fontweight="bold", y=1.01)

    for idx, state in enumerate(states):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ev = _sort(eval_df[eval_df["state"] == state])
        fu = _sort(future_df[future_df["state"] == state])

        ax.plot(ev["week_start"], ev["actual"],
                color=MODEL_STYLE["actual"]["color"], lw=1.5)
        if "ensemble_pred" in ev.columns:
            ax.plot(ev["week_start"], ev["ensemble_pred"],
                    color=MODEL_STYLE["ensemble_pred"]["color"], lw=1.4)
        if not fu.empty and "ensemble_pred" in fu.columns:
            _add_future_band(ax, fu["week_start"])
            ax.plot(fu["week_start"], fu["ensemble_pred"],
                    color=MODEL_STYLE["ensemble_pred"]["color"],
                    lw=1.8, marker="o", markersize=3)

        if "ensemble_pred" in ev.columns:
            r2 = _compute_r2(ev["actual"].values, ev["ensemble_pred"].values)
            ax.text(0.03, 0.95, f"R²={r2:.2f}", transform=ax.transAxes,
                    fontsize=7.5, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              alpha=0.7, edgecolor="none"))

        ax.set_title(state, fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.tick_params(axis="y", labelsize=7)
        _fmt_y(ax)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    out = HERE / "forecast_summary_grid.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")
    return out


# ===========================================================================
# 3. Future 4-week bar chart
# ===========================================================================

def plot_future_bars(future_df: pd.DataFrame) -> Path:
    states = sorted(future_df["state"].unique())
    n_states = len(states)
    if not n_states:
        return None

    bar_cols   = [c for c in PRED_COLS if c != "ensemble_pred"]
    bar_labels = [MODEL_STYLE[c]["label"] for c in bar_cols]
    bar_colors = [MODEL_STYLE[c]["color"] for c in bar_cols]

    ncols = min(4, n_states)
    nrows = (n_states + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, nrows * 4),
                             squeeze=False)
    fig.suptitle("4-Week Ahead Flu Forecast by Model",
                 fontsize=13, fontweight="bold", y=1.01)

    for idx, state in enumerate(states):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        fu = _sort(future_df[future_df["state"] == state])
        week_labels = fu["week_start"].dt.strftime("W%-m/%-d").tolist()
        n_weeks     = len(fu)
        x           = np.arange(n_weeks)
        width       = 0.18
        offsets     = np.linspace(-(len(bar_cols) - 1) / 2,
                                   (len(bar_cols) - 1) / 2,
                                   len(bar_cols)) * width

        for i, (cname, lbl, clr) in enumerate(zip(bar_cols, bar_labels, bar_colors)):
            if cname not in fu.columns:
                continue
            vals = fu[cname].values
            bars = ax.bar(x + offsets[i], vals, width=width,
                          label=lbl, color=clr, alpha=0.85, edgecolor="white")
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + ax.get_ylim()[1] * 0.01,
                            f"{v:,.0f}", ha="center", va="bottom",
                            fontsize=6, rotation=45)

        ax.set_title(state, fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(week_labels, fontsize=7.5)
        ax.set_ylabel("Predicted ILI Cases", fontsize=8)
        _fmt_y(ax)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
        if idx == 0:
            ax.legend(fontsize=7, ncol=2, loc="upper right")

    for idx in range(n_states, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    out = HERE / "forecast_future_bars.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")
    return out


# ===========================================================================
# 4. Actual vs Predicted scatter (all models)
# ===========================================================================

def plot_scatter_comparison(eval_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Actual vs Predicted — All States", fontsize=13, fontweight="bold")

    for ax, col, lbl in zip(axes.flat, PRED_COLS, PRED_LABELS):
        mask = eval_df["actual"].notna() & eval_df[col].notna()
        a    = eval_df.loc[mask, "actual"].values
        p    = eval_df.loc[mask, col].values
        if not len(a):
            ax.set_visible(False)
            continue

        if len(a) > 500:
            hb = ax.hexbin(a, p, gridsize=40, cmap="YlOrRd",
                           mincnt=1, bins="log", linewidths=0.2)
            fig.colorbar(hb, ax=ax, label="log₁₀(count)")
        else:
            ax.scatter(a, p, color=MODEL_STYLE[col]["color"],
                       alpha=0.5, s=18, edgecolors="none")

        lim = max(a.max(), p.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)

        r2   = _compute_r2(a, p)
        rmse = np.sqrt(np.mean((a - p) ** 2))
        nz   = a >= 10
        mape = np.mean(np.abs((a[nz] - p[nz]) / a[nz])) * 100 if nz.any() else np.nan

        ax.set_title(f"{lbl}\nR²={r2:.3f}  RMSE={rmse:,.0f}  MAPE={mape:.1f}%", fontsize=9)
        ax.set_xlabel("Actual ILI Cases", fontsize=9)
        ax.set_ylabel("Predicted ILI Cases", fontsize=9)
        _fmt_y(ax)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Hide unused subplot (2x3 grid, 5 models)
    axes.flat[-1].set_visible(False)

    fig.tight_layout()
    out = HERE / "forecast_scatter_comparison.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")
    return out


# ===========================================================================
# 5. STL seasonal decomposition (Zheng et al. 2024 Fig. 4 equivalent)
# ===========================================================================

def plot_stl_decomposition(eval_df: pd.DataFrame) -> Path:
    """
    STL decomposition of observed ILI for each state.
    Reveals trend, annual seasonal cycle, and remainder components —
    standard diagnostic in the ARIMAX and forecasting_models papers.
    """
    from statsmodels.tsa.seasonal import STL

    states = sorted(eval_df["state"].unique())
    fig, axes = plt.subplots(len(states), 4,
                             figsize=(18, max(4, len(states) * 3.5)),
                             squeeze=False)
    fig.suptitle("STL Decomposition of Weekly ILI Cases",
                 fontsize=14, fontweight="bold")

    for row_idx, state in enumerate(states):
        ev = _sort(eval_df[eval_df["state"] == state])
        y  = ev["actual"].values
        t  = ev["week_start"]

        axes[row_idx][0].set_ylabel(state, fontsize=11, fontweight="bold", rotation=0,
                                    labelpad=40, va="center")

        try:
            result = STL(y, period=52, robust=True).fit()
            components = [y, result.trend, result.seasonal, result.resid]
            titles     = ["Observed", "Trend", "Seasonal", "Residual"]
        except Exception:
            components = [y, np.full_like(y, np.nan)] * 2
            titles     = ["Observed", "Trend", "Seasonal", "Residual"]

        for ci, (comp, title) in enumerate(zip(components, titles)):
            ax = axes[row_idx][ci]
            ax.plot(t, comp, color="#2c7bb6", lw=1.2)
            if ci == 0:
                ax.fill_between(t, 0, comp, alpha=0.15, color="#2c7bb6")
            ax.set_title(title if row_idx == 0 else "", fontsize=10)
            ax.tick_params(axis="x", rotation=30, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            _fmt_y(ax)

    fig.tight_layout()
    out = HERE / "forecast_stl_decomposition.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")
    return out


# ===========================================================================
# 6. ACF / PACF diagnostics (Zheng et al. 2024 Fig. 5 equivalent)
# ===========================================================================

def plot_acf_pacf(eval_df: pd.DataFrame) -> Path:
    """
    ACF and PACF plots for each state — used in the ARIMAX paper to
    determine AR and MA orders. Shown for the evaluation-period residuals
    to reveal remaining autocorrelation structure.
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    states = sorted(eval_df["state"].unique())
    n = len(states)
    fig, axes = plt.subplots(n, 4, figsize=(18, max(4, n * 3)),
                             squeeze=False)
    fig.suptitle("ACF / PACF — Observed ILI & Ensemble Residuals",
                 fontsize=13, fontweight="bold")

    for row_idx, state in enumerate(states):
        ev = _sort(eval_df[eval_df["state"] == state])
        y  = np.log1p(ev["actual"].fillna(0).values)
        resid = (ev["actual"] - ev.get("ensemble_pred", ev["actual"])).fillna(0).values

        titles = [f"ACF — log(ILI) {state}", f"PACF — log(ILI)",
                  "ACF — Ensemble Residuals", "PACF — Ensemble Residuals"]
        series = [y, y, resid, resid]
        funcs  = [plot_acf, plot_pacf, plot_acf, plot_pacf]

        for ci, (ser, func, title) in enumerate(zip(series, funcs, titles)):
            ax = axes[row_idx][ci]
            try:
                func(ser, lags=40, ax=ax, alpha=0.05, zero=False, color="#2c7bb6")
            except Exception:
                pass
            ax.set_title(title if row_idx == 0 else "", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("Lag (weeks)" if row_idx == n - 1 else "")

    fig.tight_layout()
    out = HERE / "forecast_acf_pacf.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")
    return out


# ===========================================================================
# 7. Cross-correlation (CCF): flu vs environmental drivers
#    (Zheng et al. 2024 Table 2 — Fig. equivalent)
# ===========================================================================

def plot_ccf(eval_df: pd.DataFrame) -> Path:
    """
    Cross-correlation function (CCF) between weekly ILI and key environmental
    variables at lags 0–12. Replicates the analysis in Zheng et al. (2024)
    which showed temperature and humidity have significant lagged correlations
    with flu incidence (max at lag 3–5 weeks).
    """
    from statsmodels.tsa.stattools import ccf as sm_ccf

    # Load environmental data to join with eval period
    try:
        env = pd.read_csv(ENV_CSV, encoding="utf-8-sig")
        env = env.rename(columns={"STATE": "state", "DATE": "year_month"})
        env["year_month"] = env["year_month"].astype(str)
    except Exception:
        print("  [CCF] Environmental CSV not found — skipping CCF plot.")
        return None

    env_vars = {
        "TAVG":  "Avg Temperature (°C)",
        "RHAV":  "Avg Relative Humidity (%)",
        "TRR":   "Temperature Range Ratio",
        "RHRR":  "Humidity Range Ratio",
    }

    states = sorted(eval_df["state"].unique())
    n_vars = len(env_vars)
    fig, axes = plt.subplots(len(states), n_vars,
                             figsize=(n_vars * 5, max(4, len(states) * 3.5)),
                             squeeze=False)
    fig.suptitle(
        "Cross-Correlation: Weekly ILI vs Environmental Drivers\n"
        "(positive lag = env leads ILI — Zheng et al. 2024 methodology)",
        fontsize=12, fontweight="bold",
    )

    for row_idx, state in enumerate(states):
        ev = _sort(eval_df[eval_df["state"] == state].copy())
        ev["year_month"] = ev["week_start"].dt.to_period("M").astype(str)
        ev_env = ev.merge(env[env["state"] == state], on="year_month", how="left")
        flu = ev_env["actual"].ffill().values

        for ci, (var, label) in enumerate(env_vars.items()):
            ax = axes[row_idx][ci]
            if var in ev_env.columns:
                x = ev_env[var].ffill().values
                # Compute CCF at lags 0..12
                max_lags = 12
                try:
                    cc = sm_ccf(flu, x, nlags=max_lags, fft=True)
                    lags = np.arange(0, max_lags + 1)
                    colors = ["#e63946" if abs(v) > 0.2 else "#457b9d" for v in cc]
                    ax.bar(lags, cc, color=colors, edgecolor="white", linewidth=0.5)
                    ci95 = 1.96 / np.sqrt(len(flu))
                    ax.axhline(ci95, color="gray", ls="--", lw=0.8)
                    ax.axhline(-ci95, color="gray", ls="--", lw=0.8)
                    # Mark max correlation lag
                    peak_lag = int(np.argmax(np.abs(cc)))
                    ax.axvline(peak_lag, color="red", lw=1, alpha=0.6)
                    ax.text(peak_lag + 0.2, 0.9 * ax.get_ylim()[1],
                            f"lag={peak_lag}", fontsize=7, color="red")
                except Exception:
                    ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                            ha="center", va="center")

            ax.set_title(f"{label}\n({state})" if row_idx == 0 else f"({state})",
                         fontsize=8)
            ax.set_xlabel("Lag (weeks)" if row_idx == len(states) - 1 else "")
            ax.set_ylabel("CCF" if ci == 0 else "")
            ax.tick_params(labelsize=7)
            ax.set_xlim(-0.5, max_lags + 0.5)

    fig.tight_layout()
    out = HERE / "forecast_ccf.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")
    return out


# ===========================================================================
# 8. XGBoost feature importance (innovation — not in any reference paper)
# ===========================================================================

def plot_xgb_importance() -> Path:
    if not IMP_CSV.exists():
        print("  [XGB importance] xgb_feature_importance.csv not found — skipping.")
        return None

    imp_df = pd.read_csv(IMP_CSV, index_col=0)
    mean_imp = imp_df.mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = sns.color_palette("RdYlGn", n_colors=len(mean_imp))
    bars   = ax.barh(mean_imp.index, mean_imp.values, color=colors, edgecolor="white")
    ax.set_xlabel("Mean Gain (XGBoost, averaged over HORIZON outputs)", fontsize=10)
    ax.set_title("XGBoost Feature Importance\n(averaged across states and forecast horizons)",
                 fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)

    # Label bars with values
    for bar in bars:
        w = bar.get_width()
        ax.text(w + mean_imp.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", fontsize=8)

    fig.tight_layout()
    out = HERE / "forecast_xgb_importance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")
    return out


# ===========================================================================
# 9. Model comparison bar chart (research paper Table equivalent)
# ===========================================================================

def plot_model_comparison(eval_df: pd.DataFrame) -> Path:
    """
    Side-by-side bar chart of RMSE, MAE, MAPE, and R² across models.
    Direct equivalent to Chen et al. (2024) Table 2.
    """
    metrics_map = {}
    actual = eval_df["actual"].values.astype(float)

    for col, lbl in zip(PRED_COLS, PRED_LABELS):
        if col not in eval_df.columns:
            continue
        pred = eval_df[col].values.astype(float)
        mask = ~np.isnan(actual) & ~np.isnan(pred)
        if mask.sum() < 2:
            continue
        a, p = actual[mask], pred[mask]
        rmse = np.sqrt(np.mean((a - p) ** 2))
        mae  = np.mean(np.abs(a - p))
        nz   = a >= 10
        mape = np.mean(np.abs((a[nz] - p[nz]) / a[nz])) * 100 if nz.any() else np.nan
        r2   = _compute_r2(a, p)
        metrics_map[lbl] = {"RMSE": rmse, "MAE": mae, "MAPE (%)": mape, "R²": r2}

    if not metrics_map:
        return None

    df = pd.DataFrame(metrics_map).T
    metric_cols = ["RMSE", "MAE", "MAPE (%)", "R²"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Model Performance Comparison (Evaluation Period, All States)",
                 fontsize=13, fontweight="bold")

    palette = [MODEL_STYLE[c]["color"] for c in PRED_COLS
               if MODEL_STYLE[c]["label"] in df.index]

    for ax, metric in zip(axes, metric_cols):
        vals   = df[metric].values
        labels = df.index.tolist()
        colors = palette[:len(labels)]
        bars   = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.7)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30, labelsize=9)

        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        # Highlight best model
        if metric == "R²":
            best = np.nanargmax(vals)
        else:
            best = np.nanargmin(vals)
        bars[best].set_edgecolor("black")
        bars[best].set_linewidth(2.5)

    fig.tight_layout()
    out = HERE / "forecast_model_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")
    return out


# ===========================================================================
# 10. Error heatmap: MAPE by CDC week-of-year × model
#     (novel — not in any reference paper)
# ===========================================================================

def plot_error_heatmap(eval_df: pd.DataFrame) -> Path:
    """
    Heatmap of median absolute percent error grouped by CDC epidemiological
    week (1–52). Reveals which weeks/seasons are hardest to forecast.
    """
    df = eval_df.copy()
    df = df[df["actual"] >= 10].copy()   # exclude suppressed weeks

    heatmap_data = {}
    for col, lbl in zip(PRED_COLS, PRED_LABELS):
        if col not in df.columns:
            continue
        sub = df[["week", "actual", col]].dropna()
        sub["ape"] = np.abs((sub["actual"] - sub[col]) / sub["actual"]) * 100
        grouped = sub.groupby("week")["ape"].median()
        heatmap_data[lbl] = grouped

    if not heatmap_data:
        return None

    heat_df = pd.DataFrame(heatmap_data).T   # models × weeks
    heat_df.columns = heat_df.columns.astype(int)
    heat_df = heat_df.reindex(sorted(heat_df.columns), axis=1)

    fig, ax = plt.subplots(figsize=(20, 4))
    sns.heatmap(
        heat_df,
        ax=ax,
        cmap="RdYlGn_r",
        annot=False,
        linewidths=0.3,
        cbar_kws={"label": "Median APE (%)"},
        vmin=0, vmax=heat_df.values.max() * 0.8,
    )
    ax.set_xlabel("CDC Epidemiological Week", fontsize=10)
    ax.set_ylabel("Model", fontsize=10)
    ax.set_title("Forecast Error by Week-of-Year × Model\n"
                 "(red = high error, green = low error; weeks 45–10 = peak flu season)",
                 fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    # Shade peak flu season (weeks 45–52 and 1–10)
    for w in list(range(1, 11)) + list(range(45, 53)):
        if w in heat_df.columns:
            xi = heat_df.columns.get_loc(w)
            ax.axvspan(xi, xi + 1, color="#e63946", alpha=0.06)

    fig.tight_layout()
    out = HERE / "forecast_error_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")
    return out


# ===========================================================================
# 11. Forecast horizon degradation: accuracy vs lead time (+1..+4 weeks)
#     (Amendolara et al. 2023 Table 1 — Fig. equivalent)
# ===========================================================================

def plot_horizon_degradation(eval_df: pd.DataFrame) -> Path:
    """
    Shows how prediction accuracy (MAE and MAPE) changes across the 4-week
    horizon. In Amendolara et al. (2023) accuracy degrades from +1 to +10 wks;
    we show +1 to +4 for each model, replicating that key finding.

    Each 'step offset' is estimated by grouping consecutive HORIZON-sized chunks
    and taking the k-th element within each chunk as step k predictions.
    """
    df = eval_df.copy().sort_values(["state", "week_start"]).reset_index(drop=True)
    HORIZON = 4

    results = {lbl: {"mae": [], "mape": []} for lbl in PRED_LABELS}

    for state, sdf in df.groupby("state"):
        sdf = sdf.reset_index(drop=True)
        n   = len(sdf)
        for col, lbl in zip(PRED_COLS, PRED_LABELS):
            if col not in sdf.columns:
                continue
            step_mae  = {k: [] for k in range(1, HORIZON + 1)}
            step_mape = {k: [] for k in range(1, HORIZON + 1)}
            # Group into HORIZON-sized windows and extract per-step errors
            for i in range(0, n - HORIZON, HORIZON):
                chunk = sdf.iloc[i : i + HORIZON]
                for k, (_, row) in enumerate(chunk.iterrows(), start=1):
                    a = row["actual"]
                    p = row[col]
                    if np.isnan(a) or np.isnan(p) or a < 10:
                        continue
                    step_mae[k].append(abs(a - p))
                    step_mape[k].append(abs(a - p) / a * 100)
            for k in range(1, HORIZON + 1):
                results[lbl]["mae"].append(
                    np.mean(step_mae[k]) if step_mae[k] else np.nan
                )
                results[lbl]["mape"].append(
                    np.mean(step_mape[k]) if step_mape[k] else np.nan
                )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Forecast Accuracy vs Lead Time (+1 to +4 Weeks)\n"
                 "(Amendolara et al. 2023 methodology)",
                 fontsize=12, fontweight="bold")

    x = np.arange(1, HORIZON + 1)
    for lbl, col in zip(PRED_LABELS, PRED_COLS):
        mae_arr  = results[lbl]["mae"]
        mape_arr = results[lbl]["mape"]
        n_chunks = len(mae_arr) // HORIZON
        if n_chunks == 0:
            continue
        # Average across states per step
        mae_per_step  = [np.nanmean(mae_arr[k::HORIZON]) for k in range(HORIZON)]
        mape_per_step = [np.nanmean(mape_arr[k::HORIZON]) for k in range(HORIZON)]

        kw = dict(color=MODEL_STYLE[col]["color"],
                  ls=MODEL_STYLE[col]["ls"],
                  lw=2, marker="o", markersize=6, label=lbl)
        axes[0].plot(x, mae_per_step,  **kw)
        axes[1].plot(x, mape_per_step, **kw)

    for ax, metric in zip(axes, ["MAE (ILI cases)", "MAPE (%)"]):
        ax.set_xlabel("Forecast Lead Time (weeks)", fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"+{k} wk" for k in x])
        ax.legend(fontsize=8, loc="upper left")
        ax.set_title(metric, fontsize=11)

    fig.tight_layout()
    out = HERE / "forecast_horizon_degradation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")
    return out


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    if not EVAL_CSV.exists():
        sys.exit(f"ERROR: {EVAL_CSV} not found. Run the forecast pipeline first.")
    if not FUTURE_CSV.exists():
        sys.exit(f"ERROR: {FUTURE_CSV} not found. Run the forecast pipeline first.")

    print("Loading CSVs…")
    eval_df, future_df = _load_data()

    states = sorted(eval_df["state"].unique())
    print(f"Found {len(states)} state(s): {', '.join(states)}\n")

    # Per-state detail plots
    print("1. Per-state detail plots…")
    for state in states:
        plot_state(state, eval_df, future_df)

    # Multi-state summary grid (only useful when multiple states present)
    if len(states) > 1:
        print("\n2. Summary grid…")
        plot_summary_grid(eval_df, future_df)

    print("\n3. Future forecast bars…")
    plot_future_bars(future_df)

    print("\n4. Actual vs predicted scatter…")
    plot_scatter_comparison(eval_df)

    print("\n5. STL decomposition…")
    plot_stl_decomposition(eval_df)

    print("\n6. ACF / PACF…")
    plot_acf_pacf(eval_df)

    print("\n7. Cross-correlation (CCF)…")
    plot_ccf(eval_df)

    print("\n8. XGBoost feature importance…")
    plot_xgb_importance()

    print("\n9. Model comparison bar chart…")
    plot_model_comparison(eval_df)

    print("\n10. Error heatmap by week-of-year…")
    plot_error_heatmap(eval_df)

    print("\n11. Forecast horizon degradation…")
    plot_horizon_degradation(eval_df)

    saved = list(HERE.glob("forecast_*.png"))
    print(f"\nDone — {len(saved)} plot(s) saved to {HERE}/")


if __name__ == "__main__":
    main()
