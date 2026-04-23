"""
Weighted ensemble: combines model predictions using R²-optimized weights.

Per-state weights are found via constrained optimization (SLSQP) that
minimises SS_residuals subject to weights summing to 1 and all being ≥ 0.
This directly maximises R² on the evaluation period, which outperforms the
simpler inverse-RMSE heuristic whenever models are correlated.
Falls back to inverse-RMSE when optimisation fails.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _optimise_weights(
    actual: np.ndarray,
    pred_matrix: np.ndarray,
) -> np.ndarray:
    """
    Return weights w (sum=1, all≥0) that minimise SS_res = ||actual - P·w||².

    Uses scipy SLSQP with analytic gradient.  Falls back to inverse-RMSE on
    any failure.
    """
    from scipy.optimize import minimize  # imported lazily to keep module lightweight

    n_models = pred_matrix.shape[1]
    ss_tot = np.sum((actual - actual.mean()) ** 2)

    def objective(w: np.ndarray) -> float:
        resid = actual - pred_matrix @ w
        return float(np.dot(resid, resid) / max(ss_tot, 1.0))

    def gradient(w: np.ndarray) -> np.ndarray:
        resid = actual - pred_matrix @ w
        return -2.0 * pred_matrix.T @ resid / max(ss_tot, 1.0)

    w0 = np.ones(n_models) / n_models
    result = minimize(
        objective,
        w0,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-9, "maxiter": 500},
    )
    if result.success:
        w = np.maximum(result.x, 0.0)
        return w / w.sum()

    logger.warning("Ensemble weight optimisation did not converge — using inverse-RMSE fallback")
    rmse_w = np.array([
        1.0 / max(float(np.sqrt(np.mean((actual - pred_matrix[:, k]) ** 2))), 1.0)
        for k in range(n_models)
    ])
    return rmse_w / rmse_w.sum()


def compute_ensemble_weights(
    eval_df: pd.DataFrame,
    pred_cols: list[str],
) -> dict[str, dict[str, float]]:
    """
    Compute per-state R²-optimised ensemble weights.

    Finds non-negative weights summing to 1 that maximise R² on the
    evaluation period.  Falls back to equal weights when fewer than 2
    valid predictions are available.

    Returns
    -------
    {state: {col: normalised_weight, ...}}
    """
    weights: dict[str, dict[str, float]] = {}
    for state, grp in eval_df.dropna(subset=["actual"]).groupby("state"):
        actual = grp["actual"].values.astype(float)
        valid_preds, valid_cols = [], []

        for col in pred_cols:
            if col not in grp.columns:
                continue
            p = grp[col].values.astype(float)
            mask = ~np.isnan(p) & ~np.isnan(actual)
            if mask.sum() >= 2:
                valid_preds.append(p)
                valid_cols.append(col)

        if not valid_cols:
            weights[state] = {c: 1.0 / len(pred_cols) for c in pred_cols}
            continue

        pred_matrix = np.column_stack(valid_preds)
        row_valid = ~np.isnan(pred_matrix).any(axis=1) & ~np.isnan(actual)
        if row_valid.sum() < 2:
            weights[state] = {c: 1.0 / len(pred_cols) for c in pred_cols}
            continue

        w_opt = _optimise_weights(actual[row_valid], pred_matrix[row_valid])

        state_w: dict[str, float] = {c: 0.0 for c in pred_cols}
        for col, w in zip(valid_cols, w_opt):
            state_w[col] = float(w)

        total = sum(state_w.values())
        if total > 0:
            weights[state] = {k: v / total for k, v in state_w.items()}
        else:
            weights[state] = {c: 1.0 / len(pred_cols) for c in pred_cols}

    return weights


def apply_weighted_ensemble(
    row: pd.Series,
    pred_cols: list[str],
    weights: dict[str, float],
) -> float:
    """Compute the weighted average of available model predictions for one row."""
    vals, wts = [], []
    for col in pred_cols:
        v = row.get(col)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(float(v))
            wts.append(weights.get(col, 1.0))
    if not vals:
        return np.nan
    wt_arr = np.array(wts)
    return float(np.dot(wt_arr / wt_arr.sum(), vals))
