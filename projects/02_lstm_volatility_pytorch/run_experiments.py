"""Run every (model x horizon x fold) experiment and persist predictions + metrics.

Outputs:
    data/predictions/{model}_{horizon}d_fold{n}.csv
        columns: date, y_true, y_pred, fold, model, horizon
    data/predictions/{model}_{horizon}d_all.csv
        concatenation of all folds for convenience
    artifacts/plots/shap_lgbm_{horizon}d.png
    artifacts/summary/metrics_summary.csv
        long-format: model, horizon, metric, value
    artifacts/summary/metrics_summary_wide.csv
        wide pivot: rows=model, columns=(horizon x metric)

Usage:
    python run_experiments.py
    python run_experiments.py --skip lstm,garch       # to skip slow models during dev
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from data.fetch import fetch_spy, save_spy
from features.engineer import HORIZONS, build_features
from training.cv import chronological_train_val_split, make_splits
from evaluation.metrics import all_metrics
from models.naive import predict_naive
from models.garch import predict_garch
from models.lgbm import DEFAULT_PARAMS as LGBM_DEFAULT_PARAMS
from models.lgbm import fit_lgbm, predict_lgbm, save_shap_summary
from models.lstm import fit_predict_lstm

SEED = 42
ALL_MODELS = ("naive", "garch", "lgbm", "lstm")
DEFAULT_TUNING_MODELS = ("lgbm", "lstm")

LGBM_TUNING_GRID = (
    {"num_leaves": 15, "min_child_samples": 20, "feature_fraction": 0.8, "lambda_l2": 0.0},
    {"num_leaves": 31, "min_child_samples": 20, "feature_fraction": 0.8, "lambda_l2": 1.0},
    {"num_leaves": 63, "min_child_samples": 30, "feature_fraction": 0.9, "lambda_l2": 1.0},
)

LSTM_TUNING_GRID = (
    {"hidden": 32, "dropout": 0.1, "lr": 1e-3, "batch_size": 32},
    {"hidden": 64, "dropout": 0.2, "lr": 1e-3, "batch_size": 64},
    {"hidden": 64, "dropout": 0.2, "lr": 5e-4, "batch_size": 64},
)


def _set_global_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _ensure_data(project_dir: Path) -> pd.DataFrame:
    spy_path = project_dir / "data" / "spy.csv"
    if not spy_path.exists():
        print(f"[data] {spy_path.name} missing, downloading...")
        df = fetch_spy()
        save_spy(df, spy_path)
        print(f"[data] saved {len(df):,} rows to {spy_path}")
    return pd.read_csv(spy_path, parse_dates=["Date"])


def _save_fold_predictions(
    out_dir: Path,
    model: str,
    horizon: int,
    fold: int,
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "date": dates.values,
            "y_true": y_true,
            "y_pred": y_pred,
            "fold": fold,
            "model": model,
            "horizon": horizon,
        }
    )
    out_path = out_dir / f"{model}_{horizon}d_fold{fold}.csv"
    df.to_csv(out_path, index=False)
    return df


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("inf")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def _tune_lgbm_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    inner_train_pos: np.ndarray,
    inner_val_pos: np.ndarray,
) -> tuple[dict, float]:
    X_tr = X_train.iloc[inner_train_pos]
    y_tr = y_train.iloc[inner_train_pos]
    X_va = X_train.iloc[inner_val_pos]
    y_va = y_train.iloc[inner_val_pos].to_numpy(dtype=float)

    best_cfg: dict | None = None
    best_rmse = float("inf")
    for cfg in LGBM_TUNING_GRID:
        params = dict(LGBM_DEFAULT_PARAMS)
        params.update(cfg)
        model = fit_lgbm(X_tr, y_tr, params=params)
        pred = predict_lgbm(model, X_va)
        score = _rmse(y_va, pred)
        if score < best_rmse:
            best_rmse = score
            best_cfg = cfg
    return (best_cfg or {}), best_rmse


def _tune_lstm_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    inner_train_pos: np.ndarray,
    inner_val_pos: np.ndarray,
    horizon: int,
) -> tuple[dict, float]:
    X_tune = X_train.iloc[inner_train_pos].reset_index(drop=True)
    y_tune = y_train.iloc[inner_train_pos].reset_index(drop=True)
    X_holdout = X_train.iloc[inner_val_pos].reset_index(drop=True)
    y_holdout = y_train.iloc[inner_val_pos].to_numpy(dtype=float)

    # Inner-inner split for early stopping while tuning.
    tune_inner_train, tune_inner_val = chronological_train_val_split(
        train_idx=np.arange(len(X_tune)),
        val_fraction=0.2,
        horizon=horizon,
    )
    if len(tune_inner_train) < 30 or len(tune_inner_val) < 5:
        return {}, float("inf")

    best_cfg: dict | None = None
    best_rmse = float("inf")
    for cfg in LSTM_TUNING_GRID:
        pos, pred = fit_predict_lstm(
            X_train=X_tune,
            y_train=y_tune,
            X_val=X_holdout,
            inner_train_idx_in_train=tune_inner_train,
            inner_val_idx_in_train=tune_inner_val,
            hidden=cfg["hidden"],
            dropout=cfg["dropout"],
            lr=cfg["lr"],
            batch_size=cfg["batch_size"],
            seed=SEED,
        )
        if len(pred) == 0:
            continue
        score = _rmse(y_holdout[pos], pred)
        if score < best_rmse:
            best_rmse = score
            best_cfg = cfg
    return (best_cfg or {}), best_rmse


def run(project_dir: Path, skip: set[str], tune_models: set[str]) -> None:
    _set_global_seeds()

    spy = _ensure_data(project_dir)
    features_df, spec = build_features(spy)
    features_path = project_dir / "data" / "features.csv"
    features_df.to_csv(features_path, index=False)
    print(f"[features] {len(features_df):,} rows, features={list(spec.feature_cols)}")

    pred_dir = project_dir / "data" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = project_dir / "artifacts" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = project_dir / "artifacts" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = list(spec.feature_cols)
    metric_rows: list[dict] = []
    tuning_rows: list[dict] = []

    for horizon in HORIZONS:
        target_col = f"target_{horizon}d"
        y_full = features_df[target_col].astype(float)
        X_full = features_df[feature_cols].astype(float)
        n = len(features_df)

        per_horizon_dfs: dict[str, list[pd.DataFrame]] = {m: [] for m in ALL_MODELS}

        # Cache last-fold LightGBM model + sample for SHAP rendering after the fold loop.
        last_lgbm_model = None
        last_lgbm_sample = None

        for fold in make_splits(n_samples=n, horizon=horizon):
            train_idx = fold.train_idx
            val_idx = fold.val_idx
            val_dates = features_df.iloc[val_idx]["Date"]
            y_val_true = y_full.iloc[val_idx].to_numpy(dtype=float)

            print(
                f"[fold {fold.fold}] horizon={horizon}d  train={len(train_idx)}  val={len(val_idx)}"
            )

            if "naive" not in skip:
                y_pred = predict_naive(features_df, val_idx, horizon)
                df = _save_fold_predictions(
                    pred_dir, "naive", horizon, fold.fold, val_dates, y_val_true, y_pred
                )
                per_horizon_dfs["naive"].append(df)

            if "garch" not in skip:
                returns = features_df["log_return"].astype(float)
                y_pred = predict_garch(returns, train_idx, val_idx, horizon)
                df = _save_fold_predictions(
                    pred_dir, "garch", horizon, fold.fold, val_dates, y_val_true, y_pred
                )
                per_horizon_dfs["garch"].append(df)

            if "lgbm" not in skip:
                X_train = X_full.iloc[train_idx]
                y_train = y_full.iloc[train_idx]
                X_val = X_full.iloc[val_idx]
                lgbm_params: dict = {}
                if "lgbm" in tune_models:
                    inner_train_pos, inner_val_pos = chronological_train_val_split(
                        train_idx=np.arange(len(train_idx)),
                        val_fraction=0.15,
                        horizon=horizon,
                    )
                    if len(inner_train_pos) > 30 and len(inner_val_pos) > 5:
                        lgbm_params, inner_rmse = _tune_lgbm_params(
                            X_train=X_train,
                            y_train=y_train,
                            inner_train_pos=inner_train_pos,
                            inner_val_pos=inner_val_pos,
                        )
                        tuning_rows.append(
                            {
                                "model": "lgbm",
                                "horizon": horizon,
                                "fold": fold.fold,
                                "inner_rmse": inner_rmse,
                                "params_json": json.dumps(lgbm_params, sort_keys=True),
                            }
                        )
                model = fit_lgbm(X_train, y_train, params=lgbm_params)
                y_pred = predict_lgbm(model, X_val)
                df = _save_fold_predictions(
                    pred_dir, "lgbm", horizon, fold.fold, val_dates, y_val_true, y_pred
                )
                per_horizon_dfs["lgbm"].append(df)
                last_lgbm_model = model
                # Build a SHAP sample from the last fold's val rows for stable explanations.
                last_lgbm_sample = X_val.copy()

            if "lstm" not in skip:
                X_train = X_full.iloc[train_idx]
                y_train = y_full.iloc[train_idx]
                X_val = X_full.iloc[val_idx]

                inner_train_pos, inner_val_pos = chronological_train_val_split(
                    train_idx=np.arange(len(train_idx)),
                    val_fraction=0.15,
                    horizon=horizon,
                )
                if len(inner_train_pos) < 30 or len(inner_val_pos) < 5:
                    # Skip extremely small folds rather than emit garbage predictions.
                    continue

                lstm_params: dict = {}
                if "lstm" in tune_models:
                    lstm_params, inner_rmse = _tune_lstm_params(
                        X_train=X_train,
                        y_train=y_train,
                        inner_train_pos=inner_train_pos,
                        inner_val_pos=inner_val_pos,
                        horizon=horizon,
                    )
                    tuning_rows.append(
                        {
                            "model": "lstm",
                            "horizon": horizon,
                            "fold": fold.fold,
                            "inner_rmse": inner_rmse,
                            "params_json": json.dumps(lstm_params, sort_keys=True),
                        }
                    )

                val_positions, preds = fit_predict_lstm(
                    X_train=X_train.reset_index(drop=True),
                    y_train=y_train.reset_index(drop=True),
                    X_val=X_val.reset_index(drop=True),
                    inner_train_idx_in_train=inner_train_pos,
                    inner_val_idx_in_train=inner_val_pos,
                    hidden=int(lstm_params.get("hidden", 64)),
                    dropout=float(lstm_params.get("dropout", 0.2)),
                    lr=float(lstm_params.get("lr", 1e-3)),
                    batch_size=int(lstm_params.get("batch_size", 64)),
                    seed=SEED,
                )
                if len(preds) > 0:
                    df = _save_fold_predictions(
                        pred_dir,
                        "lstm",
                        horizon,
                        fold.fold,
                        val_dates.iloc[val_positions],
                        y_val_true[val_positions],
                        preds,
                    )
                    per_horizon_dfs["lstm"].append(df)

        if last_lgbm_model is not None and last_lgbm_sample is not None:
            shap_path = plot_dir / f"shap_lgbm_{horizon}d.png"
            save_shap_summary(last_lgbm_model, last_lgbm_sample, shap_path, horizon)
            print(f"[shap] saved {shap_path}")

        for model_name, frames in per_horizon_dfs.items():
            if not frames:
                continue
            all_df = pd.concat(frames, ignore_index=True).sort_values("date")
            all_path = pred_dir / f"{model_name}_{horizon}d_all.csv"
            all_df.to_csv(all_path, index=False)
            mask = all_df["y_pred"].notna() & all_df["y_true"].notna()
            metrics = all_metrics(all_df.loc[mask, "y_true"], all_df.loc[mask, "y_pred"])
            for metric_name, value in metrics.items():
                metric_rows.append(
                    {
                        "model": model_name,
                        "horizon": horizon,
                        "metric": metric_name,
                        "value": value,
                    }
                )

    summary_long = pd.DataFrame(metric_rows)
    summary_long_path = summary_dir / "metrics_summary.csv"
    summary_long.to_csv(summary_long_path, index=False)

    if not summary_long.empty:
        wide = summary_long.pivot_table(
            index="model",
            columns=["horizon", "metric"],
            values="value",
        )
        wide = wide.reindex([m for m in ALL_MODELS if m in wide.index])
        wide_path = summary_dir / "metrics_summary_wide.csv"
        wide.to_csv(wide_path)
        print(f"[summary] saved {summary_long_path} and {wide_path}")
        print(wide.round(4).to_string())
    else:
        print("[summary] no metrics computed (all models skipped)")

    if tuning_rows:
        tuning_df = pd.DataFrame(tuning_rows)
        tuning_path = summary_dir / "tuning_summary.csv"
        tuning_df.to_csv(tuning_path, index=False)
        print(f"[tuning] saved {tuning_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip",
        type=str,
        default="",
        help="Comma-separated list of models to skip. E.g. --skip lstm,garch",
    )
    parser.add_argument(
        "--tune",
        type=str,
        default="lgbm,lstm",
        help="Comma-separated models to tune inside each fold. Default: lgbm,lstm",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}
    tune_models = {s.strip().lower() for s in args.tune.split(",") if s.strip()}
    tune_models = tune_models.intersection(DEFAULT_TUNING_MODELS)
    project_dir = Path(__file__).resolve().parent
    run(project_dir, skip, tune_models=tune_models)


if __name__ == "__main__":
    main()
