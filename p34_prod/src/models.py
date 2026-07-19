# models.py - train / predict.


# imports.
import numpy as np
from dataclasses import dataclass
from typing import Callable, Any


# One model description.
@dataclass(frozen=True)
class ModelSpec:
    name: str
    train_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], Any]
    predict_fn: Callable[[Any, np.ndarray], np.ndarray]
    prediction_type: str
    description: str = ""


# helper functions.
def _as_float32(X, y=None, w=None):
    X = np.asarray(X, dtype=np.float32)

    if y is None and w is None:
        return X

    y = np.asarray(y, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)

    return X, y, w

def _nan_to_zero(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        X,
        nan     = 0.0,
        posinf  = 0.0,
        neginf  = 0.0
    ).astype(np.float32)

def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float32)

    if len(w) == 0:
        return w

    mean_w = np.mean(w)

    if mean_w <= 0 or not np.isfinite(mean_w):
        return np.ones_like(w, dtype=np.float32)

    return w / mean_w

def _time_train_val_split(X, y, w, val_frac: float = 0.15):
    n = len(y)
    split_idx = int(n * (1.0 - val_frac))

    X_tr, X_val = X[:split_idx], X[split_idx:]
    y_tr, y_val = y[:split_idx], y[split_idx:]
    w_tr, w_val = w[:split_idx], w[split_idx:]

    return X_tr, X_val, y_tr, y_val, w_tr, w_val

def _winsorize_target(
    y: np.ndarray,
    quantiles: tuple[float, float] | None,
) -> np.ndarray:
    """
    Two-sided quantile clip of the regression target. Only the training
    target is clipped; predictions and scoring are untouched.
    """
    if quantiles is None:
        return y

    lo_q, hi_q = quantiles
    lo, hi = np.nanquantile(y, [lo_q, hi_q])

    if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
        return y

    return np.clip(y, lo, hi)


# models.

# rule-based baseline.
def make_rule_train_fn(
    feature_name: str,
    feature_names: list[str],
    quantile: float = 0.90,
    direction: str = "above"
):
    def train_fn(X, y, w):
        X = np.asarray(X, dtype=np.float32)

        feature_idx = feature_names.index(feature_name)
        threshold = np.nanquantile(X[:, feature_idx], quantile)

        return {
            "feature_name": feature_name,
            "feature_idx": feature_idx,
            "threshold": float(threshold),
            "direction": direction,
            "quantile": quantile
        }

    return train_fn

def make_rule_predict_fn():
    def predict_fn(model, X):
        X = np.asarray(X, dtype=np.float32)

        col = X[:, model["feature_idx"]]

        if model["direction"] == "above":
            return col.astype(np.float32)

        if model["direction"] == "below":
            return (-col).astype(np.float32)

        raise ValueError("direction must be 'above' or 'below'.")

    return predict_fn

# Ridge regression.
def make_ridge_train_fn(alpha: float = 1.0):
    def train_fn(X, y, w):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        X, y, w = _as_float32(X, y, w)

        X = _nan_to_zero(X)
        w = _normalize_weights(w)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = Ridge(alpha=alpha)
        model.fit(X_scaled, y, sample_weight=w)

        return {
            "model": model,
            "scaler": scaler,
        }

    return train_fn

def make_ridge_predict_fn():
    def predict_fn(bundle, X):
        X = _as_float32(X)
        X = _nan_to_zero(X)

        X_scaled = bundle["scaler"].transform(X)
        pred = bundle["model"].predict(X_scaled)

        return pred.astype(np.float32)

    return predict_fn

# Linear Huber ElasticNet.
def make_huber_elasticnet_train_fn(
    alpha: float = 1e-4,
    l1_ratio: float = 0.15,
    epsilon: float = 1.35,
    max_iter: int = 2000,
    tol: float = 1e-4,
):
    def train_fn(X, y, w):
        from sklearn.linear_model import SGDRegressor
        from sklearn.preprocessing import StandardScaler

        X, y, w = _as_float32(X, y, w)
        X = _nan_to_zero(X)
        w = _normalize_weights(w)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = SGDRegressor(
            loss            = "huber",
            penalty         = "elasticnet",
            alpha           = alpha,
            l1_ratio        = l1_ratio,
            epsilon         = epsilon,
            max_iter        = max_iter,
            tol             = tol,
            random_state    = 42,
            learning_rate   = "adaptive",
            eta0            = 0.01,
            average         = True
        )

        model.fit(X_scaled, y, sample_weight=w)

        return {
            "model": model,
            "scaler": scaler
        }

    return train_fn

def make_huber_elasticnet_predict_fn():
    def predict_fn(bundle, X):
        X = _as_float32(X)
        X = _nan_to_zero(X)

        X_scaled = bundle["scaler"].transform(X)
        pred = bundle["model"].predict(X_scaled)

        return pred.astype(np.float32)

    return predict_fn

# LGBM(MSE).
def make_lgbm_train_fn(
    max_depth: int = -1,
    num_leaves: int = 63,
    learning_rate: float = 0.05,
    n_estimators: int = 1000,
    min_child_samples: int = 500,
    reg_lambda: float = 1.0,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 50,
    val_frac: float = 0.15,
    verbose: int = -1,
    winsorize: tuple[float, float] | None = None,
):
    def train_fn(X, y, w):
        import lightgbm as lgb

        X, y, w = _as_float32(X, y, w)   # в train, без _nan_to_zero
        y = _winsorize_target(y, winsorize)

        X_tr, X_val, y_tr, y_val, w_tr, w_val = _time_train_val_split(
            X, y, w, val_frac=val_frac
        )

        model = lgb.LGBMRegressor(
            objective           = "regression",
            max_depth           = max_depth,
            num_leaves          = num_leaves,
            learning_rate       = learning_rate,
            n_estimators        = n_estimators,
            min_child_samples   = min_child_samples,
            reg_lambda          = reg_lambda,
            subsample           = subsample,
            colsample_bytree    = colsample_bytree,
            random_state        = 42,
            n_jobs              = -1,
            deterministic       = True,   # bit-reproducible across runs
            force_row_wise      = True,   # required with deterministic + multithread
            verbose             = verbose
        )

        model.fit(
            X_tr,
            y_tr,
            sample_weight       = w_tr,
            eval_set            = [(X_val, y_val)],
            eval_sample_weight  = [w_val],
            callbacks   = [
                lgb.early_stopping(
                    early_stopping_rounds,
                    verbose=verbose > 0,
                ),
                lgb.log_evaluation(period=0)
            ]
        )

        return model

    return train_fn

# LGBM(Huber).
def make_lgbm_huber_train_fn(
    max_depth: int = -1,
    huber_alpha: float = 0.9,
    num_leaves: int = 63,
    learning_rate: float = 0.05,
    n_estimators: int = 1000,
    min_child_samples: int = 500,
    reg_lambda: float = 1.0,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 50,
    val_frac: float = 0.15,
    verbose: int = -1,
    winsorize: tuple[float, float] | None = None,
):
    def train_fn(X, y, w):
        import lightgbm as lgb

        X, y, w = _as_float32(X, y, w)   # в train, без _nan_to_zero
        y = _winsorize_target(y, winsorize)

        X_tr, X_val, y_tr, y_val, w_tr, w_val = _time_train_val_split(
            X, y, w, val_frac=val_frac
        )

        model = lgb.LGBMRegressor(
            objective           = "huber",
            alpha               = huber_alpha,
            max_depth           = max_depth,
            num_leaves          = num_leaves,
            learning_rate       = learning_rate,
            n_estimators        = n_estimators,
            min_child_samples   = min_child_samples,
            reg_lambda          = reg_lambda,
            subsample           = subsample,
            colsample_bytree    = colsample_bytree,
            random_state        = 42,
            n_jobs              = -1,
            deterministic       = True,   # bit-reproducible across runs
            force_row_wise      = True,   # required with deterministic + multithread
            verbose             = verbose
        )

        model.fit(
            X_tr,
            y_tr,
            sample_weight       = w_tr,
            eval_set            = [(X_val, y_val)],
            eval_sample_weight  = [w_val],
            callbacks =[
                lgb.early_stopping(
                    early_stopping_rounds,
                    verbose=verbose > 0,
                ),
                lgb.log_evaluation(period=0)
            ]
        )

        return model

    return train_fn

# LGBM(Huber) ensemble.
class LGBMEnsembleRegressor:
    def __init__(self, models, weights=None, agg: str = "mean"):
        self.models = models
        self.agg = agg

        if weights is None:
            self.weights = np.ones(len(models), dtype=np.float32) / len(models)
        else:
            weights = np.asarray(weights, dtype=np.float32)
            self.weights = weights / weights.sum()

    def predict(self, X):
        preds = np.column_stack([
            model.predict(X).astype(np.float32)
            for model in self.models
        ])

        if self.agg == "mean":
            return preds.mean(axis=1).astype(np.float32)

        if self.agg == "weighted_mean":
            return (preds @ self.weights).astype(np.float32)

        if self.agg == "median":
            return np.median(preds, axis=1).astype(np.float32)

        if self.agg == "rank_mean":
            # Scale-free aggregation: members are combined through their
            # prediction ranks, so member output scales do not matter.
            from scipy.stats import rankdata

            ranks = np.column_stack([
                rankdata(preds[:, k]) for k in range(preds.shape[1])
            ])
            return (ranks.mean(axis=1) / len(preds)).astype(np.float32)

        raise ValueError(f"Unknown aggregation: {self.agg}")

    @property
    def feature_importances_(self):
        importances = []

        for model in self.models:
            if hasattr(model, "feature_importances_"):
                importances.append(model.feature_importances_)

        if len(importances) == 0:
            return None

        return np.mean(np.vstack(importances), axis=0)
    
def make_lgbm_huber_ensemble_train_fn(
    configs: list[dict] | None = None,
    agg: str = "mean",
    early_stopping_rounds: int = 50,
    val_frac: float = 0.15,
    verbose: int = -1,
    base_random_state: int = 42,
    winsorize: tuple[float, float] | None = None,
):
    if configs is None:
        configs = [
            {
                "max_depth": -1,
                "huber_alpha": 0.85,
                "num_leaves": 31,
                "learning_rate": 0.03,
                "n_estimators": 1200,
                "min_child_samples": 300,
                "reg_lambda": 1.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            {
                "max_depth": -1,
                "huber_alpha": 0.90,
                "num_leaves": 63,
                "learning_rate": 0.05,
                "n_estimators": 1000,
                "min_child_samples": 500,
                "reg_lambda": 1.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            {
                "max_depth": 6,
                "huber_alpha": 0.90,
                "num_leaves": 31,
                "learning_rate": 0.04,
                "n_estimators": 1200,
                "min_child_samples": 700,
                "reg_lambda": 2.0,
                "subsample": 0.9,
                "colsample_bytree": 0.7,
            },
            {
                "max_depth": 8,
                "huber_alpha": 0.95,
                "num_leaves": 127,
                "learning_rate": 0.03,
                "n_estimators": 1500,
                "min_child_samples": 800,
                "reg_lambda": 3.0,
                "subsample": 0.7,
                "colsample_bytree": 0.9,
            },
            {
                "max_depth": 4,
                "huber_alpha": 0.85,
                "num_leaves": 15,
                "learning_rate": 0.06,
                "n_estimators": 800,
                "min_child_samples": 400,
                "reg_lambda": 0.5,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
            },
        ]

    def train_fn(X, y, w):
        import lightgbm as lgb

        X, y, w = _as_float32(X, y, w)
        y = _winsorize_target(y, winsorize)

        X_tr, X_val, y_tr, y_val, w_tr, w_val = _time_train_val_split(
            X, y, w, val_frac=val_frac
        )

        models = []

        for k, cfg in enumerate(configs):
            cfg = dict(cfg)

            model_name = cfg.pop("name", f"lgbm_huber_{k}")
            model_weight = cfg.pop("weight", None)

            model = lgb.LGBMRegressor(
                objective           = "huber",
                alpha               = cfg.get("huber_alpha", 0.9),
                max_depth           = cfg.get("max_depth", -1),
                num_leaves          = cfg.get("num_leaves", 63),
                learning_rate       = cfg.get("learning_rate", 0.05),
                n_estimators        = cfg.get("n_estimators", 1000),
                min_child_samples   = cfg.get("min_child_samples", 500),
                reg_lambda          = cfg.get("reg_lambda", 1.0),
                subsample           = cfg.get("subsample", 0.8),
                colsample_bytree    = cfg.get("colsample_bytree", 0.8),
                random_state        = cfg.get("random_state", base_random_state + k),
                n_jobs              = -1,
                deterministic       = True,   # bit-reproducible across runs
                force_row_wise      = True,   # required with deterministic + multithread
                verbose             = verbose,
            )

            if verbose > 0:
                print(f"Training ensemble member {k + 1}/{len(configs)}: {model_name}")

            model.fit(
                X_tr,
                y_tr,
                sample_weight       = w_tr,
                eval_set            = [(X_val, y_val)],
                eval_sample_weight  = [w_val],
                callbacks=[
                    lgb.early_stopping(
                        early_stopping_rounds,
                        verbose=verbose > 0,
                    ),
                    lgb.log_evaluation(period=0),
                ],
            )

            models.append(model)

        weights = [
            cfg.get("weight", 1.0)
            for cfg in configs
        ]

        return LGBMEnsembleRegressor(
            models=models,
            weights=weights,
            agg=agg,
        )

    return train_fn

def make_lgbm_predict_fn():
    def predict_fn(model, X):
        X = _as_float32(X)

        pred = model.predict(X)

        return pred.astype(np.float32)

    return predict_fn


# registry.
ENSEMBLE_MEMBER_CONFIGS = [
    {
        "name": "huber_080_small",
        "random_state": 42,
        "huber_alpha": 0.80,
        "max_depth": 5,
        "num_leaves": 31,
        "learning_rate": 0.04,
        "n_estimators": 1200,
        "min_child_samples": 700,
        "reg_lambda": 3.0,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "weight": 1.0,
    },
    {
        "name": "huber_085_medium",
        "random_state": 42,
        "huber_alpha": 0.85,
        "max_depth": 6,
        "num_leaves": 63,
        "learning_rate": 0.04,
        "n_estimators": 1200,
        "min_child_samples": 600,
        "reg_lambda": 2.0,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "weight": 1.0,
    },
    {
        "name": "huber_090_base",
        "random_state": 42,
        "huber_alpha": 0.90,
        "max_depth": -1,
        "num_leaves": 63,
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "min_child_samples": 500,
        "reg_lambda": 1.0,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "weight": 1.0,
    },
    {
        "name": "huber_095_large",
        "random_state": 42,
        "huber_alpha": 0.95,
        "max_depth": -1,
        "num_leaves": 127,
        "learning_rate": 0.03,
        "n_estimators": 1500,
        "min_child_samples": 800,
        "reg_lambda": 2.0,
        "subsample": 0.75,
        "colsample_bytree": 0.90,
        "weight": 1.0,
    },
]


def build_model_registry(feature_names: list[str]) -> dict[str, ModelSpec]:
    return {
        "rule_dist_to_mid": ModelSpec(
            name        = "rule_dist_to_mid",
            train_fn    = make_rule_train_fn(
                feature_name    = "dist_to_mid",
                feature_names   = feature_names,
                quantile        = 0.90,
                direction       = "above"
            ),
            predict_fn      = make_rule_predict_fn(),
            prediction_type = "toxicity",
            description     = "Rule-based filter by high dist_to_mid."
        ),

        "ridge": ModelSpec(
            name            ="ridge",
            train_fn        = make_ridge_train_fn(alpha=1.0),
            predict_fn      = make_ridge_predict_fn(),
            prediction_type = "pnl",
            description     = "Ridge regression with weighted MSE."
        ),

        "huber_elasticnet": ModelSpec(
            name        = "huber_elasticnet",
            train_fn    = make_huber_elasticnet_train_fn(
                alpha    =1e-4,
                l1_ratio =0.15,
                epsilon  =1.35,
                max_iter =2000
            ),
            predict_fn      =make_huber_elasticnet_predict_fn(),
            prediction_type ="pnl",
            description ="Linear Huber regression with ElasticNet regularization."
        ),

        "lgbm": ModelSpec(
            name            = "lgbm",
            train_fn        = make_lgbm_train_fn(),
            predict_fn      = make_lgbm_predict_fn(),
            prediction_type = "pnl",
            description     ="LightGBM regressor with weighted MSE."
        ),

        "lgbm_huber": ModelSpec(
            name            = "lgbm_huber",
            train_fn        = make_lgbm_huber_train_fn(huber_alpha=0.9),
            predict_fn      = make_lgbm_predict_fn(),
            prediction_type = "pnl",
            description     = "LightGBM regressor with Huber loss."
        ),

        "lgbm_huber_ensemble_mean": ModelSpec(
            name            = "lgbm_huber_ensemble_mean",
            train_fn        = make_lgbm_huber_ensemble_train_fn(
                configs               = ENSEMBLE_MEMBER_CONFIGS,
                agg                   = "mean",
                early_stopping_rounds = 50,
                val_frac              = 0.15,
                verbose               = -1,
                base_random_state     = 42,
            ),
            predict_fn      = make_lgbm_predict_fn(),
            prediction_type = "pnl",
            description     = "Ensemble of LightGBM Huber regressors with different Huber alpha and tree hyperparameters. Predictions are averaged."
        ),

        "lgbm_huber_ensemble_rank": ModelSpec(
            name            = "lgbm_huber_ensemble_rank",
            train_fn        = make_lgbm_huber_ensemble_train_fn(
                configs               = ENSEMBLE_MEMBER_CONFIGS,
                agg                   = "rank_mean",
                early_stopping_rounds = 50,
                val_frac              = 0.15,
                verbose               = -1,
                base_random_state     = 42,
            ),
            predict_fn      = make_lgbm_predict_fn(),
            prediction_type = "pnl",
            description     = (
                "Ensemble of LightGBM Huber regressors aggregated by rank "
                "averaging - scale-free, matches the rank-based downstream filter."
            )
        ),

    }


# get model.
def get_model(
    name: str,
    feature_names: list[str],
) -> ModelSpec:
    registry = build_model_registry(feature_names)

    if name not in registry:
        available = ", ".join(registry.keys())
        raise ValueError(
            f"Unknown model '{name}'. Available models: {available}"
        )

    return registry[name]


# list of the models.
def list_models(feature_names: list[str]) -> list[str]:
    return list(build_model_registry(feature_names).keys())



