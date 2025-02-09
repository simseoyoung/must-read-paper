from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb.catboost as wandb_cb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from omegaconf import DictConfig

from models.base import BaseModel


class XGBoostTrainer(BaseModel):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> xgb.Booster:
        dtrain = xgb.DMatrix(X_train, y_train)
        dvalid = xgb.DMatrix(X_valid, y_valid)

        model = xgb.train(
            dict(self.cfg.models.params),
            dtrain=dtrain,
            evals=[(dtrain, "train"), (dvalid, "eval")],
            num_boost_round=self.cfg.models.num_boost_round,
            early_stopping_rounds=self.cfg.models.early_stopping_rounds,
            verbose_eval=self.cfg.models.verbose_eval,
        )

        return model


class CatBoostTrainer(BaseModel):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> CatBoostClassifier:
        train_set = Pool(X_train, y_train, cat_features=self.cfg.features.cat_features)
        valid_set = Pool(X_valid, y_valid, cat_features=self.cfg.features.cat_features)

        model = CatBoostClassifier(
            random_state=self.cfg.models.seed,
            cat_features=self.cfg.features.cat_features,
            **self.cfg.models.params,
        )

        model.fit(
            train_set,
            eval_set=valid_set,
            verbose_eval=self.cfg.models.verbose_eval,
            early_stopping_rounds=self.cfg.models.early_stopping_rounds,
            callbacks=[wandb_cb.WandbCallback()],
        )
        return model


class LightGBMTrainer(BaseModel):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> lgb.Booster:
        train_set = lgb.Dataset(X_train, y_train, categorical_feature=self.cfg.features.cat_features)
        valid_set = lgb.Dataset(X_valid, y_valid, categorical_feature=self.cfg.features.cat_features)

        model = lgb.train(
            train_set=train_set,
            valid_sets=[train_set, valid_set],
            params=dict(self.cfg.models.params),
            num_boost_round=self.cfg.models.num_boost_round,
            callbacks=[
                lgb.log_evaluation(self.cfg.models.verbose_eval),
                lgb.early_stopping(self.cfg.models.early_stopping_rounds),
            ],
        )

        return model
