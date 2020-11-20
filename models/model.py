import multiprocessing as mp
from collections import namedtuple
from functools import partial
from itertools import product
from typing import Callable, Iterable, List, NoReturn, Optional

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from tqdm import tqdm
from xgboost import XGBRegressor

from .dataset import create_ar_filter_table, create_next_day_price


MODEL_MAPPING = {
    "LinearRegression": LinearRegression(n_jobs=-1),
    "RidgeRegression": Ridge(),
    "LassoRegression": Lasso(),
    "SVM": SVR(),
    "XGBoostRegression": XGBRegressor(n_jobs=-1),
    "RandomForestRegression": RandomForestRegressor(n_jobs=-1),
}
MovingAverageGridParams = namedtuple("MovingAverageGridParams", ["q", "moving_average", "p"])
FinalMetrics = namedtuple("FinalMetrics", ["mae", "mse", "r2"])


class BestFilterFinder:
    def __init__(self, model_name: str, validation_percent: float, processes: int = 10):
        self._validation_percent = validation_percent
        self._model_name = model_name
        self._processes = processes

    def _load_model(self) -> BaseEstimator:
        return MODEL_MAPPING.get(self._model_name, LinearRegression())

    @staticmethod
    def get_scores(y_true: Iterable, y_predict: Iterable) -> FinalMetrics:
        mae = mean_absolute_error(y_true, y_predict)
        mse = mean_squared_error(y_true, y_predict)
        r2 = r2_score(y_true, y_predict)
        return FinalMetrics(mae=mae, mse=mse, r2=r2)

    @staticmethod
    def get_moving_average_filter(variable: pd.Series, grid_params: MovingAverageGridParams) -> pd.Series:
        return variable.rolling(window=grid_params.moving_average).mean()

    def _train_and_evaluate_moving_average(
        self, grid_params: MovingAverageGridParams, variable: pd.Series, get_filter_method: Callable
    ) -> FinalMetrics:
        filter_variable = get_filter_method(variable=variable, grid_params=grid_params)
        x = create_ar_filter_table(
            variable=variable, p=grid_params.p, q=grid_params.q, filter_variable=filter_variable, filter_name="ma"
        )
        x["next_day_price"] = create_next_day_price(variable=variable)

        x_train, x_test = (
            x.iloc[: int(len(variable) * (1 - self._validation_percent))],
            x.iloc[int(len(variable) * (1 - self._validation_percent)) :],
        )
        x_train = x_train.dropna()
        x_test = x_test.dropna()
        x_train, y_train = x_train.iloc[:, :-1], x_train.iloc[:, -1]

        x_test, y_test = x_test.iloc[:, :-1], x_test.iloc[:, -1]

        model = self._load_model()
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        metrics = BestFilterFinder.get_scores(y_test, y_predict)
        return metrics

    def grid_search_moving_average(self, variable: pd.Series, p: int, q: Optional[int]) -> NoReturn:
        q_range = range(1, min(int(len(variable) * 0.1), 105), 5)
        moving_average_range = range(1, min(int(len(variable) * 0.1), 105), 5)

        all_variants: List[MovingAverageGridParams] = [MovingAverageGridParams(q=0, moving_average=0, p=p)]
        for q, moving_average in product(q_range, moving_average_range):
            all_variants.append(MovingAverageGridParams(q=q, moving_average=moving_average, p=p))

        with mp.Pool(processes=self._processes) as pool:
            all_metrics = list(
                tqdm(
                    pool.imap(
                        partial(
                            self._train_and_evaluate_moving_average,
                            variable=variable,
                            get_filter_method=BestFilterFinder.get_moving_average_filter,
                        ),
                        all_variants,
                    ),
                    total=len(all_variants),
                )
            )
        print(sorted(zip(all_metrics, all_variants), key=lambda x: -x[0].mae, reverse=True)[:5])
