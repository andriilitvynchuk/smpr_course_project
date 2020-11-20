import multiprocessing as mp
from collections import namedtuple
from functools import partial
from itertools import product
from typing import List, NoReturn, Optional

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.svm import SVR
from tqdm import tqdm
from xgboost import XGBRegressor

from .dataset import create_ar_filter_table


MODEL_MAPPING = {
    "LinearRegression": LinearRegression(),
    "RidgeRegression": Ridge(),
    "LassoRegression": Lasso(),
    "SVM": SVR(),
    "XGBoostRegression": XGBRegressor(),
    "RandomForestRegression": RandomForestRegressor(),
}
MovingAverageGridParams = namedtuple("MovingAverageGridParams", ["q", "moving_average", "p"])


class BestFilterFinder:
    def __init__(self, model_name: str, validation_percent: float, processes: int = 1):
        self._validation_percent = validation_percent
        self._model_name = model_name
        self._processes = processes

    def _load_model(self) -> BaseEstimator:
        return MODEL_MAPPING.get(self._model_name, LinearRegression())

    def _train_and_evaluate_moving_average(
        self, grid_params: MovingAverageGridParams, variable: pd.Series
    ) -> NoReturn:
        model = self._load_model()

        filter_variable = variable.rolling(window=grid_params.moving_average).mean()
        x = create_ar_filter_table(
            variable=variable, p=grid_params.p, q=grid_params.q, filter_variable=filter_variable, filter_name="ma"
        )

    def grid_search_moving_average(self, variable: pd.Series, p: int, q: Optional[int]) -> NoReturn:
        q_range = range(1, min(int(len(variable) * 0.1), 105), 5)
        moving_average_range = range(0, min(int(len(variable) * 0.1), 105), 5)

        all_variants: List[MovingAverageGridParams] = [MovingAverageGridParams(q=0, moving_average=0, p=p)]
        for q, moving_average in product(q_range, moving_average_range):
            all_variants.append(MovingAverageGridParams(q=q, moving_average=moving_average, p=p))

        with mp.Pool(processes=self._processes) as pool:
            list(
                tqdm(
                    pool.imap(partial(self._train_and_evaluate_moving_average, variable=variable), all_variants),
                    total=len(all_variants),
                )
            )
