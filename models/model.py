from typing import NoReturn

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor


MODEL_MAPPING = {
    "LinearRegression": LinearRegression(),
    "RidgeRegression": Ridge(),
    "LassoRegression": Lasso(),
    "SVM": SVR(),
    "XGBoostRegression": XGBRegressor(),
    "RandomForestRegression": RandomForestRegressor(),
}


class BestFilterFinder:
    def __init__(self, model_name: str, validation_percent: float):
        self._validation_percent = validation_percent
        self._load_model(model_name)

    def _load_model(self, model_name: str) -> NoReturn:
        self._model = MODEL_MAPPING.get(model_name, LinearRegression())
