import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, val_array, test_array):
        try:
            logging.info("Splitting train, validation, and test arrays")

            # Separar features y target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_val, y_val = val_array[:, :-1], val_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Definir modelo base
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)

            # Hiperparámetros para grid search
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"]
            }

            logging.info("Starting GridSearchCV for Random Forest")
            grid_search = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                cv=3,  # validación cruzada dentro de train
                scoring="r2",
                verbose=2,
                n_jobs=-1
            )

            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            logging.info(f"Best Random Forest params: {grid_search.best_params_}")

            # Evaluar en validación
            y_val_pred = best_model.predict(X_val)
            val_r2 = r2_score(y_val, y_val_pred)

            logging.info(f"Validation R2 score: {val_r2:.4f}")

            if val_r2 < 0.6:
                raise CustomException("Validation score too low, model not reliable")

            # Guardar modelo entrenado
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Best Random Forest model saved successfully")

            # Evaluación final en test
            y_test_pred = best_model.predict(X_test)
            test_r2 = r2_score(y_test, y_test_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            report = {
                "best_model": "Random Forest",
                "best_params": grid_search.best_params_,
                "val_r2": val_r2,
                "test_r2": test_r2,
                "test_mse": test_mse,
                "test_mae": test_mae
            }

            return report

        except Exception as e:
            raise CustomException(e, sys)
