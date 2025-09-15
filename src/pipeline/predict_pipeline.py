import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from sklearn.impute import SimpleImputer

class PredictPipeline:
    def __init__(self):
        try:
            # Cargar modelo y preprocessor (imputador)
            self.model_path = os.path.join("artifacts", "model.pkl")
            self.imputer_path = os.path.join("artifacts", "preprocessor.pkl")

            self.model = load_object(file_path=self.model_path)
            self.imputer = load_object(file_path=self.imputer_path)  # SimpleImputer
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        try:
            # Aplicar imputación solo si hay valores faltantes
            if features.isnull().values.any():
                features_imputed = self.imputer.transform(features)
            else:
                features_imputed = features.values  # usar directamente

            # Predecir
            preds = self.model.predict(features_imputed)
            return preds

        except Exception as e:
            raise CustomException(e, sys)

