import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Simple preprocessing: imputar valores numéricos faltantes con la mediana.
        No escalamos, ya que Random Forest no lo necesita.
        """
        try:
            num_imputer = SimpleImputer(strategy="median")
            return num_imputer
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, val_path, test_path):
        try:
            # Leer datasets
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train, validation and test data completed")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "DT"
            numerical_columns = ['RHOB', 'GR', 'NPHI', 'PEF']

            # Separar features y target
            X_train = train_df[numerical_columns]
            y_train = train_df[target_column_name]

            X_val = val_df[numerical_columns]
            y_val = val_df[target_column_name]

            X_test = test_df[numerical_columns]
            y_test = test_df[target_column_name]

            logging.info("Applying preprocessing object on training, validation and testing dataframes.")

            # Aplicar imputación
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_val_arr = preprocessing_obj.transform(X_val)
            X_test_arr = preprocessing_obj.transform(X_test)

            # Concatenar features y target
            train_arr = np.c_[X_train_arr, np.array(y_train)]
            val_arr = np.c_[X_val_arr, np.array(y_val)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            logging.info("Saved preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, val_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
