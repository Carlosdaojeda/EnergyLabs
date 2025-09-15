import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    val_data_path: str = os.path.join('artifacts', 'val.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Leer dataset
            df = pd.read_csv(r'notebook\data\volve_wells.csv')
            logging.info("Dataset read successfully")

            # Imputar valores faltantes en target 'DT'
            if df['DT'].isnull().sum() > 0:
                df['DT'].fillna(df['DT'].median(), inplace=True)
                logging.info("Missing values in 'DT' column imputed with median")

            # Crear carpeta artifacts si no existe
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Guardar dataset completo
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved successfully")

            # Definir pozos
            training_wells = ['15/9-F-11 B', '15/9-F-11 A']
            val_well = '15/9-F-1 A'
            test_wells = ['15/9-F-1 B']

            # Split basado en pozos
            train_df = df[(df['WELL'].isin(training_wells)) & (df['WELL'] != val_well)]
            val_df = df[df['WELL'] == val_well]
            test_df = df[df['WELL'].isin(test_wells)]

            # Guardar datasets
            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            val_df.to_csv(self.ingestion_config.val_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data ingestion completed successfully with train/val/test split by wells")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.val_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Data ingestion
    ingestion = DataIngestion()
    train_path, val_path, test_path = ingestion.initiate_data_ingestion()

    # Data transformation
    transformer = DataTransformation()
    train_arr, val_arr, test_arr, _ = transformer.initiate_data_transformation(train_path, val_path, test_path)

    # Model training
    trainer = ModelTrainer()
    model_report = trainer.initiate_model_trainer(train_arr, val_arr, test_arr)
    print(model_report)
