import dill
import fsspec
import os
import implicit
import datetime as dt
import numpy as np
import logging
import pandas as pd
from pathy import Pathy
from rectools import Columns
from rectools.models.vector import Distance, Factors, ImplicitRanker, InternalIds, Scores
from smartrec.lib.base import RecomItem, RecommenderModel
from settings import recsys_config
from typing import Any, Dict, List, Optional, Tuple, Union
from rectools.dataset import Dataset
from implicit.als import AlternatingLeastSquares
from rectools.models import ImplicitALSWrapperModel, ImplicitItemKNNWrapperModel
from smartrec.lib.save_model import upload_model_files, clean_old_model_versions, edit_config_pbtxt


logger = logging.getLogger(f"ALS Model")
logger.setLevel(logging.INFO)


class RecommenderALS(RecommenderModel):
    def __init__(
        self,
        model_name: str,
        model_version: str,
    ) -> None:
        super().__init__()
    
        self.model_version = model_version
        self.model_name = "als_v3"
        self.als_factors = recsys_config.ALS_FACTORS

        # base and feature models
        self.model: ImplicitALSWrapperModel = None
        self.item_model: ImplicitALSWrapperModel = None
      
    def train(self, dataset: Dataset):

        logger.info("Fitting model...")
        
        model = ImplicitALSWrapperModel(
            AlternatingLeastSquares(
                factors=recsys_config.ALS_FACTORS,  
                regularization=recsys_config.ALS_REGULARIZATION_FACTOR,
                iterations=recsys_config.ALS_ITERATIONS,
                alpha=recsys_config.ALS_ALPHA,  
                random_state=recsys_config.ALS_RANDOM_STATE,
            ),
            fit_features_together=False,  # way to fit paired features
        )
        model.fit(dataset)
        logger.info("Base model trained.")
        self.model = model
        
    def save_model(self, save_dir: str) -> None:
        """
        Save the model to either a file or a stream.

        Parameters:
            save_dir (str): The directory where the model will be saved.

        Returns:
            When saving to a file, returns None.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logger.info(f"Saving model to {os.path.abspath(save_dir)}")

        with open(os.path.join(save_dir, "model.pkl"), "wb") as file:
            dill.dump(self.__dict__, file)
        logger.info(f"Model saved successfully!")
        

    def load_model(
        self,
        load_dir: str,
    ):
        """
        Load a trained model from a specified directory.

        Parameters:
            load_dir (str): The directory from which to load the model.

        Returns:
            RecommenderALS: The loaded model.
        """
        with open(os.path.join(load_dir, "model.pkl"), "rb") as file:
            state_dict = dill.load(file)
        self.__dict__.update(state_dict)
        logger.info("Model loaded successfully!")
        return self
    
    def save_model_triton(self, fs: fsspec.AbstractFileSystem, base_s3_url: Pathy, num_to_keep: int) -> None:
        """
        Save the model to either a file or a stream.

        Parameters:
            :param fs: fsspec filesystem object.
            :param base_s3_url: The base URL in S3 (e.g., s3://bucket-name).
            :param num_to_keep: number of recent versions to keep.

        Returns:
            When saving to a file, returns None.
        """
        logger.info(f"Saving model to {base_s3_url}")
        edit_config_pbtxt(fs, base_s3_url, model_name=self.model_name)
        upload_model_files(fs, base_s3_url, model_name=self.model_name, model_version=self.model_version, model_data=self.__dict__)
        logger.info(f"Model saved successfully!")
        clean_old_model_versions(fs, base_s3_url, model_name=self.model_name, num_to_keep=num_to_keep)
        logger.info(f"Old models deleted!")
        return None
        