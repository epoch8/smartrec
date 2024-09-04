import logging
import os
import pandas as pd
from typing import Dict, List, Optional, Tuple

import dill
from rectools import Columns
from implicit.als import AlternatingLeastSquares
from pathy import Pathy
from rectools.dataset import Dataset
from rectools.models import ImplicitALSWrapperModel
from rectools.dataset.identifiers import IdMap

from smartrec.lib.base import RecommenderModel
from smartrec.lib.save_and_load_triton_models import (
    clean_old_model_versions,
    edit_config_pbtxt,
    load_model,
    upload_model_files,
)
from smartrec.lib.model import ALSSettings, RecomItemUsers, RecomItem

logger = logging.getLogger(f"ALS Model")
logger.setLevel(logging.INFO)


class RecommenderALS(RecommenderModel):
    model_architecture = "als"
    
    def __init__(
        self,
        recsys_config: Optional[ALSSettings] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.model_version = model_version
        self.model_name = model_name
        self.recsys_config = recsys_config

        # base and feature models
        self.model: ImplicitALSWrapperModel = None
        self.item_model: ImplicitALSWrapperModel = None
        self.dataset: Dataset = None # might take unnecessary memory
        self.item_id_map: IdMap = None
        self.user_id_map: IdMap = None

    def train(self, dataset: Dataset):
        self.dataset = dataset

        logger.info("Fitting model...")

        self.model = ImplicitALSWrapperModel(
            AlternatingLeastSquares(
                factors=self.recsys_config.ALS_FACTORS,
                regularization=self.recsys_config.ALS_REGULARIZATION_FACTOR,
                iterations=self.recsys_config.ALS_ITERATIONS,
                alpha=self.recsys_config.ALS_ALPHA,
                random_state=self.recsys_config.ALS_RANDOM_STATE,
            ),
            fit_features_together=False,  # way to fit paired features
        )
        self.model.fit(dataset)
        self.user_id_map = dataset.user_id_map
        self.item_id_map = dataset.item_id_map
        logger.info("Base model trained.")
        
    def recommend(
        self,
        user_ids: str,
        top_n: int = 20,
        filter_viewed: bool = True,
    ) -> List[RecomItem]:  # Return type is a list of RecomItemUsers
        logger.info(f"Predicting for user {user_ids}")

        # user can be in the short memory, long memory or nowhere
        recos: pd.DataFrame = self.model.recommend(
            users=[user_ids],
            dataset=self.dataset,
            k=top_n,
            filter_viewed=filter_viewed,
        )
        recos = recos.sort_values(['user_id', 'score'], ascending=False).reset_index(drop=True)  # Assuming 'user_id' is the column name

        # Initialize an empty list to store the result
        # recom_item_users_list: List[RecomItemUsers] = []

        # Group the DataFrame by user_id
        # grouped = recos.groupby('user_id')


        reco_items = [
            RecomItem(item_id=str(row['item_id']), score=row['score'])
            for _, row in recos.iterrows()
        ]

        # # Create a RecomItemUsers object for the current user and append it to the list
        # recom_item_users = RecomItemUsers(user_id=user_id, reco_items=reco_items)
        # recom_item_users_list.append(recom_item_users)

        return reco_items
            

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

    @classmethod
    def load_model(
        cls,
        load_dir: str,
    ):
        """
        Load a trained model from a specified directory.

        Parameters:
            load_dir (str): The directory from which to load the model.

        Returns:
            RecommenderALS: The loaded model.
        """
        logger.info("Trying to load model from pkl")
    
        with open(os.path.join(load_dir, "model.pkl"), "rb") as file:
            state_dict = dill.load(file)
        
        # Create a new instance of RecommenderALS
        instance = cls()
        
        # Update the instance's __dict__ with the state_dict
        instance.__dict__.update(state_dict)
        
        logger.info("Model loaded successfully!")
        return instance

    
    @classmethod
    def load_model_triton(cls, base_s3_url: Pathy, model_name: str):
        """
        Load a trained model from a specified triton directory.

        Parameters:
            :param base_s3_url: The base URL in S3 (e.g., s3://bucket-name).
            :param model_name: The name of the model to load.

        Returns:
            RecommenderALS: The loaded model instance.
        """
        # Assuming load_model is a function that returns a state dictionary
        state_dict = load_model(base_s3_url=base_s3_url, model_name=model_name)
        
        # Create a new instance of RecommenderALS
        instance = cls()
        
        # Update the instance's __dict__ with the state_dict
        instance.__dict__.update(state_dict)
        
        logger.info("Model loaded successfully!")
        
        return instance

    def save_model_triton(self, base_s3_url: Pathy, num_to_keep: int) -> None:
        """
        Save the model to either a file or a stream.

        Parameters:
            :param fs: fsspec filesystem object.
            :param base_s3_url: The base URL in S3 (e.g., s3://bucket-name).
            :param num_to_keep: number of recent versions to keep.

        Returns:
            When saving to a file, returns None.
        """
        if self.model_version is None:
            raise Exception("There isn't model_version, please fill this field")

        logger.info(f"Saving model to {base_s3_url}")
        upload_model_files(
            base_s3_url,
            model_architecture = RecommenderALS.model_architecture,
            model_name=self.model_name,
            model_version=self.model_version,
            model_data=self.__dict__,
        )
        logger.info(f"Model saved successfully!")
        edit_config_pbtxt(
            base_s3_url=base_s3_url, 
            model_name=self.model_name
        )
        clean_old_model_versions(
            base_s3_url=base_s3_url, 
            model_name=self.model_name, 
            num_to_keep=num_to_keep
        )
        logger.info(f"Old models deleted!")

        return None
