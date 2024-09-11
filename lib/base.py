import os
import logging
from typing import Any, Dict, List, Optional, Tuple

from pathy import Pathy
import dill
from smartrec.lib.model import RecomItems

import pandas as pd

from smartrec.lib.save_and_load_triton_models import load_model


logger = logging.getLogger(f"Base Model")
logger.setLevel(logging.INFO)


class RecommenderModel:
    model_class: str
    model_version: str

    def __init__(self) -> None:
        pass

    def save_model(self, save_dir: str) -> None:
        raise NotImplementedError()

    @classmethod
    def load_model(cls, load_dir: str) -> "RecommenderModel":
        raise NotImplementedError()

    @classmethod
    def get_train_data(cls) -> pd.DataFrame:
        raise NotImplementedError()

    @classmethod
    def train(cls, train) -> "RecommenderModel":
        raise NotImplementedError()

    def status(self) -> Dict[str, Any]:
        raise NotImplementedError()
    
    def calc_metrics(self) -> Dict[str, Any]:
        raise NotImplementedError() 

    def recommend(
        self,
        user_id: str,
        user_history: List[str],
        top_n: int,
        filter_already_liked_items: bool,
        items: Optional[List[str]],
    ) -> RecomItems:
        raise NotImplementedError()

    def look_alike_items(
        self,
        item_id: str,
        top_n: int = 5,
        items: Optional[List[str]] = None,
    ) -> RecomItems:
        raise NotImplementedError()

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
            RecommenderModel: The loaded model.
        """
        logger.info("Trying to load model from pkl")
    
        with open(os.path.join(load_dir, "model.pkl"), "rb") as file:
            state_dict = dill.load(file)
        
        # Create a new instance of RecommenderModel
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
            RecommenderModel: The loaded model instance.
        """
        # Assuming load_model is a function that returns a state dictionary
        state_dict = load_model(base_s3_url=base_s3_url, model_name=model_name)
        
        # Create a new instance of RecommenderModel
        instance = cls()
        
        # Update the instance's __dict__ with the state_dict
        instance.__dict__.update(state_dict)
        
        logger.info("Model loaded successfully!")
        
        return instance