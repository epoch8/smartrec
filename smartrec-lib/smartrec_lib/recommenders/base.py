import logging
import os
from abc import abstractmethod
from typing import Any, Dict, List, Optional

import dill
from pathy import Pathy
from rectools.dataset import Dataset

from smartrec_lib.model import RecomItems
from smartrec_lib.save_and_load_triton_models import load_model_s3

logger = logging.getLogger("Base Model")
logger.setLevel(logging.INFO)


class RecommenderModel:
    model_class: str
    model_version: str

    def __init__(self) -> None:
        pass

    @abstractmethod
    def train(cls, train) -> "RecommenderModel":
        raise NotImplementedError()

    @abstractmethod
    def calc_metrics(self, k: int, dataset: Dataset) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def recommend(
        self,
        user_ids: int,
        top_n: int,
        filter_viewed: bool,
        items_to_recommend: Optional[List[int]] = None,  # TODO change to List[str]
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
        logger.info("Model saved successfully!")

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
        state_dict = load_model_s3(base_s3_url=base_s3_url, model_name=model_name)

        # Create a new instance of RecommenderModel
        instance = cls()

        # Update the instance's __dict__ with the state_dict
        instance.__dict__.update(state_dict)

        logger.info("Model loaded successfully!")

        return instance
