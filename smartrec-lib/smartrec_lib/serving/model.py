import json
import os
import logging

import numpy as np
import pandas as pd
import triton_python_backend_utils as pb_utils

from smartrec_lib.recommenders import (
    RecommenderALS,
    RecommenderLightFM,
    RecommenderPopular,
    RecommenderRandom,
)

logger = logging.getLogger(__name__)


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        # Get item_ids configuration
        item_ids_config = pb_utils.get_output_config_by_name(model_config, "item_ids")

        # Get scores configuration
        scores_config = pb_utils.get_output_config_by_name(model_config, "scores")

        strategy_config = pb_utils.get_output_config_by_name(model_config, "strategy")
        # Convert Triton types to numpy types

        self.item_ids_dtype = pb_utils.triton_string_to_numpy(item_ids_config["data_type"])
        self.scores_dtype = pb_utils.triton_string_to_numpy(scores_config["data_type"])

        self.strategy_dtype = pb_utils.triton_string_to_numpy(strategy_config["data_type"])

        script_path = os.path.dirname(os.path.abspath(__file__))

        # TODO переделать
        if "als" in self.model_config["name"]:
            self.model = RecommenderALS.load_model(load_dir=script_path)
        if "lightfm" in self.model_config["name"]:
            self.model = RecommenderLightFM.load_model(load_dir=script_path)
        if "popular" in self.model_config["name"]:
            self.model = RecommenderPopular.load_model(load_dir=script_path)
        if "random" in self.model_config["name"]:
            self.model = RecommenderRandom.load_model(load_dir=script_path)

    def convert_model_response_to_triton_response(self, model_responses):
        item_ids = pd.DataFrame(model_responses.item_ids)
        scores = pd.DataFrame(model_responses.scores)
        strategy = pd.DataFrame([model_responses.strategy])

        # Create output tensors. You need pb_utils.Tensor
        # objects to create pb_utils.InferenceResponse.
        item_ids_tensor = pb_utils.Tensor("item_ids", item_ids.values.astype(self.item_ids_dtype))
        scores_tensor = pb_utils.Tensor("scores", scores.values.astype(self.scores_dtype))
        strategy_tensor = pb_utils.Tensor("strategy", strategy.values.astype(self.strategy_dtype))

        inference_response = pb_utils.InferenceResponse(
            output_tensors=[item_ids_tensor, scores_tensor, strategy_tensor]
        )
        return inference_response

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        decoder = np.vectorize(lambda x: x.decode("UTF-8"))

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:

            user_ids = pb_utils.get_input_tensor_by_name(request, "user_ids").as_numpy()[0].decode("utf-8")
            top_n = pb_utils.get_input_tensor_by_name(request, "top_n").as_numpy()[0]
            filter_viewed = pb_utils.get_input_tensor_by_name(request, "filter_viewed").as_numpy()[0]

            # # Check for retrain_mode
            # retrain_mode_tensor = pb_utils.get_input_tensor_by_name(request, "retrain_mode")
            # retrain_mode = False
            # if retrain_mode_tensor is not None:
            #     retrain_mode = retrain_mode_tensor.as_numpy()[0]

            # # If retrain mode is enabled, process retraining data
            # if retrain_mode:
            #     logger.info("Retrain mode enabled - processing retraining request")
            #     try:
            #         # Get histories with user interactions
            #         histories_tensor = pb_utils.get_input_tensor_by_name(request, "histories")
            #         if histories_tensor is not None:
            #             histories = decoder(histories_tensor.as_numpy())
            #             logger.info(f"Received {len(histories)} user histories for retraining")

            #             # Convert to list of tour IDs
            #             user_tours = []
            #             for history_str in histories:
            #                 if history_str:
            #                     tours = [str(t.strip()) for t in history_str.split(",") if t.strip()]
            #                     user_tours.extend(tours)

            #             logger.info(f"Total interactions for retraining: {len(user_tours)}")

            #             # Perform model retraining
            #             await self._perform_model_retraining(user_tours)

            #             # Return success response
            #             item_ids = pd.DataFrame([["retrain_success"]])
            #             scores = pd.DataFrame([[1.0]])
            #             strategy = pd.DataFrame([["retraining_completed"]])

            #             item_ids_tensor = pb_utils.Tensor("item_ids", item_ids.values.astype(self.item_ids_dtype))
            #             scores_tensor = pb_utils.Tensor("scores", scores.values.astype(self.scores_dtype))
            #             strategy_tensor = pb_utils.Tensor("strategy", strategy.values.astype(self.strategy_dtype))

            #             inference_response = pb_utils.InferenceResponse(
            #                 output_tensors=[item_ids_tensor, scores_tensor, strategy_tensor]
            #             )
            #             responses.append(inference_response)
            #             continue
            #     except Exception as e:
            #         logger.error(f"Error in retrain mode: {e}", exc_info=True)

            # # Normal inference path
            item_ids = pb_utils.get_input_tensor_by_name(request, "item_ids")
            if item_ids is not None:
                item_ids = item_ids.as_numpy()[0].decode("utf-8")
            else:
                item_ids = None

            items_to_recommend = pb_utils.get_input_tensor_by_name(request, "items_to_recommend")
            if items_to_recommend is not None:
                items_to_recommend = decoder(items_to_recommend.as_numpy())
            else:
                items_to_recommend = None

            history = pb_utils.get_input_tensor_by_name(request, "history")
            if history is not None and len(history.as_numpy()) > 0:
                history = decoder(history.as_numpy())
            else:
                history = None

            recommendations = self.model.recommend(
                user_ids=user_ids,
                items_to_recommend=items_to_recommend,
                top_n=top_n,
                filter_viewed=filter_viewed,
                history=history,
            )

            inference_response = self.convert_model_response_to_triton_response(recommendations)

            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    # async def _perform_model_retraining(self, tour_ids):
    #     """
    #     Perform model retraining on received tour interactions.

    #     This is called when retrain_mode=True in the request.
    #     """
    #     try:
    #         logger.info(f"Starting model retraining with {len(tour_ids)} interactions")

    #         # Note: Full implementation would create a Dataset from tours
    #         # and call self.model.train_partial(dataset)
    #         # For now, this is a placeholder that logs the retraining intent

    #         logger.info("Model retraining completed successfully")
    #     except Exception as e:
    #         logger.error(f"Failed to perform model retraining: {e}", exc_info=True)

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
