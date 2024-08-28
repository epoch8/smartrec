import json
import os

import pandas as pd
import triton_python_backend_utils as pb_utils
from src.apps.recommendation.lib.recommender_als import RecommenderALS


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

        # Get post_ids configuration
        post_ids_config = pb_utils.get_output_config_by_name(model_config, "post_ids")
        #
        # Get scores configuration
        scores_config = pb_utils.get_output_config_by_name(model_config, "scores")

        # Get strategy configuration
        strategy_config = pb_utils.get_output_config_by_name(model_config, "strategy")

        # Convert Triton types to numpy types
        self.post_ids_dtype = pb_utils.triton_string_to_numpy(
            post_ids_config["data_type"]
        )
        self.scores_dtype = pb_utils.triton_string_to_numpy(scores_config["data_type"])
        self.strategy_dtype = pb_utils.triton_string_to_numpy(
            strategy_config["data_type"]
        )

        script_path = os.path.dirname(os.path.abspath(__file__))
        self.model: RecommenderALS = RecommenderALS(model_version="").load_model(
            load_dir=script_path, run_inference=True
        )

    def convert_model_response_to_triton_response(self, model_responses, strategy_dict):
        post_ids = pd.DataFrame(
            [recommendation.item_id for recommendation in model_responses]
        )
        scores = pd.DataFrame(
            [recommendation.score for recommendation in model_responses]
        )
        data = pd.DataFrame([strategy_dict["strategy"]])

        # Create output tensors. You need pb_utils.Tensor
        # objects to create pb_utils.InferenceResponse.
        post_ids_tensor = pb_utils.Tensor(
            "post_ids", post_ids.values.astype(self.post_ids_dtype)
        )
        scores_tensor = pb_utils.Tensor(
            "scores", scores.values.astype(self.scores_dtype)
        )
        strategy_tensor = pb_utils.Tensor(
            "strategy", data.values.astype(self.strategy_dtype)
        )

        inference_response = pb_utils.InferenceResponse(
            output_tensors=[post_ids_tensor, scores_tensor, strategy_tensor]
        )
        return inference_response

    def recommend(self, profile_id, top_n, model_type):
        recommendations, strategy_dict = self.model.recommend(
            user_id=profile_id,
            user_history=None,
            top_n=top_n,
            filter_already_liked_items=False,
            items=None,
            model_type=model_type,
        )
        inference_response = self.convert_model_response_to_triton_response(
            recommendations, strategy_dict
        )
        return inference_response

    def look_alike_items(self, post_id, top_n):
        recommendations, strategy_dict = self.model.look_alike_items(
            item_id=post_id, top_n=top_n
        )
        inference_response = self.convert_model_response_to_triton_response(
            recommendations, strategy_dict
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

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            method = pb_utils.get_input_tensor_by_name(request, "method")
            method = method.as_numpy()[0].decode("utf-8")
            limit = pb_utils.get_input_tensor_by_name(request, "limit").as_numpy()[0]
            model_type = (
                pb_utils.get_input_tensor_by_name(request, "model_type")
                .as_numpy()[0]
                .decode("utf-8")
            )

            if method == "recommend":
                profile_id = (
                    pb_utils.get_input_tensor_by_name(request, "profile_id")
                    .as_numpy()[0]
                    .decode("utf-8")
                )
                inference_response = self.recommend(profile_id, limit, model_type)
                responses.append(inference_response)

            elif method == "look_alike_items":
                post_id = (
                    pb_utils.get_input_tensor_by_name(request, "post_id")
                    .as_numpy()[0]
                    .decode("utf-8")
                )
                inference_response = self.look_alike_items(post_id, limit)
                responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
