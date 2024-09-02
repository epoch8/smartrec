import json
import os

import pandas as pd
import triton_python_backend_utils as pb_utils
from smartrec.lib.recommender_als import RecommenderALS



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
        
        print(f"{self.model_config=}")
        # Get user_ids configuration
        user_ids_config = pb_utils.get_output_config_by_name(model_config, "user_ids")
        
        # Get item_ids configuration
        item_ids_config = pb_utils.get_output_config_by_name(model_config, "item_ids")
        #
        # Get scores configuration
        scores_config = pb_utils.get_output_config_by_name(model_config, "scores")
        
        print(f"{user_ids_config=}")


        # Convert Triton types to numpy types
        self.user_ids_dtype = pb_utils.triton_string_to_numpy(
            user_ids_config["data_type"]
        )
        self.item_ids_dtype = pb_utils.triton_string_to_numpy(
            item_ids_config["data_type"]
        )
        self.scores_dtype = pb_utils.triton_string_to_numpy(scores_config["data_type"])
        
        script_path = os.path.dirname(os.path.abspath(__file__))
        self.model: RecommenderALS = RecommenderALS.load_model(
            load_dir=script_path
        )
        
    def convert_model_response_to_triton_response(self, model_responses):
        user_ids = pd.DataFrame(
            [recommendation.user_id for recommendation in model_responses]
        )
        item_ids = pd.DataFrame(
            [recommendation.item_id for recommendation in model_responses]
        )
        scores = pd.DataFrame(
            [recommendation.score for recommendation in model_responses]
        )

        # Create output tensors. You need pb_utils.Tensor
        # objects to create pb_utils.InferenceResponse.
        user_ids_tensor = pb_utils.Tensor(
            "user_ids", user_ids.values.astype(self.user_ids_dtype)
        )
        item_ids_tensor = pb_utils.Tensor(
            "item_ids", item_ids.values.astype(self.item_ids_dtype)
        )
        scores_tensor = pb_utils.Tensor(
            "scores", scores.values.astype(self.scores_dtype)
        )

        inference_response = pb_utils.InferenceResponse(
            output_tensors=[user_ids_tensor, item_ids_tensor, scores_tensor]
        )
        return inference_response

    def recommend(self, user_ids, top_n, filter_viewed):
        recommendations = self.model.recommend(
            user_ids=user_ids, top_n=top_n, filter_viewed=filter_viewed
        )
        inference_response = self.convert_model_response_to_triton_response(
            recommendations
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
        print(f"{requests=}")
        for request in requests:
            user_ids = (
                pb_utils.get_input_tensor_by_name(request, "user_ids")
                .as_numpy().decode("utf-8")
            )
            top_n = pb_utils.get_input_tensor_by_name(request, "top_n").as_numpy()[0]
            filter_viewed = (
                pb_utils.get_input_tensor_by_name(request, "filter_viewed")
                .as_numpy()
            )
            inference_response = self.recommend(user_ids, top_n, filter_viewed)
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
