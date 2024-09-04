from typing import List, Dict
import numpy as np
from urllib.parse import urlparse
from tritonclient.utils import InferenceServerException


class TritonModelClient:
    """
    A wrapper over a model served by the Triton Inference Server.
    """

    def __init__(self, url: str, model_name: str):
        parsed_url = urlparse(url)
        self.model_name = model_name

        try:
            if parsed_url.scheme == "grpc":
                from tritonclient.grpc import InferenceServerClient, InferInput

                self.client = InferenceServerClient(parsed_url.netloc)  # Triton GRPC client
                self.metadata = self.client.get_model_metadata(self.model_name, as_json=True)
                self.infer_input = InferInput

            else:
                from tritonclient.http import InferenceServerClient, InferInput

                self.client = InferenceServerClient(parsed_url.netloc)  # Triton HTTP client
                self.metadata = self.client.get_model_metadata(self.model_name)
                self.infer_input = InferInput

        except InferenceServerException as e:
            raise RuntimeError(f"Failed to get metadata for model '{self.model_name}': {str(e)}")


    def __call__(self, **kwargs) -> Dict:
        """
        Invokes the model with named inputs provided via kwargs.
        Returns the inference result as a dictionary.
        """
        inputs = self._create_inputs(**kwargs)
        try:
            response = self.client.infer(model_name=self.model_name, inputs=inputs)
        except InferenceServerException as e:
            raise RuntimeError(f"Failed to perform inference on model '{self.model_name}': {str(e)}, params {kwargs}")

        result = {}

        # Process the output, decode if data type is 'BYTES', and skip None values
        for output in self.metadata["outputs"]:
            output_name = output["name"]
            output_data = response.as_numpy(output_name)

            if output_data is None:
                continue  # Skip if the output data is None

            if output["datatype"] == "BYTES":  # If it's BYTES, decode to UTF-8
                decoded_data = [x.decode('utf-8') for x in output_data.flatten()]  # Convert to list of strings
                result[output_name] = decoded_data
            else:
                result[output_name] = output_data.flatten().tolist()  # Convert to list of floats

        return result
    
    def _create_inputs(self, **kwargs):
        """Creates input tensors from kwargs."""
        placeholders = []
        
        for i in self.metadata["inputs"]:
            input_name = i["name"]
            value = kwargs.get(input_name)
            
            if value is not None:
                infer_input = self.infer_input(
                    name=input_name,
                    shape=[int(s) for s in i["shape"]],
                    datatype=i["datatype"]
                )
                
                # Set the data for the input
                infer_input.set_data_from_numpy(value)
                placeholders.append(infer_input)

        return placeholders
    
    
def recommendations_triton(triton_server_url: str, user_ids: str, model_name: str, top_n: int, filter_viewed: bool) -> dict:
    """
    Method to obtain recommendations from Triton Inference Server.

    :param triton_server_url: URL to Triton.
    :param user_ids: The list of user IDs.
    :param model_name: Model version name.    
    :param top_n: Limit on the number of results.
    :param filter_viewed: Whether to filter viewed items.
    
    :return: Dictionary with model version, data, and strategy.
    """
    model = TritonModelClient(triton_server_url, model_name)

    inputs = {
        "user_ids": np.array([user_ids], dtype=np.object_),  # Ensure user_ids is a 1D array
        "top_n": np.array([top_n], dtype=np.int32),
        "filter_viewed": np.array([filter_viewed], dtype=np.bool_),
    }

    recom_item_users = model(**inputs)
    
    print(model.metadata)

    return {"model_version": model.metadata["versions"][0], "data": recom_item_users}