import json
import os
import sys
import logging
from time import perf_counter

import numpy as np
import pandas as pd
import triton_python_backend_utils as pb_utils

from smartrec_lib.recommenders import (
    RecommenderALS,
    RecommenderCoVis,
    RecommenderEASE,
    RecommenderPopular,
)

# Явно настраиваем логгер на stdout — без этого логи не видны в kubectl logs
logger = logging.getLogger("triton_model")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setLevel(logging.DEBUG)
    _handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)


class _TritonLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            if record.levelno >= logging.ERROR:
                pb_utils.Logger.log_error(message)
            elif record.levelno >= logging.WARNING:
                if hasattr(pb_utils.Logger, "log_warn"):
                    pb_utils.Logger.log_warn(message)
                elif hasattr(pb_utils.Logger, "log_warning"):
                    pb_utils.Logger.log_warning(message)
                else:
                    pb_utils.Logger.log_info(f"[WARN] {message}")
            else:
                pb_utils.Logger.log_info(message)
        except Exception:
            return


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    @staticmethod
    def _log_info(message: str) -> None:
        pb_utils.Logger.log_info(message)

    @staticmethod
    def _configure_python_logging_bridge() -> None:
        recommender_logger_names = (
            "ALS Model",
            "Popular Model",
            "EASE Model",
            "CoVis Model",
            "Base Model",
            "ALS Model saving stage:",
        )
        for logger_name in recommender_logger_names:
            target_logger = logging.getLogger(logger_name)
            has_bridge = any(isinstance(handler, _TritonLoggingHandler) for handler in target_logger.handlers)
            if has_bridge:
                continue
            handler = _TritonLoggingHandler()
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter("%(message)s"))
            target_logger.addHandler(handler)
            target_logger.setLevel(logging.INFO)
            target_logger.propagate = False

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
        init_start = perf_counter()

        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])
        self._configure_python_logging_bridge()

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

        model_load_start = perf_counter()
        model_name = self.model_config["name"]
        # Ordered resolution: substring matching bites compound names, so the
        # chain is a single elif ladder with the most specific names first.
        # "als_covis_youtravel" MUST load as RecommenderALS (the covis session
        # layer lives inside it) - the old independent ifs let the bare covis
        # branch overwrite it, serving an empty-neighbors RecommenderCoVis.
        if "als_covis" in model_name:
            self.model = RecommenderALS.load_model(load_dir=script_path)
        elif "als" in model_name:
            self.model = RecommenderALS.load_model(load_dir=script_path)
        elif "popular" in model_name:
            self.model = RecommenderPopular.load_model(load_dir=script_path)
        elif "ease" in model_name:
            self.model = RecommenderEASE.load_model(load_dir=script_path)
        elif "covis" in model_name:
            self.model = RecommenderCoVis.load_model(load_dir=script_path)
        model_load_ms = (perf_counter() - model_load_start) * 1000

        cache_warm_ms = 0.0
        if "als" in self.model_config["name"] and hasattr(self.model, "_ensure_user_item_matrix_binary"):
            cache_warm_start = perf_counter()
            self.model._ensure_lookup_caches()
            self.model._ensure_user_item_matrix_binary()
            cache_warm_ms = (perf_counter() - cache_warm_start) * 1000

        init_ms = (perf_counter() - init_start) * 1000
        self._log_info(
            f"[TRITON MODEL INIT] model={self.model_config['name']} | "
            f"total={init_ms:.1f}ms, model_load={model_load_ms:.1f}ms, cache_warm={cache_warm_ms:.1f}ms"
        )

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
        execute_start = perf_counter()
        decoder = np.vectorize(lambda x: x.decode("UTF-8"))

        print(f"[TRITON EXECUTE] batch_size={len(requests)}", flush=True)

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            request_start = perf_counter()

            # ----- Парсинг входных данных -----
            parse_start = perf_counter()
            user_ids = pb_utils.get_input_tensor_by_name(request, "user_ids").as_numpy()[0].decode("utf-8")
            top_n = pb_utils.get_input_tensor_by_name(request, "top_n").as_numpy()[0]
            filter_viewed = pb_utils.get_input_tensor_by_name(request, "filter_viewed").as_numpy()[0]

            # Normal inference path
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

            parse_ms = (perf_counter() - parse_start) * 1000

            # ----- Вызов модели рекомендаций -----
            recommend_start = perf_counter()
            recommend_kwargs = dict(
                user_ids=user_ids,
                items_to_recommend=items_to_recommend,
                top_n=top_n,
                filter_viewed=filter_viewed,
                history=history,
            )
            recommendations = self.model.recommend(**recommend_kwargs)
            recommend_ms = (perf_counter() - recommend_start) * 1000

            # ----- Конвертация ответа в Triton формат -----
            convert_start = perf_counter()
            inference_response = self.convert_model_response_to_triton_response(recommendations)
            convert_ms = (perf_counter() - convert_start) * 1000

            request_ms = (perf_counter() - request_start) * 1000

            # ----- Логирование -----
            history_size = len(history) if history is not None else 0
            items_count = len(items_to_recommend) if items_to_recommend is not None else 0
            result_count = len(recommendations.item_ids) if recommendations.item_ids is not None else 0

            log_msg = (
                f"[TRITON MODEL] user={user_ids}, top_n={top_n}, history={history_size}, items_to_rec={items_count} | "
                f"total={request_ms:.1f}ms | parse={parse_ms:.1f}ms, recommend={recommend_ms:.1f}ms, convert={convert_ms:.1f}ms | "
                f"strategy={recommendations.strategy}, results={result_count}"
            )
            self._log_info(log_msg)
            logger.info(log_msg)
            print(log_msg, flush=True)

            responses.append(inference_response)

        # Логирование общего времени batch execute (всегда, даже для 1 запроса)
        execute_ms = (perf_counter() - execute_start) * 1000
        execute_log = f"[TRITON MODEL EXECUTE] requests={len(requests)}, total_execute={execute_ms:.1f}ms"
        self._log_info(execute_log)
        logger.info(execute_log)
        print(execute_log, flush=True)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
