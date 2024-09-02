FROM nvcr.io/nvidia/tritonserver:24.06-py3 as base

RUN apt-get update

RUN pip install python-box==7.1.1
RUN pip install pydantic-settings
RUN pip install rectools
RUN pip install implicit
RUN pip install pathy
RUN pip install dill fsspec

ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV MODEL_REPOSITORY=/models
COPY ./ ./smartrec


EXPOSE 8000 8001 8002
