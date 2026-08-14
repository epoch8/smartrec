import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import dill
import fsspec
from pathy import Pathy

CURRENT_DIR = Path(__file__).parent
SERVING_FOLDER_PATH = CURRENT_DIR.parent / "smartrec_lib/serving"

logger = logging.getLogger("ALS Model saving stage:")


def get_s3_filesystem(url: str):
    """
    Get S3 filesystem with proper configuration for Yandex Cloud or AWS.

    :param url: S3 URL (e.g., s3://bucket-name)
    :return: Configured fsspec filesystem
    """
    storage_options = {}

    # Check if we have custom S3 endpoint (for Yandex Cloud or other S3-compatible storage)
    if "S3_ENDPOINT" in os.environ or "AWS_ENDPOINT_URL" in os.environ:
        endpoint = os.getenv("S3_ENDPOINT") or os.getenv("AWS_ENDPOINT_URL")
        storage_options["client_kwargs"] = {"endpoint_url": endpoint}
        logger.info(f"Using custom S3 endpoint: {endpoint}")

        # Add region if specified
        if "AWS_DEFAULT_REGION" in os.environ:
            region = os.getenv("AWS_DEFAULT_REGION")
            storage_options["client_kwargs"]["region_name"] = region
            logger.info(f"Using region: {region}")

    # Add credentials if specified
    if "AWS_ACCESS_KEY_ID" in os.environ:
        key_id = os.getenv("AWS_ACCESS_KEY_ID")
        storage_options["key"] = key_id
        logger.info(f"Using AWS_ACCESS_KEY_ID: {key_id[:10]}...")
    if "AWS_SECRET_ACCESS_KEY" in os.environ:
        storage_options["secret"] = os.getenv("AWS_SECRET_ACCESS_KEY")
        logger.info("AWS_SECRET_ACCESS_KEY is set")

    logger.info(f"Connecting to S3 URL: {url}")
    fs, path = fsspec.url_to_fs(url, **storage_options)
    return fs, path


def _sync_serving_files(fs, model_s3_folder: Pathy, model_name: str) -> None:
    """
    Sync config.pbtxt and model.py from local serving folder to S3 base model folder.
    Called on every model upload to keep serving files up to date.
    """
    # config.pbtxt
    local_config = SERVING_FOLDER_PATH / "config.pbtxt"
    with open(local_config, "r") as f:
        content = f.read()
    if f'name: "{model_name}"' not in content:
        content = content.rstrip("\n") + f'\nname: "{model_name}"\n'
    with fs.open(str(model_s3_folder / "config.pbtxt"), "w") as f:
        f.write(content)
    logger.info("Synced config.pbtxt")

    # model.py
    local_model = SERVING_FOLDER_PATH / "model.py"
    with open(local_model, "rb") as f:
        data = f.read()
    with fs.open(str(model_s3_folder / "model.py"), "wb") as f:
        f.write(data)
    logger.info("Synced model.py")


def upload_model_files(
    base_s3_url: Pathy,
    model_version: str,
    model_name: str,
    model_data: Any,
) -> None:
    """
    Upload a new version of a model to the given base S3 URL.

    :param fs: fsspec filesystem object.
    :param model_version: The version of the model.
    :param model_name: The name of the model.
    :param model_data: In-memory bytes of the model file.
    """
    logger.info("Uploading model files to bucket...")

    fs, _ = get_s3_filesystem(str(base_s3_url))

    models_folder_path = base_s3_url / "models"

    # Check if the "models" folder exists
    if not fs.exists(str(models_folder_path / model_name)):
        logger.info("Models folder does not exist, creating structure...")
        create_initial_structure(base_s3_url, model_name)

    # Create the new version folder
    model_version_folder = models_folder_path / model_name / model_version
    if not fs.exists(str(model_version_folder)):
        fs.makedirs(str(model_version_folder))

    # Serialize to local temp file first, then upload to S3.
    # Streaming dill.dump directly into S3 can stall the connection
    # and cause Airflow to kill the pod due to log-stream timeout.
    model_pkl_path = model_version_folder / "model.pkl"
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=True) as tmp:
        logger.info("Serializing model to local temp file...")
        dill.dump(model_data, tmp)
        tmp_size = tmp.tell()
        logger.info(f"Serialized model size: {tmp_size / 1024 / 1024:.1f} MB")
        tmp.flush()

        logger.info(f"Uploading model.pkl to {model_pkl_path}...")
        fs.put(tmp.name, str(model_pkl_path))

    logger.info(f"Uploaded model.pkl for {model_name}, version {model_version}")

    # Sync config.pbtxt and model.py from local serving folder to base S3 folder
    model_base_folder = models_folder_path / model_name
    _sync_serving_files(fs, model_base_folder, model_name)

    # Copy model.py from base folder into the version folder (Triton needs it there)
    copy_model_py(model_base_folder, model_version_folder)
    logger.info(f"Copied model.py to version {model_version}")


def create_initial_structure(base_s3_url: Pathy, model_name: str) -> None:
    """
    Create the initial structure in the S3 bucket by copying config.pbtxt and model.py from the serving folder.

    :param base_s3_url: The base URL in S3 (e.g., s3://bucket-name or s3://bucket-name/folder).
    :param model_name: The name of the model.
    """
    fs, _ = get_s3_filesystem(str(base_s3_url))

    models_folder_path = base_s3_url / "models" / model_name

    try:
        fs.makedirs(str(models_folder_path), exist_ok=True)
        logger.info(f"Created/verified directory: {models_folder_path}")
    except Exception as e:
        logger.warning(f"Could not create directory {models_folder_path}: {e}")

    _sync_serving_files(fs, models_folder_path, model_name)
    logger.info("Initial structure created successfully.")


def copy_model_py(src_folder: Pathy, dest_folder: Pathy) -> None:
    """
    Copy the model.py file from the source folder to the destination folder.

    :param src_folder: The source folder path in S3.
    :param dest_folder: The destination folder path in S3.
    """
    # might be wrong if different filesystems, example: gs://bucket or s3://bucket
    fs, _ = get_s3_filesystem(str(src_folder))

    src_model_py_path = src_folder / "model.py"
    dest_model_py_path = dest_folder / "model.py"

    fs.copy(str(src_model_py_path), str(dest_model_py_path))


def clean_old_model_versions(base_s3_url: Pathy, model_name: str, num_to_keep: int) -> None:
    """
    Clean old model versions from the bucket, keeping only a specified number of recent versions.
    Only removes numeric version folders, skips runtime_envs and other non-version directories.

    :param base_s3_url: The base URL in S3 (e.g., s3://bucket-name).
    :param model_name: model name.
    :param num_to_keep: number of recent versions to keep.
    """
    logger.info(f"Cleaning old model versions from bucket (keeping {num_to_keep} latest)...")

    fs, _ = get_s3_filesystem(str(base_s3_url))

    s3_url = base_s3_url / "models" / model_name

    # List all items (files and directories) in the folder
    all_items = fs.ls(str(s3_url), detail=True)

    # Filter to get only directories with numeric names (model versions)
    all_folders = []
    for item in all_items:
        if item["type"] == "directory":
            folder_name = Path(item["name"]).name
            # Only include directories with numeric names (model versions)
            if folder_name.isdigit():
                all_folders.append(item["name"])

    if not all_folders:
        logger.info("No versions found to clean")
        return

    # Sort the folders by version (numeric)
    sorted_folders = sorted(all_folders, key=lambda x: int(Path(x).name), reverse=True)

    # Delete the older versions, keeping only the latest 'num_to_keep' versions
    folders_to_delete = sorted_folders[num_to_keep:]
    if folders_to_delete:
        logger.info(f"Deleting {len(folders_to_delete)} old versions...")
        for folder in folders_to_delete:
            fs.rm(folder, recursive=True)
            logger.info(f"Deleted: {Path(folder).name}")
    else:
        logger.info("No old versions to delete")


def copy_file(base_s3_url: Pathy, src_file_path: str, new_file_path: str) -> str:
    """
    Copy a file from one directory to another Bucket.

    :param base_s3_url: The base URL in S3 (e.g., s3://bucket-name).
    :param src_file_path: source file path (e.g., "config_files/project-env.tar.gz").
    :param new_file_path: new file path (e.g., "models/als_v2/project-env.tar.gz").

    :return: path to the copied file.
    """
    fs, _ = get_s3_filesystem(str(base_s3_url))

    src_url = base_s3_url / src_file_path
    dst_url = base_s3_url / new_file_path

    fs.copy(str(src_url), str(dst_url))

    return dst_url


def get_model_versions(base_s3_url: Pathy, model_name: str) -> list:
    """
    Get all numeric version folders for a model, sorted from newest to oldest.

    :param base_s3_url: The base URL in S3 (e.g., s3://bucket-name).
    :param model_name: The name of the model.
    :return: List of version folder paths (newest first).
    """
    fs, _ = get_s3_filesystem(str(base_s3_url))
    s3_url = base_s3_url / "models" / model_name

    # List all items (files and directories) in the model folder
    all_items = fs.ls(str(s3_url), detail=True)

    # Filter to get only directories
    all_folders = [item["name"] for item in all_items if item["type"] == "directory"]

    if not all_folders:
        return []

    # Filter only numeric version folders (skip runtime_envs, etc.)
    numeric_folders = []
    for folder in all_folders:
        folder_name = Path(folder).name
        if folder_name.isdigit():
            numeric_folders.append(folder)

    # Sort the folders by version (numeric, newest first)
    sorted_folders = sorted(numeric_folders, key=lambda x: int(Path(x).name), reverse=True)
    return sorted_folders


def load_model_s3(base_s3_url: Pathy, model_name: str) -> tuple:
    """
    Load the latest version of a model from the bucket.

    :param base_s3_url: The base URL in S3 (e.g., s3://bucket-name).
    :param model_name: The name of the model.
    :return: A tuple containing the loaded model and the model version.
    """
    logger.info(f"Loading the latest version of the model: {model_name}...")

    sorted_folders = get_model_versions(base_s3_url, model_name)

    if not sorted_folders:
        raise FileNotFoundError(f"No numeric version folders found in {base_s3_url}/models/{model_name}")

    latest_version_folder = sorted_folders[0]
    logger.info(f"Latest version folder: {Path(latest_version_folder).name}")

    # Load the model from the latest version folder
    fs, _ = get_s3_filesystem(str(base_s3_url))
    model_pkl_path = Pathy(latest_version_folder) / "model.pkl"

    with fs.open(str(model_pkl_path), "rb") as model_file:
        model = dill.load(model_file)

    # Extract the version from the folder name
    model_version = Path(latest_version_folder).name

    logger.info(f"Loaded model version: {model_version}")

    return model
