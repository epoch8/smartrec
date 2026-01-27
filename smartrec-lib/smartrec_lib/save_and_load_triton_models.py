import logging
import os
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
    if not fs.exists(models_folder_path / model_name):
        logger.info("Models folder does not exist, creating structure...")
        create_initial_structure(base_s3_url, model_name)

    # Create the new version folder
    model_version_folder = models_folder_path / model_name / model_version
    if not fs.exists(model_version_folder):
        fs.makedirs(model_version_folder)

    # Upload the model.pkl directly from memory
    model_pkl_path = model_version_folder / "model.pkl"
    with fs.open(model_pkl_path, "wb") as file:
        dill.dump(model_data, file)

    logger.info(f"Uploaded model.pkl for {model_name}, version {model_version}")

    # Copy the existing model.py into the new version folder
    copy_model_py(models_folder_path / model_name, model_version_folder)
    logger.info(f"Copied model.py for {model_name}, version {model_version}")


def create_initial_structure(base_s3_url: Pathy, model_name: str) -> None:
    """
    Create the initial structure in the S3 bucket by copying config.pbtxt and model.py from the serving folder.

    :param base_s3_url: The base URL in S3 (e.g., s3://bucket-name or s3://bucket-name/folder).
    :param model_name: The name of the model.
    """
    fs, _ = get_s3_filesystem(str(base_s3_url))

    models_folder_path = base_s3_url / "models" / model_name

    # Try to create directory structure (will skip if exists)
    try:
        fs.makedirs(str(models_folder_path), exist_ok=True)
        logger.info(f"Created/verified directory: {models_folder_path}")
    except Exception as e:
        logger.warning(f"Could not create directory {models_folder_path}: {e}")
        # Continue anyway - directory might already exist

    # Copy config.pbtxt and model.py from the local serving folder
    serving_config_path = SERVING_FOLDER_PATH / "config.pbtxt"
    serving_model_path = SERVING_FOLDER_PATH / "model.py"

    serving_config_s3_path = models_folder_path / "config.pbtxt"
    serving_model_py_s3_path = models_folder_path / "model.py"

    logger.info("Creating initial structure with config.pbtxt and model.py...")

    with fs.open(serving_config_s3_path, "wb") as config_s3_file:
        with open(serving_config_path, "rb") as local_config_file:
            config_s3_file.write(local_config_file.read())

    logger.info("Uploaded config.pbtxt")

    with fs.open(serving_model_py_s3_path, "wb") as model_s3_file:
        with open(serving_model_path, "rb") as local_model_file:
            model_s3_file.write(local_model_file.read())

    logger.info("Uploaded model.py")

    edit_config_pbtxt(base_s3_url=base_s3_url, model_name=model_name)

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

    :param base_s3_url: The base URL in S3 (e.g., s3://bucket-name).
    :param model_name: model name.
    :param num_to_keep: number of recent versions to keep.
    """
    logger.info("Cleaning old model versions from bucket...")

    fs, _ = get_s3_filesystem(str(base_s3_url))

    s3_url = base_s3_url / "models" / model_name

    # List all items (files and directories) in the folder
    all_items = fs.ls(s3_url, detail=True)

    # Filter to get only directories with numeric names (model versions)
    all_folders = []
    for item in all_items:
        if item["type"] == "directory":
            folder_name = Path(item["name"]).name
            # Only include directories with numeric names (model versions)
            if folder_name.isdigit():
                all_folders.append(item["name"])

    # Sort the folders by version (numeric)
    sorted_folders = sorted(all_folders, key=lambda x: int(Path(x).name), reverse=True)

    # Delete the older versions, keeping only the latest 'num_to_keep' versions
    for folder in sorted_folders[num_to_keep:]:
        fs.rm(folder, recursive=True)
        logger.info(f"Deleted folder: {folder}")


def edit_config_pbtxt(base_s3_url: Pathy, model_name: str) -> None:
    """
    Update the base config.pbtxt in the specified directory.

    :param base_s3_url: The base URL in S3 (e.g., s3://bucket-name).
    :param model_name: new model name to be updated in the config.
    """
    fs, _ = get_s3_filesystem(str(base_s3_url))

    s3_url = base_s3_url / "models" / model_name / "config.pbtxt"

    # Read the existing config file
    with fs.open(s3_url, "r") as f:
        file_content = f.read()

    # Modify the last line of the config file
    lines = file_content.split("\n")
    lines[-1] = f'name: "{model_name}"'
    new_file_content = "\n".join(lines)

    # Write the updated config back to S3
    with fs.open(s3_url, "w") as f:
        f.write(new_file_content)


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

    fs.copy(src_url, dst_url)

    return dst_url


def load_model_s3(base_s3_url: Pathy, model_name: str) -> tuple:
    """
    Load the latest version of a model from the bucket.

    :param base_s3_url: The base URL in S3 (e.g., s3://bucket-name).
    :param model_name: The name of the model.
    :return: A tuple containing the loaded model and the model version.
    """
    logger.info(f"Loading the latest version of the model: {model_name}...")

    fs, _ = get_s3_filesystem(str(base_s3_url))

    s3_url = base_s3_url / "models" / model_name

    # List all items (files and directories) in the model folder
    all_items = fs.ls(s3_url, detail=True)

    # Filter to get only directories
    all_folders = [item["name"] for item in all_items if item["type"] == "directory"]

    if not all_folders:
        raise FileNotFoundError(f"No model versions found in {s3_url}")

    # Sort the folders by version (assumed to be numeric)
    sorted_folders = sorted(all_folders, key=lambda x: int(Path(x).name), reverse=True)
    latest_version_folder = sorted_folders[0]
    logger.info(f"Latest version folder: {latest_version_folder}")

    # Load the model from the latest version folder
    model_pkl_path = Pathy(latest_version_folder) / "model.pkl"

    with fs.open(model_pkl_path, "rb") as model_file:
        model = dill.load(model_file)

    # Extract the version from the folder name
    model_version = Path(latest_version_folder).name

    logger.info(f"Loaded model version: {model_version}")

    return model
