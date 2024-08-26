import os
import dill
from typing import Optional
import logging
import fsspec
import io
from pathy import Pathy
from pathlib import Path


CURRENT_DIR = Path(__file__).parent
SERVING_FOLDER_PATH = CURRENT_DIR.parent / 'serving_als'

logger = logging.getLogger("ALS Model saving stage:")


def upload_model_files(fs: fsspec.AbstractFileSystem, base_s3_url: Pathy, model_version: str, model_name: str, model_data: io.BytesIO) -> None:
    """
    Upload a new version of a model to the given base S3 URL.

    :param fs: fsspec filesystem object.
    :param base_s3_url: The base URL in S3 (e.g., s3://bucket-name).
    :param model_version: The version of the model.
    :param model_name: The name of the model.
    :param model_data: In-memory bytes of the model file.
    """
    logger.info("Uploading model files to bucket...")

    models_folder_path = base_s3_url / "models"
    
    # Check if the "models" folder exists
    if not fs.exists(models_folder_path):
        logger.info("Models folder does not exist, creating structure...")
        create_initial_structure(fs, base_s3_url, model_name)

    # Create the new version folder
    model_version_folder = models_folder_path / model_name / model_version
    if not fs.exists(model_version_folder):
        fs.makedirs(model_version_folder)

    # Upload the model.pkl directly from memory
    model_pkl_path = model_version_folder / "model.pkl"
    with fs.open(model_pkl_path, 'wb') as file:
        dill.dump(model_data, file)
    
    logger.info(f"Uploaded model.pkl for {model_name}, version {model_version}")

    # Copy the existing model.py into the new version folder
    copy_model_py(fs, models_folder_path / model_name, model_version_folder)
    logger.info(f"Copied model.py for {model_name}, version {model_version}")


def create_initial_structure(fs: fsspec.AbstractFileSystem, base_s3_url: Pathy,  model_name: str) -> None:
    """
    Create the initial structure in the S3 bucket by copying config.pbtxt and model.py from the serving folder.

    :param fs: fsspec filesystem object.
    :param base_s3_url: The base URL in S3 (e.g., s3://bucket-name or s3://bucket-name/folder).
    """
    models_folder_path = base_s3_url / "models" / model_name
    fs.makedirs(models_folder_path)

    # Copy config.pbtxt and model.py from the local serving folder
    serving_config_path = SERVING_FOLDER_PATH / "config.pbtxt"
    serving_model_path = SERVING_FOLDER_PATH / "model.py"
    serving_python_path = SERVING_FOLDER_PATH / "triton_python_backend_stub"

    config_s3_path = models_folder_path / "config.pbtxt"
    model_py_s3_path = models_folder_path / "model.py"
    python_s3_path = models_folder_path / "triton_python_backend_stub"

    logger.info("Creating initial structure with config.pbtxt and model.py...")

    with fs.open(config_s3_path, 'wb') as config_s3_file:
        with open(serving_config_path, 'rb') as local_config_file:
            config_s3_file.write(local_config_file.read())

    with fs.open(model_py_s3_path, 'wb') as model_s3_file:
        with open(serving_model_path, 'rb') as local_model_file:
            model_s3_file.write(local_model_file.read())

    with fs.open(python_s3_path, 'wb') as model_s3_file:
        with open(serving_python_path, 'rb') as local_model_file:
            model_s3_file.write(local_model_file.read())
            
    logger.info("Initial structure created successfully.")


def copy_model_py(fs: fsspec.AbstractFileSystem, src_folder: Pathy, dest_folder: Pathy) -> None:
    """
    Copy the model.py file from the source folder to the destination folder.

    :param fs: fsspec filesystem object.
    :param src_folder: The source folder path in S3.
    :param dest_folder: The destination folder path in S3.
    """
    src_model_py_path = src_folder / "model.py"
    dest_model_py_path = dest_folder / "model.py"
    
    fs.copy(str(src_model_py_path), str(dest_model_py_path))
    
    
def clean_old_model_versions(fs: fsspec.AbstractFileSystem, base_s3_url: Pathy, model_name: str, num_to_keep: int) -> None:
    """
    Clean old model versions from the bucket, keeping only a specified number of recent versions.

    :param fs: fsspec filesystem object.
    :param base_s3_url: The base URL in S3 (e.g., s3://bucket-name).
    :param model_name: model name.
    :param num_to_keep: number of recent versions to keep.
    """
    logger.info("Cleaning old model versions from bucket...")
    
    s3_url = base_s3_url / "models" / model_name
    
    # List all items (files and directories) in the folder
    all_items = fs.ls(s3_url, detail=True)
    
    # Filter to get only directories
    all_folders = [item['name'] for item in all_items if item['type'] == 'directory']
    
    # Sort the folders by version (assumed to be numeric)
    sorted_folders = sorted(all_folders, key=lambda x: int(Path(x).name), reverse=True)
    
    # Delete the older versions, keeping only the latest 'num_to_keep' versions
    for folder in sorted_folders[num_to_keep:]:
        fs.rm(folder, recursive=True)
        logger.info(f"Deleted folder: {folder}")
        
        
def edit_config_pbtxt(fs: fsspec.AbstractFileSystem, base_s3_url: Pathy, model_name: str) -> None:
    """
    Update the base config.pbtxt in the specified directory.
    
    :param fs: fsspec filesystem object.
    :param base_s3_url: The base URL in S3 (e.g., s3://bucket-name).
    :param model_name: new model name to be updated in the config.
    """
    s3_url = base_s3_url / "models" / model_name / "config.pbtxt"
    
    # Read the existing config file
    with fs.open(s3_url, 'r') as f:
        file_content = f.read()

    # Modify the last line of the config file
    lines = file_content.split("\n")
    lines[-1] = f'name: "{model_name}"'
    new_file_content = "\n".join(lines)

    # Write the updated config back to S3
    with fs.open(s3_url, 'w') as f:
        f.write(new_file_content)


def copy_file(fs: fsspec.AbstractFileSystem, base_s3_url: Pathy, src_file_path: str, new_file_path: str) -> str:
    """
    Copy a file from one directory to another Bucket.

    :param fs: fsspec filesystem object.
    :param base_s3_url: The base URL in S3 (e.g., s3://bucket-name).
    :param src_file_path: source file path (e.g., "config_files/project-env.tar.gz").
    :param new_file_path: new file path (e.g., "models/als_v2/project-env.tar.gz").

    :return: path to the copied file.
    """
    src_url = base_s3_url / src_file_path
    dst_url = base_s3_url / new_file_path
    
    fs.copy(src_url, dst_url)
    
    return dst_url