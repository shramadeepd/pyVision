import random
import os
import shutil
import requests
from dotenv import load_dotenv
import traceback
import zipfile
import yaml
import json
import socket
import time
import pathlib
from logger import logger
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

def split_dataset(image_files, label_files, split_ratios, seed=46):
    """Custom function to split the dataset manually without using sklearn"""
    try:
        assert sum(split_ratios) == 1, "Split ratios must sum up to 1."

        # Ensure reproducibility
        random.seed(seed)

        # Shuffle data
        combined = list(zip(image_files, label_files))
        random.shuffle(combined)
        image_files, label_files = zip(*combined)

        # Compute split indices
        total = len(image_files)
        train_end = int(split_ratios[0] * total)
        valid_end = train_end + int(split_ratios[1] * total)

        # Split into train, validation, and test sets
        train_images, train_labels = image_files[:train_end], label_files[:train_end]
        valid_images, valid_labels = image_files[train_end:valid_end], label_files[train_end:valid_end]
        test_images, test_labels = image_files[valid_end:], label_files[valid_end:]

        return train_images, train_labels, valid_images, valid_labels, test_images, test_labels
    except Exception as e:
        logger.error(f"Error in split_dataset: {e}", exc_info=True)
        raise

def check_classes():
    """Check the classes present in the dataset and their counts"""
    try:
        class_lengths = []
        for folder in os.listdir('_temp'):
            folder_path = os.path.join('_temp', folder)
            if os.path.isdir(folder_path):
                classes_file = os.path.join(folder_path, 'classes.txt')
                if os.path.exists(classes_file):
                    with open(classes_file, 'r') as f:
                        class_length = len(f.readlines())
                        class_lengths.append((folder, class_length))

        if not class_lengths:
            return True, "No classes.txt found in any folder."

        base_length = class_lengths[0][1]
        different_folders = []
        for folder, length in class_lengths[1:]:
            if length != base_length:
                different_folders.append((folder, length))

        if different_folders:
            return False, f"Different class lengths found in folders: {different_folders}"

        return True, "All folders have the same class length."
    except Exception as e:
        logger.error(f"Error in check_classes: {e}", exc_info=True)
        raise

def _take_samples():
    try:
        temp_folder = '_temp'
        subfolders = [f.path for f in os.scandir(temp_folder) if f.is_dir()]

        min_image_count = float('inf')
        image_counts = {}

        # Count the number of images in each subfolder
        for folder in subfolders:
            image_files = os.listdir(os.path.join(folder, 'images'))
            image_counts[folder] = image_files
            min_image_count = min(min_image_count, len(image_files))

        logger.info(f"Minimum image count: {min_image_count}")
        
        # Create combined data folder
        combined_data_folder = os.path.join(temp_folder, '../data')
        combined_images_folder = os.path.join(combined_data_folder, 'images')
        combined_labels_folder = os.path.join(combined_data_folder, 'labels')
        os.makedirs(combined_images_folder, exist_ok=True)
        os.makedirs(combined_labels_folder, exist_ok=True)

        # Take min_image_count samples from each subfolder and combine them
        for subfolder, image_files in image_counts.items():
            images_path = os.path.join(subfolder, 'images')
            labels_path = os.path.join(subfolder, 'labels')
            sampled_images = image_files[:min_image_count]

            for image_file in sampled_images:
                src_image_path = os.path.join(images_path, image_file)
                dst_image_path = os.path.join(combined_images_folder, image_file)
                if src_image_path != dst_image_path:
                    shutil.copy(src_image_path, dst_image_path)
                label_file = image_file.replace('.jpg', '.txt')  # Assuming label files have the same name as image files but with .txt extension
                src_label_path = os.path.join(labels_path, label_file)
                dst_label_path = os.path.join(combined_labels_folder, label_file)
                if src_label_path != dst_label_path:
                    shutil.copy(src_label_path, dst_label_path)

        # Copy classes.txt and notes.json if they exist
        for subfolder in subfolders:
            for file_name in ['classes.txt', 'notes.json']:
                src_file_path = os.path.join(subfolder, file_name)
                dst_file_path = os.path.join(combined_data_folder, file_name)
                if os.path.exists(src_file_path) and src_file_path != dst_file_path:
                    shutil.copy(src_file_path, dst_file_path)

        # Removes the temp folder after combining
        shutil.rmtree(temp_folder)
    except Exception as e:
        logger.error(f"Error in _take_samples: {e}", exc_info=True)
        raise

def move_files(file_list, source_dir, target_dir):
    """Move files to their respective directories"""
    try:
        os.makedirs(target_dir, exist_ok=True)
        for file_name in file_list:
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(target_dir, file_name))  # Using copy for testing
    except Exception as e:
        logger.error(f"Error in move_files: {e}", exc_info=True)
        raise

def _apply_aug():
    pass

def download_and_unzip(LS_ID, Type='YOLO'):
    def download_single(ls_id, target_folder):
        try:
            unzipped_folder_name = target_folder
            _HOST = os.getenv("HOST")
            _PORT = os.getenv("PORT")
            url = f"http://{_HOST}:{_PORT}/api/projects/{ls_id}/export?exportType={Type}"
            token = os.getenv("LS_TOKEN")
            headers = {"Authorization": f"Token {token}"}
            logger.info(f'Data download started for LS_ID {ls_id}.......')
            response = requests.get(url, headers=headers, stream=True)
            logger.info(f"Response status code: {response.status_code}")
            if response.status_code == 200:
                with zipfile.ZipFile(BytesIO(response.raw.read()), 'r') as zip_ref:
                    os.makedirs(unzipped_folder_name, exist_ok=True)
                    zip_ref.extractall(unzipped_folder_name)
                logger.info(f'Download ✅ for LS_ID {ls_id}')
            else:
                logger.error(f"Downloading failed from label studio for LS_ID {ls_id}", exc_info=True)
                # logger.error(f"Error in downloading data for LS_ID {ls_id}")
                return False
        except Exception as e:
            logger.error(f"Download and unzip failed for LS_ID {ls_id}: {e}", exc_info=True)
            return False
        return True

    try:
        failed_ls_ids = []
        if isinstance(LS_ID, list) and len(LS_ID) > 1: # Download multiple LS_IDs
            temp_folder = '_temp'
            os.makedirs(temp_folder, exist_ok=True)
            for ls_id in LS_ID:
                if not download_single(ls_id, os.path.join(temp_folder, str(ls_id))):
                    failed_ls_ids.append(ls_id)

            # Retry for failed LS_IDs up to 3 times
            for attempt in range(1, 4):
                if not failed_ls_ids:
                    break
                logger.info(f"Retry attempt {attempt} for failed LS_IDs: {failed_ls_ids}")
                new_failed_ls_ids = []
                for ls_id in failed_ls_ids:
                    if not download_single(ls_id, os.path.join(temp_folder, str(ls_id))):
                        new_failed_ls_ids.append(ls_id)
                failed_ls_ids = new_failed_ls_ids
            if failed_ls_ids:
                # logger.error(f"Failed to download after 3 attempts for LS_IDs: {failed_ls_ids}")
                logger.error(f"Failed to download after 3 attempts for LS_IDs: {failed_ls_ids}")
            if len(LS_ID) > 1:
                success, message = check_classes()
                if not success:
                    logger.error(message)
                    return
                _take_samples()
            return True
        else:
            if isinstance(LS_ID, list):
                LS_ID = LS_ID[0]
            download_single(LS_ID, 'data')
            return True
    except Exception as e:
        logger.error(f"Error in download_and_unzip: {e}", exc_info=True)
        raise

# download_and_unzip([309,306])
