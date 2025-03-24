import os
import shutil
from utils import split_dataset, move_files

class DatasetSplitter:
    def __init__(self, extracted_folder_path, split_ratios=(0.95, 0.025, 0.025), output_dir='datasets'):
        self.extracted_folder_path = extracted_folder_path
        self.output_dir = output_dir
        self.split_ratios = split_ratios

        assert sum(self.split_ratios) == 1, "Split ratios must sum up to 1."

        # Directories for images and labels
        self.image_dir = os.path.join(self.extracted_folder_path, 'images')
        self.label_dir = os.path.join(self.extracted_folder_path, 'labels')

        if not os.path.exists(self.image_dir) or not os.path.exists(self.label_dir):
            raise FileNotFoundError("Image or label directory not found.")

        # Create output directories for train, valid, test sets
        for split in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(self.output_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, split, 'labels'), exist_ok=True)

    def get_image_label_files(self):
        """Get sorted lists of image and label files"""
        image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png','.jpeg'))])
        label_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith('.txt')])

        if len(image_files) != len(label_files):
            raise ValueError("Mismatch between the number of images and labels.")

        assert all(os.path.splitext(img)[0] == os.path.splitext(lbl)[0] for img, lbl in zip(image_files, label_files)), \
            "Filenames of images and labels do not match."

        return image_files, label_files

    def organize_data(self):
        """Organize and move the dataset to the output directory"""
        image_files, label_files = self.get_image_label_files()

        # Use the custom dataset splitting function
        train_images, train_labels, valid_images, valid_labels, test_images, test_labels = split_dataset(
            image_files, label_files, self.split_ratios
        )

        # Move files to their respective directories
        move_files(train_images, self.image_dir, os.path.join(self.output_dir, 'train', 'images'))
        move_files(train_labels, self.label_dir, os.path.join(self.output_dir, 'train', 'labels'))
        move_files(valid_images, self.image_dir, os.path.join(self.output_dir, 'valid', 'images'))
        move_files(valid_labels, self.label_dir, os.path.join(self.output_dir, 'valid', 'labels'))
        move_files(test_images, self.image_dir, os.path.join(self.output_dir, 'test', 'images'))
        move_files(test_labels, self.label_dir, os.path.join(self.output_dir, 'test', 'labels'))

        shutil.rmtree(self.extracted_folder_path, ignore_errors=True)
        print("Dataset split and organized successfully.")
