import os , yaml
from logger import logger
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import torch


class DatasetYamlWriter:
    def __init__(self,output_file="dataset_path.yaml"):
        # Fixed paths for train, validation, and test datasets
        self.train_path = "datasets/train/images"
        self.val_path = "datasets/valid/images"
        self.test_path = "datasets/test/images"
        self.file_path = f"data/classes.txt"
        # Default class names and number of classes
        # self.num_classes = num_classes
        # # self.class_names = [
        # #     "bicycles", "buses", "chimneys", "crosswalks", "fire hydrants", "motorcycles",
        # #     "parking meters", "stairs", "taxis", "tractors", "traffic lights", "vehicles"
        # # ]
        # self.class_names = class_names
        
        self.output_file = output_file

    def generate_yaml_content(self):
        with open(self.file_path, 'r') as file:
            file_contents = file.readlines()
            nc = len(file_contents)
            config_data = {
            'train': f'{os.getcwdb().decode("utf-8")}/datasets/train/images',
            'val': f'{os.getcwdb().decode("utf-8")}/datasets/valid/images',
            "nc": nc,
            "names": [cls.strip() for cls in file_contents]  # List of class names
            }
            yaml_file_path = self.output_file
            with open(yaml_file_path, 'w') as yaml_file:
                print(config_data)
                yaml.dump(config_data, yaml_file, default_flow_style=False)
        logger.info('Config file ✅')
    def write_yaml(self):
        """Write the YAML content to the output file"""
        self.generate_yaml_content()
        
        # # Write the YAML content to the output file
        # with open(self.output_file, "w") as file:
        #     file.write(yaml_content)
        print(f"YAML file '{self.output_file}' written successfully.")
