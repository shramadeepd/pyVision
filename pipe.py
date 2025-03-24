import os
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import torch
from datasetsplitter import DatasetSplitter
from datasetyaml import DatasetYamlWriter
from trainer import YOLOTrainer
from utils import download_and_unzip
import logging

class FullPipeline:
    def __init__(self, model="yolo11n.pt", epochs=500, batch_size=8, LS=None):
        self.extracted_folder_path = 'data'
        self.data_config = "dataset_path.yaml"
        self.output_dir = "datasets"
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.LS = LS

        # if self.LS is None:
        #     logger.warning("LS_ID not provided. Skipping dataset download and split.")



    def run(self):
        # Check if datasets folder already exists
        if os.path.exists(self.output_dir):
            logging.info(f"'{self.output_dir}' already exists. Skipping dataset split.")
            if not os.path.exists(self.data_config):
                logging.error("Please create the config file")
            else:
                logging.info("Config file exists")
        else:
            # Split dataset
            # if isinstance(self.LS, list) and (self.LS is None or len(self.LS) == 0):
            #     logging.error("LS_ID not provided. Skipping dataset download and split.")
            #     return
            if self.LS is not None and (isinstance(self.LS, list) or isinstance(self.LS, int)):
                if not download_and_unzip(self.LS):
                    return 
                yaml_writer = DatasetYamlWriter()
                yaml_writer.write_yaml()
                splitter = DatasetSplitter(self.extracted_folder_path, output_dir=self.output_dir)
                splitter.organize_data()
            else:
                logging.warning("LS_ID not provided correctly. Skipping dataset download and split.")
                return

        # Train the model
        trainer = YOLOTrainer(self.model, data_config=self.data_config, epochs=self.epochs, batch_size=self.batch_size)
        trainer.start_training()

