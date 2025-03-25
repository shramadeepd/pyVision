import os
import shutil
import logging
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOTrainer:
    def __init__(self, model, epochs, data_config="dataset_path.yaml", batch_size=8):
        self.model_path = model
        self.data_config = data_config
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self):
        """Load the YOLO model"""
        try:
            logger.info(f"Loading model from {self.model_path}...")
            self.model = YOLO(self.model_path)
            logger.info(f"Model loaded successfully on {self.device}.")
        except FileNotFoundError:
            logger.error(f"Model file not found at {self.model_path}.")
        except Exception as e:
            logger.error(f"An error occurred while loading the model: {e}")

    def train_model(self):
        """Train the YOLO model"""
        if self.model is None:
            raise ValueError("Model is not loaded. Please load the model first using `load_model()`.")
        
        logger.info(f"Training model for {self.epochs} epochs with batch size {self.batch_size}...")

        try:
            logger.info(self.data_config)
            results = self.model.train(data=self.data_config, epochs=self.epochs, batch=self.batch_size , optimizer='SGD',lr0=0.01,seed=42,momentum=0.9)
            logger.info("Training completed successfully.")
            return results
        except FileNotFoundError:
            logger.error(f"Data configuration file not found at {self.data_config}.")
        except ValueError as ve:
            logger.error(f"Value error: {ve}")
        except Exception as e:
            logger.error(f"An error occurred during training: {e}")
            return None

    def start_training(self):
        """Start the full process of loading and training the model"""
        self.load_model()
        return self.train_model()
