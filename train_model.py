from ultralytics import YOLO
import yaml
import cv2
import threading
import numpy as np

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Function to train the YOLO model
def train_yolo_model(model, yaml_path='./data/data.yaml', epochs=128, model_path='yolov10n.pt'):
    # Load the YAML configuration
    data_config = load_yaml(yaml_path)

    # Print dataset details
    print(f"Training dataset: {data_config['train']}")
    print(f"Validation dataset: {data_config['val']}")
    print(f"Number of classes: {data_config['nc']}")
    print(f"Class names: {data_config['names']}")

    # Train the model
    model.train(data=yaml_path, epochs=epochs)
    print("Model training completed.")
