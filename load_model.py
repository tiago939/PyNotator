from ultralytics import YOLO
import os

class YOLOModelLoader:
    def __init__(self, model_base_dir='./runs/detect', model_path='yolov10n.pt'):
        self.model_base_dir = model_base_dir
        self.model_path = model_path

    def find_latest_model(self):
        subdirectories = [os.path.join(self.model_base_dir, d) for d in os.listdir(self.model_base_dir) if os.path.isdir(os.path.join(self.model_base_dir, d))]
        latest_model_path = None
        latest_model_time = 0
        
        for subdirectory in subdirectories:
            for root, _, files in os.walk(subdirectory):
                for file in files:
                    if file.endswith('.pt'):
                        file_path = os.path.join(root, file)
                        file_time = os.path.getctime(file_path)
                        if file_time > latest_model_time:
                            latest_model_path = file_path
                            latest_model_time = file_time
        
        if not latest_model_path:
            raise FileNotFoundError("No model files found in the specified directory.")
        
        return latest_model_path

    def load_model(self, use_custom=False):
        if use_custom:
            model_path = self.find_latest_model()
            print(f"Loading custom model from {model_path}")
        else:
            model_path = self.model_path
            print(f"Loading default model from {model_path}")
        
        model = YOLO(model_path)
        return model
