import json
import os
from PIL import Image

# Define a function to convert bounding boxes to YOLO format
def convert_bbox_to_yolo(image_width, image_height, xmin, ymin, xmax, ymax):
    x_center = (xmin + xmax) / 2.0 / image_width
    y_center = (ymin + ymax) / 2.0 / image_height
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height
    return x_center, y_center, width, height

# Paths to the directories
labels_dir = './data/dataset/labels'
images_dir = './data/dataset/images'
output_dir = './data/dataset/yolo_labels'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate through each label file
for label_file in os.listdir(labels_dir):
    if label_file.endswith('.json'):
        label_path = os.path.join(labels_dir, label_file)
        image_name = label_file.replace('.json', '.jpg')  # Adjust extension if different
        image_path = os.path.join(images_dir, image_name)

        # Load JSON label file
        with open(label_path, 'r') as file:
            data = json.load(file)

        # Open image to get its dimensions
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        # Convert each bounding box and save to corresponding text file
        yolo_annotations = []
        for annotation in data:
            label = annotation['label']
            xmin = annotation['xmin']
            ymin = annotation['ymin']
            xmax = annotation['xmax']
            ymax = annotation['ymax']

            x_center, y_center, width, height = convert_bbox_to_yolo(image_width, image_height, xmin, ymin, xmax, ymax)
            yolo_annotations.append(f"{label} {x_center} {y_center} {width} {height}")

        # Write YOLO annotations to a text file
        yolo_label_path = os.path.join(output_dir, label_file.replace('.json', '.txt'))
        with open(yolo_label_path, 'w') as yolo_file:
            yolo_file.write('\n'.join(yolo_annotations))
