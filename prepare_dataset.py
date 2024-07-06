import os
import shutil
import random

# Paths to the directories
images_dir = './data/dataset/images'
labels_dir = './data/dataset/yolo_labels'

# Output directories
output_base_dir = './data/yolo_dataset'
train_images_dir = os.path.join(output_base_dir, 'images', 'train')
val_images_dir = os.path.join(output_base_dir, 'images', 'val')
test_images_dir = os.path.join(output_base_dir, 'images', 'test')
train_labels_dir = os.path.join(output_base_dir, 'labels', 'train')
val_labels_dir = os.path.join(output_base_dir, 'labels', 'val')
test_labels_dir = os.path.join(output_base_dir, 'labels', 'test')

# Create output directories if they don't exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

# Get list of images and corresponding label files
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
label_files = [f.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt') for f in image_files]

# Combine them into pairs and shuffle
dataset = list(zip(image_files, label_files))
random.shuffle(dataset)

# Calculate split sizes
total_count = len(dataset)
train_count = int(0.70 * total_count)
val_count = int(0.15 * total_count)

# Split the dataset
train_set = dataset[:train_count]
val_set = dataset[train_count:train_count + val_count]
test_set = dataset[train_count + val_count:]

# Function to copy files
def copy_files(file_set, image_dir, label_dir):
    for image_file, label_file in file_set:
        shutil.copy(os.path.join(images_dir, image_file), os.path.join(image_dir, image_file))
        shutil.copy(os.path.join(labels_dir, label_file), os.path.join(label_dir, label_file))

# Copy files to respective directories
copy_files(train_set, train_images_dir, train_labels_dir)
copy_files(val_set, val_images_dir, val_labels_dir)
copy_files(test_set, test_images_dir, test_labels_dir)

print(f"Dataset split into {len(train_set)} training, {len(val_set)} validation, and {len(test_set)} testing samples.")
