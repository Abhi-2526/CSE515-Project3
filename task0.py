import warnings
from collections import defaultdict
import torch
import json
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import Caltech101
from skimage.feature import hog
from skimage.color import rgb2gray
import numpy as np
from PIL import Image
import pandas
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tensorly.decomposition import parafac
import sqlite3
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew
from skimage.color import rgb2gray
import torch
import os, sys
from tqdm import tqdm
import networkx as nx
import datetime
import types
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


warnings.simplefilter(action='ignore', category=Warning)

# Initialize the ResNet50 model
resnet50 = models.resnet50(pretrained=True)
resnet50.eval()

conn = sqlite3.connect('image_features.db')
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS features (
    imageID INTEGER PRIMARY KEY,
    label TEXT,
    ColorMoments BLOB,
    HOG BLOB,
    AvgPool BLOB,
    Layer3 BLOB,
    FCLayer BLOB,
    RESNET BLOB
)
''')


def store_in_database(imageID, features):
    # Convert numpy arrays to bytes for storage
    n0_usage, label = dataset[imageID]
    # print("Size of ColorMoments before storing:", features['ColorMoments'].shape)
    ColorMoments_bytes = features['ColorMoments'].astype(np.float32).tobytes()
    # print("Size of ColorMoments before storing:", len(ColorMoments_bytes))

    HOG_bytes = features['HOG'].astype(np.float32).tobytes()

    # Reduce dimensionality of ResNet-AvgPool and ResNet-Layer3 features
    ResNetAvgPool1024 = np.array(features['AvgPool'])
    AvgPool_bytes = ResNetAvgPool1024.tobytes()
    ResNetLayer31024 = np.array(features['Layer3'])
    Layer3_bytes = ResNetLayer31024.tobytes()
    FCLayer_bytes = features['FCLayer'].tobytes()
    ResNetOutput_bytes = features['RESNET'].tobytes()

    # Check if imageID already exists in the database
    cursor.execute("SELECT 1 FROM features WHERE imageID=?", (imageID,))
    exists = cursor.fetchone()

    if not exists:
        # Insert a new record if imageID doesn't exist
        cursor.execute('''
        INSERT INTO features (imageID, label, ColorMoments, HOG, AvgPool, Layer3, FCLayer, RESNET)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            imageID, label, ColorMoments_bytes, HOG_bytes, AvgPool_bytes, Layer3_bytes, FCLayer_bytes, ResNetOutput_bytes))

        conn.commit()


def load_features_from_database():
    cursor.execute("SELECT imageID, label, ColorMoments, HOG, AvgPool, Layer3, FCLayer, RESNET FROM features")
    rows = cursor.fetchall()
    for row in rows:
        database.append({
            "imageID": row[0],
            "label": row[1],
            "features": {
                "ColorMoments": np.frombuffer(row[2], dtype=np.float32),
                "HOG": np.frombuffer(row[3], dtype=np.float32),
                "AvgPool": np.frombuffer(row[4], dtype=np.float32),
                "Layer3": np.frombuffer(row[5], dtype=np.float32),
                "FCLayer": np.frombuffer(row[6], dtype=np.float32),
                "RESNET": np.frombuffer(row[7], dtype=np.float32)
            }
        })


# Load Caltech101 dataset
dataset = Caltech101(root="./data", download=True)

# Extract class names from the Caltech101 dataset
label_name_to_idx = {name: idx for idx, name in enumerate(dataset.categories)}

# Database simulation (in reality, you'd use an actual database)
database = []


def custom_transform(image):
    # Check if the image is grayscale
    if len(image.getbands()) == 1:
        # Convert grayscale to RGB by repeating the channel 3 times
        image = Image.merge("RGB", (image, image, image))
    return image


custom_transforms = transforms.Lambda(lambda x: custom_transform(x))

# Define transformations for the image
transform = transforms.Compose([
    custom_transforms,  # Use the custom transforms here
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def compute_cell_moments(image_np, start_row, end_row, start_col, end_col):
    cell_values = image_np[start_row:end_row, start_col:end_col]
    total_pixels = cell_values.shape[0] * cell_values.shape[1]

    # Compute mean for each channel
    mean = np.sum(cell_values, axis=(0, 1)) / total_pixels

    # Compute variance for each channel
    variance = np.sum((cell_values - mean) ** 2, axis=(0, 1)) / total_pixels

    # Compute standard deviation for each channel
    std_dev = np.sqrt(variance)

    return mean, std_dev, variance


def compute_color_moments_matrix(image_np):
    if len(image_np.shape) == 2:  # Grayscale image
        image_np = np.stack([image_np, image_np, image_np], axis=-1)  # Convert to RGB by repeating the channel 3 times

    height, width, channels = image_np.shape
    cell_height, cell_width = height // 10, width // 10

    mean_matrix = np.zeros((10, 10, channels))
    std_dev_matrix = np.zeros((10, 10, channels))
    variance_matrix = np.zeros((10, 10, channels))

    for i in range(10):
        for j in range(10):
            start_row, end_row = i * cell_height, (i + 1) * cell_height
            start_col, end_col = j * cell_width, (j + 1) * cell_width
            mean, std_dev, variance = compute_cell_moments(image_np, start_row, end_row, start_col, end_col)
            mean_matrix[i, j] = mean
            std_dev_matrix[i, j] = std_dev
            variance_matrix[i, j] = variance

    return mean_matrix, std_dev_matrix, variance_matrix


def extract_features(image):
    # Extract color moments manually
    image_np_resized = np.array(image.resize((100, 100)))
    mean_matrix, std_dev_matrix, variance_matrix = compute_color_moments_matrix(image_np_resized)
    color_moments = np.concatenate([mean_matrix, std_dev_matrix, variance_matrix], axis=2).flatten()
    # print("ColorMoments shape:", color_moments.shape)

    # Convert image to grayscale for HOG
    image_tensor = transform(image)  # Use the main transform here
    gray_image = rgb2gray(image_tensor.permute(1, 2, 0).numpy())

    # Extract HOG features
    hog_features = hog(gray_image, pixels_per_cell=(16, 16), cells_per_block=(4, 4), visualize=False)
    hog_features = hog_features.flatten()[:900]

    outputs = {}

    def hook(module, input, output):
        outputs[module._get_name()] = output

    # Attach hooks to the desired layers
    hook_handles = []
    hook_handles.append(resnet50.avgpool.register_forward_hook(hook))
    hook_handles.append(resnet50.layer3.register_forward_hook(hook))
    hook_handles.append(resnet50.fc.register_forward_hook(hook))

    # Extract features using RESNET50
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        resnet_output = resnet50(image_tensor)

    # Remove hooks
    for handle in hook_handles:
        handle.remove()

    avgpool_output = outputs['AdaptiveAvgPool2d'].squeeze().numpy()
    avgpool_1024 = (avgpool_output[::2] + avgpool_output[1::2]) / 2

    layer3_output = outputs['Sequential'].squeeze().numpy()
    layer3_output_flattened = layer3_output.reshape(-1)
    stride = len(layer3_output_flattened) // 1024
    layer3_1024 = [np.mean(layer3_output_flattened[i:i + stride]) for i in
                   range(0, len(layer3_output_flattened), stride)]

    fc_1000 = outputs['Linear'].squeeze().numpy()
    resnet = resnet_output.squeeze().numpy()

    return {
        "ColorMoments": color_moments,
        "HOG": hog_features,
        "AvgPool": avgpool_1024,
        "Layer3": layer3_1024,
        "FCLayer": fc_1000,
        "RESNET": resnet
    }


def task_0a(feature_name="ColorMoments"):
    feature_selection = ['ColorMoments', 'HOG', 'AvgPool', 'Layer3', 'FCLayer', 'RESNET']
    # id_list = []
    feature_data = {}
    if feature_name in feature_selection:
        for i, (image, label) in tqdm(enumerate(dataset)):
            # print(label)
            # Only process even-numbered images
            if i % 2 == 0:
                features = extract_features(image)
                # Store features in the database
                # store_in_database(i, features)
                # id_list.append(i)
                feature_data[i]=features[feature_name]
                # feature_array.append(features[feature_name])
    else:
        print("wrong name!")
    

    # Standardize the features
    features_array = list(feature_data.values()) # [[f11, f12, f13, ...], 
                                                 #  [f21, f22, f23, ...]]
    scaler = StandardScaler()
    features_array = scaler.fit_transform(features_array)

    # PCA process
    pca = PCA()
    pca.fit(features_array)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    threshold = 0.95
    inherent_dimensionality = np.argmax(cumulative_variance >= threshold) + 1

    print(f"Inherent dimensionality: {inherent_dimensionality}")
    # Inherent dimensionality: 158 for ColorMoments


def task_0b(feature_name="ColorMoments"):

    features_by_label = {}

    feature_selection = ['ColorMoments', 'HOG', 'AvgPool', 'Layer3', 'FCLayer', 'RESNET']
    # id_list = []
    feature_data = {}
    if feature_name in feature_selection:
        for i, (image, label) in tqdm(enumerate(dataset)):
            # Only process even-numbered images
            if i % 2 == 0:
                # print(label)
                features = extract_features(image)
                # Store features in the database
                # store_in_database(i, features)
                # id_list.append(i)
                # feature_data[i]=features[feature_name]
                # group by label
                if label not in features_by_label:
                    features_by_label[label] = []

                features_by_label[label].append(features[feature_name])
                # feature_array.append(features[feature_name])
    else:
        print("wrong name!")
    
    inherent_dimensionalities = {}

    for label, features in features_by_label.items():
        # Convert the list of features to a numpy array
        features_array = np.array(features)

        # Perform PCA on the features
        pca = PCA()
        pca.fit(features_array)

        # Calculate the cumulative sum of explained variance ratio
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        # Determine the number of components that explain 95% of the variance
        threshold = 0.95
        inherent_dimensionality = np.argmax(cumulative_variance >= threshold) + 1
        inherent_dimensionalities[label] = inherent_dimensionality

        # Print the inherent dimensionality for the label
        print(f"Label: {label}, Inherent Dimensionality: {inherent_dimensionality}")
    
    





# task_0a("ColorMoments")
task_0b("ColorMoments")