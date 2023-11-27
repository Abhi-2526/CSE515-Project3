import sqlite3
import numpy as np
from sklearn.decomposition import PCA
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import ResNet50_Weights
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.svm import SVC


def create_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_features (
            image_id INTEGER PRIMARY KEY,
            label TEXT,
            features BLOB
        )
    ''')
    conn.commit()
    conn.close()


def extract_features(dataset, model, device):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    features = []

    with torch.no_grad():
        for i, (input, target) in enumerate(tqdm(loader)):
            input = input.to(device)
            output = model(input)
            label = dataset.classes[target[0]] if hasattr(dataset, 'classes') else str(target[0].item())
            features.append((i, label, output.cpu().numpy().flatten()))

    return features


def store_features_in_db(features, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for image_id, label, feature in tqdm(features):
        cursor.execute('''
            INSERT INTO image_features (image_id, label, features) 
            VALUES (?, ?, ?)
        ''', (image_id, label, feature))

    conn.commit()
    conn.close()


def custom_pca(features, n_components=0.95):
    pca = PCA(n_components=n_components)
    pca.fit(features)
    return pca


def custom_metrics(predicted_labels, true_labels):
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='macro', zero_division=1)
    accuracy = accuracy_score(true_labels, predicted_labels)
    return precision, recall, f1_score, accuracy


def custom_dbscan(features, eps=0.5, min_samples=5):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    return clustering.labels_


def custom_mds(features, n_components=2):
    mds = MDS(n_components=n_components)
    return mds.fit_transform(features)


def train_and_predict_svm(X_train, y_train, X_test, kernel='linear'):
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)
    predictions = svm.decision_function(X_test)
    ranked_indices = np.argsort(-predictions)
    return ranked_indices


def plot_clusters_mds(mds_coordinates, labels):
    plt.figure(figsize=(10, 8))
    unique_labels = set(labels)
    for label in unique_labels:
        idx = labels == label
        plt.scatter(mds_coordinates[idx, 0], mds_coordinates[idx, 1], label=label)
    plt.legend()
    plt.title("Clusters in 2D MDS Space")
    plt.xlabel("MDS1")
    plt.ylabel("MDS2")
    plt.show()


def load_features_and_labels_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT features, label FROM image_features")
    data = cursor.fetchall()

    features = []
    labels = []
    for feature, label in data:
        features.append(np.frombuffer(feature, dtype=np.float32))  # Assuming features are stored as blobs
        labels.append(label)

    conn.close()
    return features, labels


def load_even_images_and_labels(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT features, label FROM image_features WHERE image_id % 2 = 0")
    data = cursor.fetchall()

    even_features = []
    even_labels = []
    for feature, label in data:
        even_features.append(np.frombuffer(feature, dtype=np.float32))
        even_labels.append(label)

    conn.close()
    return even_features, even_labels


def load_odd_images_and_labels(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT features, label FROM image_features WHERE image_id % 2 != 0")
    data = cursor.fetchall()

    odd_features = []
    odd_labels = []
    for feature, label in data:
        odd_features.append(np.frombuffer(feature, dtype=np.float32))
        odd_labels.append(label)

    conn.close()
    return odd_features, odd_labels


def run_extraction():
    db_path = 'caltech101_features.db'
    create_db(db_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.Caltech101('./data', download=True, transform=transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model = model.to(device)
    model.eval()

    features = extract_features(dataset, model, device)
    store_features_in_db(features, db_path)


if __name__ == '__main__':
    run_extraction()
