import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from utils import load_odd_images_and_labels, load_even_images_and_labels


def compute_latent_semantics(data, k=10):
    pca = PCA(n_components=k)
    pca.fit(data)
    return pca.components_


def predict_label(test_data, latent_semantics, label):
    distances = [np.linalg.norm(test_data - ls) for ls in latent_semantics]
    return label, min(distances)


def main():
    k = int(input("Write the number of latent semantics needed: "))
    db_path = 'caltech101_features.db'
    even_images, even_labels = load_even_images_and_labels(db_path)
    odd_images, odd_labels = load_odd_images_and_labels(db_path)

    even_features = [image.flatten() for image in even_images]
    odd_features = [image.flatten() for image in odd_images]

    unique_labels = sorted(set(even_labels))  # Sort the labels
    label_latent_semantics = {}
    for label in unique_labels:
        label_data = [even_features[i] for i in range(len(even_features)) if even_labels[i] == label]
        label_latent_semantics[label] = compute_latent_semantics(label_data, k)

    predicted_labels = []
    for feature in odd_features:
        label_predictions = [predict_label(feature, label_latent_semantics[label], label) for label in unique_labels]
        predicted_label = min(label_predictions, key=lambda x: x[1])[0]
        predicted_labels.append(predicted_label)

    precision, recall, f1_score, _ = precision_recall_fscore_support(odd_labels, predicted_labels, average=None, labels=unique_labels)
    overall_accuracy = accuracy_score(odd_labels, predicted_labels)

    for label, semantics in label_latent_semantics.items():
        print(f"Label: {label}, Latent Semantics: {semantics}")

    print("\n")
    for label, p, r, f1 in zip(unique_labels, precision, recall, f1_score):
        print(f"Label: {label}, Precision: {p}, Recall: {r}, F1-Score: {f1}")

    print("\n")
    for i, predicted_label in enumerate(predicted_labels):
        print(f"Odd image {i + 1} - Predicted Label ID - {predicted_label}")

    print(f"\nOverall Accuracy: {overall_accuracy}")


if __name__ == "__main__":
    main()
