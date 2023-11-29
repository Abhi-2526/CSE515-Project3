import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import Caltech101
import utils
import sys
from sklearn.metrics.pairwise import cosine_similarity


def probabilistic_relevance_feedback(query_vector, even_vectors, even_labels, feedback_data, query_label, t):
    print("Starting Probabilistic Relevance Feedback...")

    relevant_vectors = [query_vector]
    non_relevant_vectors = []

    for idx, label in enumerate(even_labels):
        if label == query_label and idx != query_image_id:
            relevant_vectors.append(even_vectors[idx])

    for idx, label, rating in feedback_data:
        if rating in ['r+']:
            relevant_vectors.append(even_vectors[idx])
        else:
            non_relevant_vectors.append(even_vectors[idx])

    relevant_centroid = np.mean(relevant_vectors, axis=0) if relevant_vectors else query_vector
    non_relevant_centroid = np.mean(non_relevant_vectors, axis=0) if non_relevant_vectors else np.zeros_like(query_vector)

    print("Scoring each image based on its similarity to relevant and non-relevant centroids...")
    scores = []
    for idx, vector in enumerate(even_vectors):
        sim_with_relevant = cosine_similarity([vector], [relevant_centroid])[0][0]
        sim_with_non_relevant = cosine_similarity([vector], [non_relevant_centroid])[0][0]
        score = sim_with_relevant - sim_with_non_relevant
        scores.append((idx, score))

    print("Sorting images based on their scores...")
    scores.sort(key=lambda x: x[1], reverse=True)
    top_t_indices = [idx for idx, _ in scores[:t+5]]

    return top_t_indices


def load_feedback_data(feedback_file):
    feedback_data = []
    with open(feedback_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                idx, label, rating = int(parts[0]), int(parts[1]), parts[2]
                feedback_data.append((idx, label, rating))
    return feedback_data


def train_svm(even_vectors, even_labels, feedback_data, query_label):
    X_train = []
    y_train = []

    feedback_indices = {idx for idx, _, _ in feedback_data}
    query_label_str = str(query_label)

    for idx, vector in enumerate(even_vectors):
        label = str(even_labels[idx])
        if idx in feedback_indices:
            if (idx, label, 'r+') in feedback_data:
                y_train.append(2)
            elif (idx, label, 'r') in feedback_data:
                y_train.append(1)
            elif (idx, label, 'i-') in feedback_data:
                y_train.append(-2)
            else:
                y_train.append(-1)
            X_train.append(vector)
        else:
            y_train.append(1 if label == query_label_str else -1)
            X_train.append(vector)

    top_t_indices = utils.train_and_predict_svm(X_train, y_train, even_vectors)
    return top_t_indices[:t]


def visualize_images(query_image_id, top_t_indices, dataset, feedback_data, t):
    feedback_positive_indices = {idx for idx, _, rating in feedback_data if rating in ['r+']}

    final_indices = []

    for idx in feedback_positive_indices:
        if len(final_indices) >= t + 1:
            break
        if idx != query_image_id:
            final_indices.append(idx)

    for idx in top_t_indices:
        if len(final_indices) >= t + 1:
            break
        if idx != query_image_id and idx not in final_indices:
            final_indices.append(idx)

    print(f"Top {t} Image Matches: {final_indices[:t]}")

    num_images = len(final_indices)
    fig, axes = plt.subplots(1, num_images, figsize=(20, 10))

    query_image = dataset[query_image_id][0]
    axes[0].imshow(query_image)
    axes[0].axis('off')
    axes[0].set_title(f'Query Id: {query_image_id}')

    for ax, idx in zip(axes[1:], final_indices):
        image = dataset[idx][0]
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f'ID: {idx}')

    plt.show()


if __name__ == "__main__":
    dataset = Caltech101(root="./data", download=True)
    db_path = 'caltech101_features.db'
    even_vectors, even_labels = utils.load_features_and_labels_from_db(db_path)

    query_image_id = int(sys.argv[1])
    query_label = int(sys.argv[2])
    t = int(sys.argv[3])

    feedback_data = load_feedback_data('feedback_data.txt')
    while True:
        choice = input("1) SVM\n2) Probabilistic\nChoose feedback model (1/2): ").strip()
        if choice == '1':
            top_t_indices = train_svm(even_vectors, even_labels, feedback_data, query_label)
            if top_t_indices.size > 0:
                visualize_images(query_image_id, top_t_indices, dataset, feedback_data, t)
            else:
                print("SVM training was not successful.")

        elif choice == '2':
            query_vector = even_vectors[query_image_id]
            top_t_indices = probabilistic_relevance_feedback(query_vector, even_vectors, even_labels, feedback_data, query_label, t)
            if top_t_indices:
                # print(f"Top {t} Image Matches after Probabilistic Relevance Feedback: {top_t_indices}")
                visualize_images(query_image_id, top_t_indices, dataset, feedback_data, t)
            else:
                print("Probabilistic Relevance Feedback was not successful.")
        else:
            print("Invalid choice. Exiting.")

        ch = input("Repeat with another model? (Yes/no): ")
        if ch.lower() in ['yes', 'y']:
            pass
        else:
            print("Exiting Now!")
            break
