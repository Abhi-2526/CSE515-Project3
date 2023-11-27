import numpy as np
import utils
import matplotlib.pyplot as plt
import subprocess
from torchvision.datasets import Caltech101


class LSH:
    def __init__(self, L, h, dimension):
        self.L = L
        self.h = h
        self.dimension = dimension
        self.hash_tables = [{} for _ in range(L)]
        self._generate_hash_functions()

    def _generate_hash_functions(self):
        self.hash_functions = []
        for _ in range(self.L):
            random_vectors = np.random.randn(self.h, self.dimension)
            self.hash_functions.append(random_vectors)

    def _hash(self, vector, hash_function):
        return tuple(np.dot(hash_function, vector) > 0)

    def index(self, vectors):
        for idx, vector in enumerate(vectors):
            for l, hash_function in enumerate(self.hash_functions):
                hash_key = self._hash(vector, hash_function)
                if hash_key not in self.hash_tables[l]:
                    self.hash_tables[l][hash_key] = []
                self.hash_tables[l][hash_key].append(idx)

    def print_hash_tables(self):
        for l, hash_table in enumerate(self.hash_tables):
            print(f"Layer {l + 1}:")
            for hash_key, indices in hash_table.items():
                print(f"  Hash Key: {hash_key} -> Indices: {indices}")
            print("\n")

    def query(self, query_vector):
        candidates = set()
        for l, hash_function in enumerate(self.hash_functions):
            hash_key = self._hash(query_vector, hash_function)
            if hash_key in self.hash_tables[l]:
                candidates.update(self.hash_tables[l][hash_key])
        return list(candidates)

    def find_similar_images(self, query_vector, vectors, labels, query_label, t):
        candidates = self.query(query_vector)
        #candidates_sl = [idx for idx in candidates if labels[idx] == query_label]
        sorted_candidates = sorted(candidates, key=lambda idx: np.linalg.norm(vectors[idx] - query_vector))
        unique_considered = len(set(candidates))
        overall_considered = sum(len(self.hash_tables[l][self._hash(query_vector, self.hash_functions[l])])
                                 for l in range(self.L)
                                 if self._hash(query_vector, self.hash_functions[l]) in self.hash_tables[l])
        return [(idx, labels[idx]) for idx in sorted_candidates[:t]], unique_considered, overall_considered


def retrieve_image(dataset, image_id):
    return dataset[image_id][0]


def visualize_images(query_image_id, similar_image_indices_and_labels, dataset):
    num_images = len(similar_image_indices_and_labels) + 1
    fig, axes = plt.subplots(1, num_images, figsize=(20, 10))

    query_image = retrieve_image(dataset, query_image_id)
    axes[0].imshow(query_image)
    axes[0].axis('off')
    axes[0].set_title(f'Query ID: {query_image_id}')

    for ax, (idx, label) in zip(axes[1:], similar_image_indices_and_labels):
        image = retrieve_image(dataset, idx)
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f'ID: {idx}')

    plt.show()


def collect_user_feedback(similar_image_indices_and_labels, dataset):
    feedback = []
    for idx, label in similar_image_indices_and_labels:
        image = retrieve_image(dataset, idx)
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        user_input = input(f"Rate image ID {idx} with label {label} (R+, R, I, I-): ").strip().lower()
        if user_input in ['r+', 'r', 'i', 'i-']:
            feedback.append((idx, label, user_input))
        else:
            print("Invalid input. Skipping this image.")
    return feedback


def main():
    dataset = Caltech101(root="./data", download=True)
    db_path = 'caltech101_features.db'
    even_vectors, even_labels = utils.load_features_and_labels_from_db(db_path)

    L = int(input("Enter the number of layers (L): "))
    h = int(input("Enter the number of hashes per layer (h): "))
    dimension = even_vectors[0].shape[0]

    lsh = LSH(L, h, dimension)
    lsh.index(even_vectors)

    print("LSH Index Structure:")
    lsh.print_hash_tables()

    proceed = input("Do you want to proceed with Task 4b (Similar Image Search)? (yes/no): ").strip().lower()
    if proceed.lower() in ['yes', 'y']:
        query_image_id = int(input("Enter the ID of the query image: "))
        query_vector = even_vectors[query_image_id]
        query_label = even_labels[query_image_id]
        t = int(input("Enter the number of similar images to retrieve (t): "))

        similar_image_indices_and_labels, unique_considered, overall_considered = lsh.find_similar_images(query_vector, even_vectors, even_labels, query_label, t)

        print("Query Image ID:", query_image_id)
        print("Similar Image Indices and Labels:", similar_image_indices_and_labels)
        print("Unique images considered:", unique_considered)
        print("Overall images considered:", overall_considered)

        visualize_images(query_image_id, similar_image_indices_and_labels, dataset)
    else:
        print("Exiting Now!")
        exit()

    start_task5 = input("Do you want to start Task 5 (Relevance Feedback)? (yes/no): ")
    if start_task5.lower() in ['yes', 'y']:
        feedback = collect_user_feedback(similar_image_indices_and_labels, dataset)
        with open('feedback_data.txt', 'w') as f:
            for idx, label, rating in feedback:
                f.write(f"{idx} {label} {rating}\n")
        subprocess.run(["python", "task5.py", str(query_image_id), str(query_label), str(t)])
    else:
        print("Exiting Now!")
