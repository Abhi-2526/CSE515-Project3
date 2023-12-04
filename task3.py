import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import utils
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore")


class PersonalizedPageRankClassifier:
    def __init__(self, db_path, alpha=0.85, max_iter=100, tol=1e-6):
        self.db_path = db_path
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = self.get_alpha_input()

    def get_alpha_input(self):
        while True:
            try:
                alpha = float(input("Enter the random jump probability (alpha) for PPR (between 0 and 1): "))
                if 0 <= alpha <= 1:
                    return alpha
                else:
                    print("Alpha must be between 0 and 1.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def compute_similarity_graph(self, features):
        similarity = cosine_similarity(features)
        similarity = np.nan_to_num(similarity)
        column_sums = similarity.sum(axis=0)
        column_sums[column_sums == 0] = 1
        similarity = similarity / column_sums
        return similarity

    def compute_restart_vector(self, odd_feature, even_features):
        similarities = cosine_similarity([odd_feature], even_features)[0]
        restart_vector = similarities / np.sum(similarities)
        return restart_vector

    def personalized_page_rank(self, similarity_graph, restart_vector):
        n = similarity_graph.shape[0]
        ppr = np.copy(restart_vector)
        dangling_weights = np.where(np.sum(similarity_graph, axis=0) == 0, 1.0 / n, 0)
        for _ in range(self.max_iter):
            prev_ppr = np.copy(ppr)
            ppr = self.alpha * (np.dot(similarity_graph, ppr) + dangling_weights @ ppr) + (
                        1 - self.alpha) * restart_vector
            if np.linalg.norm(ppr - prev_ppr, 1) < self.tol:
                break
            ppr = np.nan_to_num(ppr)
        return ppr

    def run(self):
        with open(f"PPROutput{self.alpha}.txt", 'w') as f:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                warnings.simplefilter("ignore", category=UndefinedMetricWarning)

            even_features, even_labels = utils.load_even_images_and_labels(self.db_path)
            odd_features, odd_labels = utils.load_odd_images_and_labels(self.db_path)
            # odd_features = odd_features[600:610]
            # odd_labels = odd_labels[600:610]

            similarity_graph = self.compute_similarity_graph(even_features)
            predicted_labels = []

            for index, odd_feature in enumerate(odd_features):
                restart_vector = self.compute_restart_vector(odd_feature, even_features)
                ppr_scores = self.personalized_page_rank(similarity_graph, restart_vector)
                top_index = np.argmax(ppr_scores)
                predicted_label = even_labels[top_index]
                predicted_labels.append(predicted_label)
                # Generate odd image IDs
                odd_image_ids = np.arange(1, len(odd_features) * 2, 2)
                for image_id, label in zip(odd_image_ids, predicted_labels):
                    output = f"Odd Image ID {image_id}: Predicted Label - {label}"
                    print(output)
                    f.write(output + '\n')

            output = "\nClassification Report:"
            output += classification_report(odd_labels, predicted_labels, zero_division=0)
            print(output)
            f.write(output)

            # precision, recall, fscore, _ = precision_recall_fscore_support(odd_labels, predicted_labels, average='micro', zero_division=0)
            accuracy = accuracy_score(odd_labels, predicted_labels)
            output = f'accuracy = {accuracy}'
            print(output) 
            f.write(output)


class CustomKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [np.sqrt(np.sum((x_train - x) ** 2)) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class CustomDecisionTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        if y.dtype.kind in 'UO':
            unique_labels, y = np.unique(y, return_inverse=True)
            self.label_dict = {i: label for i, label in enumerate(unique_labels)}
        else:
            self.label_dict = None

        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == c) for c in np.unique(y)]
        predicted_class = np.argmax(num_samples_per_class)

        # stopping criteria
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return Node(value=predicted_class)

        feature_idxs = np.arange(X.shape[1])
        best_feature, best_thresh = self._best_criteria(X, y, feature_idxs)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left_child = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left_child, right_child)

    def _best_criteria(self, X, y, feature_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feature_idx in feature_idxs:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        # parent loss
        parent_entropy = entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _predict(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict(x, node.left)
        return self._predict(x, node.right)


def train_decision_tree(features, labels, **dt_params):
    from sklearn.tree import DecisionTreeClassifier
    decision_tree = DecisionTreeClassifier(**dt_params)
    decision_tree.fit(features, labels)
    return decision_tree


def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


def main():
    even_features, even_labels = utils.load_even_images_and_labels('caltech101_features.db')

    scaler = StandardScaler()
    X_even = scaler.fit_transform(even_features)
    y_even = np.array(even_labels)

    # User input for classifier selection
    classifier_choice = input("1) k-NN\n2) Decision Tree\n3) PPR\n\nSelect classifier: ").strip()

    # Classifier initialization
    if classifier_choice == '1':
        k = int(input("Enter the number of neighbors for k-NN: "))
        classifier = CustomKNN(k)
        classifier.fit(X_even, y_even)
    elif classifier_choice == '2':
        classifier = train_decision_tree(X_even, y_even)
    elif classifier_choice == '3':
        db_path = "caltech101_features.db"
        classifier = PersonalizedPageRankClassifier(db_path)
        classifier.run()
        exit()
    else:
        print("Invalid classifier choice.")
        exit()

    # Load odd features and labels
    odd_features, odd_labels = utils.load_odd_images_and_labels('caltech101_features.db')

    # Standardize odd features
    X_odd = scaler.transform(odd_features)
    y_odd = np.array(odd_labels)

    # Predict labels for odd-numbered images
    predicted_labels = classifier.predict(X_odd)

    # Print predictions for each odd image
    odd_image_ids = np.arange(1, len(odd_features) * 2, 2)
    for image_id, label in zip(odd_image_ids, predicted_labels):
        print(f"Odd Image ID {image_id}: Label - {label}")

    y_odd = y_odd.astype(int)
    predicted_labels = predicted_labels.astype(int)

    # Sort labels numerically for consistent reporting
    sorted_labels = np.unique(np.concatenate((y_odd, predicted_labels)))

    # Classification report and accuracy
    print("\nClassification Report for Odd-Numbered Images:")
    print(classification_report(y_odd, predicted_labels, labels=sorted_labels))
    print("Overall Accuracy:", accuracy_score(y_odd, predicted_labels))


if __name__ == "__main__":
    main()