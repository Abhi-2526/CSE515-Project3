import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import utils
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import classification_report


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
from sklearn.tree import DecisionTreeClassifier
class ImageClassifier: 

    def __init__(self, **dt_params): 
        self.dt_params = dt_params 
        self.classifier = None
        self.scaler = StandardScaler() 

    def train(self,features,labels):
        features = self.scaler.fit_transform(features) 
        self.classifier = DecisionTreeClassifier(**self.dt_params) 
        self.classifier.fit(features,labels) 

    def predict_and_evaluate(self,features,true_labels):
        features = self.scaler.transform(features)
        predicted_labels = self.classifier.predict(features)
        # Print predictions for each odd image

        for image_id, label in enumerate(predicted_labels, start=1):
            if image_id % 2 != 0:  # Check if the image ID is odd
                print(f"Odd Image ID {image_id}: Predicted Label - {label}")
        # Evaluate performance

        accuracy = accuracy_score(true_labels, predicted_labels)
        class_report = classification_report(true_labels, predicted_labels)
        # Print overall accuracy
        print("Overall Accuracy:", accuracy)
        # Print per-label precision, recall, and F1-score
        print("\nClassification Report:")
        print(class_report)

        # f1 = f1_score(true_labels, predicted_labels, average='weighted')
        # recall = recall_score(true_labels, predicted_labels, average='weighted')
        # return {"accuracy": accuracy, "f1_score": f1, "recall": recall}



def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

even_features, even_labels = utils.load_even_images_and_labels('caltech101_features.db')

# Standardize features
scaler = StandardScaler()
X_even = scaler.fit_transform(even_features)
y_even = np.array(even_labels)

# User input for classifier selection
classifier_choice = input("1) k-NN\n2) Decision Tree\n3) PPR\n\nSelect classifier: ").strip()
flag = 0 
# Classifier initialization
if classifier_choice == '1':
    k = int(input("Enter the number of neighbors for k-NN: "))
    classifier = CustomKNN(k)
    classifier.fit(X_even, y_even)
    flag = 1 
elif classifier_choice == '2':
    classifier = ImageClassifier(max_depth = 10)
    classifier.train(even_features,even_labels)
    flag = 2 
else:
    print("Invalid classifier choice.")
    exit()

# Load odd features and labels
odd_features, odd_labels = utils.load_odd_images_and_labels('caltech101_features.db')

# Standardize odd features
X_odd = scaler.transform(odd_features)
y_odd = np.array(odd_labels)
if flag ==2 : 
    odd_labels = np.array(odd_labels) 
    # evaluation_results = classifier.predict_and_evaluate(odd_features, odd_labels)
    # print(evaluation_results)
    classifier.predict_and_evaluate(odd_features, odd_labels)
# Predict labels for odd-numbered images
if flag ==1 :
    predicted_labels = classifier.predict(X_odd)

    # Print predictions for each odd image
    odd_image_ids = np.arange(1, len(odd_features) * 2, 2)  # Assuming odd indices represent odd-numbered images
    for image_id, label in zip(odd_image_ids, predicted_labels):
        print(f"Odd Image ID {image_id}: Label - {label}")

    # Classification report and accuracy
    print("\nClassification Report for Odd-Numbered Images:")
    print(classification_report(y_odd, predicted_labels, labels=np.unique(y_odd)))
    print("Overall Accuracy:", accuracy_score(y_odd, predicted_labels))
