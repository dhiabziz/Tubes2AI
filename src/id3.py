from collections import Counter
import time
import numpy as np

class Node:
    """
    A class used to represent a node in a decision tree.

    Attributes
    ----------
    feature : int or str, optional
        The feature to split on (default is None).
    value : any, optional
        The value of the feature to split on (default is None).
    results : dict, optional
        Stores class labels if the node is a leaf node (default is None).
    true_branch : Node, optional
        The branch for values that are True for the feature (default is None).
    false_branch : Node, optional
        The branch for values that are False for the feature (default is None).

    Methods
    -------
    __init__(self, feature=None, value=None, results=None, true_branch=None, false_branch=None)
        Initializes the Node with the given attributes.
    """
    def __init__(self, feature: int = None, value: any = None, results: dict = None, true_branch = None, false_branch = None):
        self.feature = feature  # Feature to split on
        self.value = value      # Value of the feature to split on
        self.results = results  # Stores class labels if node is a leaf node
        self.true_branch = true_branch  # Branch for values that are True for the feature
        self.false_branch = false_branch  # Branch for values that are False for the feature
        
        
def entropy(data: np.ndarray) -> float:
    """
    Calculate the entropy of a dataset.

    Entropy is a measure of the amount of uncertainty or randomness in the data.
    It is calculated using the formula:
    
        entropy = -sum(p * log2(p) for p in probabilities if p > 0)
    
    where `p` is the probability of each unique value in the dataset.

    Parameters:
    data (array-like): The dataset for which to calculate the entropy. It should be a 1-dimensional array-like object containing discrete values.

    Returns:
    float: The entropy of the dataset.
    """
    counts = np.bincount(data)
    probabilities = counts / len(data)
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy

def split_data(X: np.ndarray, y: np.ndarray, feature: int, value: float) -> tuple:
    """
    Splits the dataset into two subsets based on a feature and a threshold value.

    Parameters:
    X (numpy.ndarray): The feature matrix of shape (n_samples, n_features).
    y (numpy.ndarray): The target vector of shape (n_samples,).
    feature (int): The index of the feature to split on.
    value (float): The threshold value to split the feature on.

    Returns:
    tuple: A tuple containing four elements:
        - true_X (numpy.ndarray): The subset of X where the feature value is less than or equal to the threshold.
        - true_y (numpy.ndarray): The subset of y corresponding to true_X.
        - false_X (numpy.ndarray): The subset of X where the feature value is greater than the threshold.
        - false_y (numpy.ndarray): The subset of y corresponding to false_X.
    """
    true_indices = np.where(X[:, feature] <= value)[0]
    false_indices = np.where(X[:, feature] > value)[0]
    true_X, true_y = X[true_indices], y[true_indices]
    false_X, false_y = X[false_indices], y[false_indices]
    return true_X, true_y, false_X, false_y

def build_tree(X: np.ndarray, y: np.ndarray, max_depth: int = 10, min_samples_split: int = 2) -> Node:
    """
    Optimized decision tree construction using ID3 algorithm.
    
    Parameters:
    X (numpy.ndarray): The feature matrix.
    y (numpy.ndarray): The target values.
    max_depth (int): Maximum depth of the tree to prevent overfitting.
    min_samples_split (int): Minimum number of samples required to split a node.
    
    Returns:
    Node: The root node of the constructed decision tree.
    """
    # Early stopping conditions
    if (len(set(y)) == 1 or  # All samples have the same class
        max_depth == 0 or    # Maximum depth reached
        len(y) < min_samples_split):  # Not enough samples to split
        # Return the most frequent class
        unique, counts = np.unique(y, return_counts=True)
        return Node(results=unique[np.argmax(counts)])

    # Calculate current entropy
    current_entropy = entropy(y)

    # Track best split
    best_gain = 0
    best_criteria = None
    best_sets = None
    n_features = X.shape[1]

    # Randomly sample features to reduce computation
    feature_subset = np.random.choice(n_features, min(20, n_features), replace=False)

    for feature in feature_subset:
        # Use percentiles for feature values to reduce computational complexity
        feature_values = np.percentile(X[:, feature], [25, 50, 75])
        
        for value in feature_values:
            # Split data
            true_X, true_y, false_X, false_y = split_data(X, y, feature, value)
            
            # Skip if split is too small
            if (len(true_y) < min_samples_split or 
                len(false_y) < min_samples_split):
                continue

            # Calculate entropy for split branches
            true_entropy = entropy(true_y)
            false_entropy = entropy(false_y)
            
            # Calculate information gain
            p = len(true_y) / len(y)
            gain = current_entropy - p * true_entropy - (1 - p) * false_entropy

            # Update best split if gain is improved
            if gain > best_gain:
                best_gain = gain
                best_criteria = (feature, value)
                best_sets = (true_X, true_y, false_X, false_y)

    # If no good split found, return most frequent class
    if best_gain <= 0:
        unique, counts = np.unique(y, return_counts=True)
        return Node(results=unique[np.argmax(counts)])

    # Recursively build branches with reduced depth
    true_branch = build_tree(best_sets[0], best_sets[1], max_depth - 1, min_samples_split)
    false_branch = build_tree(best_sets[2], best_sets[3], max_depth - 1, min_samples_split)
    
    return Node(
        feature=best_criteria[0], 
        value=best_criteria[1], 
        true_branch=true_branch, 
        false_branch=false_branch
    )

def single_instance_predict(tree: Node, sample: dict) -> dict:
    """
    Predicts the class label for a given sample using a decision tree.

    Args:
        tree (DecisionNode): The root node of the decision tree.
        sample (dict): A dictionary representing the sample to classify, 
                       where keys are feature names and values are feature values.

    Returns:
        dict: The predicted class label(s) with their respective probabilities.
    """
    if tree.results is not None:
        return tree.results
    else:
        branch = tree.false_branch
        if sample[tree.feature] <= tree.value:
            branch = tree.true_branch
        return single_instance_predict(branch, sample)

def predict(tree: Node, samples: np.ndarray) -> np.ndarray:
    """
    Predicts the class labels for a given set of samples using a decision tree.

    Args:
        tree (DecisionNode): The root node of the decision tree.
        samples (numpy.ndarray): The samples to classify, where rows are instances and columns are features.

    Returns:
        numpy.ndarray: The predicted class labels for the samples.
    """
    if samples.ndim == 1:
        return single_instance_predict(tree, samples)
    else:
        return np.array([single_instance_predict(tree, sample) for sample in samples])
    
# Example on how to use with single prediction
X = np.array([
    [2.7, 2.5],
    [1.3, 3.1],
    [3.1, 1.8],
    [3.8, 2.7],
    [2.5, 2.3],
    [1.5, 2.8],
    [3.2, 3.0],
    [2.0, 1.5],
    [3.0, 2.0],
    [2.2, 2.9]
])
y = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])

# Building the tree
decision_tree = build_tree(X, y)

# Making a prediction
samples = np.array([[2.7, 2.5],
    [1.3, 3.1],
    [3.1, 1.8]])
prediction = predict(decision_tree, samples)
print(f"Prediction for sample {samples}: {prediction}")



start = time.time()
np.random.seed(42)
X = np.random.randn(100000, 30)  # 100 samples, 4 features
y = np.random.choice([0, 1], size=150000)  # Binary target (0 or 1)

# Train-test split
train_size = 0.7
split = int(len(X) * train_size)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Example on how to use with multiple predictions (maybe i will make a different function for handling multiple predictions)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the decision tree
print("Building decision tree...")
decision_tree = build_tree(X_train, y_train)

print("Decision tree built.")
print("Making predictions...")
# Make predictions on the test set
y_pred = predict(decision_tree, X_test)
print("Predictions made.")
# Calculate the accuracy
# accuracy = accuracy_score(y_test, y_pred) # Error
# print(f"Accuracy: {accuracy:.4f}")

execution_time = time.time() - start
print(f"Execution time: {execution_time:.2f} seconds")