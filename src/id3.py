from collections import Counter
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
    def __init__(self, feature=None, value=None, results=None, true_branch=None, false_branch=None):
        self.feature = feature  # Feature to split on
        self.value = value      # Value of the feature to split on
        self.results = results  # Stores class labels if node is a leaf node
        self.true_branch = true_branch  # Branch for values that are True for the feature
        self.false_branch = false_branch  # Branch for values that are False for the feature
        
def entropy(data):
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

def split_data(X, y, feature, value):
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

def build_tree(X, y):
    """
    Builds a decision tree using the ID3 algorithm.
    
    Parameters:
    X (numpy.ndarray): The feature matrix where each row represents an instance and each column represents a feature.
    y (numpy.ndarray): The target values corresponding to each instance in X.
    
    Returns:
    Node: The root node of the constructed decision tree.
    The function works by recursively splitting the data based on the feature that provides the highest information gain 
    until all instances in a node belong to the same class or no further information gain can be achieved.
    """
    if len(set(y)) == 1:
        return Node(results=y[0])

    best_gain = 0
    best_criteria = None
    best_sets = None
    n_features = X.shape[1]

    current_entropy = entropy(y)

    for feature in range(n_features):
        feature_values = set(X[:, feature])
        for value in feature_values:
            true_X, true_y, false_X, false_y = split_data(X, y, feature, value)
            true_entropy = entropy(true_y)
            false_entropy = entropy(false_y)
            p = len(true_y) / len(y)
            gain = current_entropy - p * true_entropy - (1 - p) * false_entropy

            if gain > best_gain:
                best_gain = gain
                best_criteria = (feature, value)
                best_sets = (true_X, true_y, false_X, false_y)

    if best_gain > 0:
        true_branch = build_tree(best_sets[0], best_sets[1])
        false_branch = build_tree(best_sets[2], best_sets[3])
        return Node(feature=best_criteria[0], value=best_criteria[1], true_branch=true_branch, false_branch=false_branch)

    return Node(results=y[0])

def predict(tree, sample):
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
        return predict(branch, sample)
    
# Unit Testing Example
X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
y = np.array([1, 1, 0, 0])

# Building the tree
decision_tree = build_tree(X, y)

# Displaying the tree visually
print("Decision Tree:")
for i in range(2):
    print(f"Level {i}:")
    print(f"Feature: {decision_tree.feature}")
    print(f"Value: {decision_tree.value}")
    decision_tree = decision_tree.true_branch
    